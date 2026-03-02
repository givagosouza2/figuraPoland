# main.py
import io
import re
import numpy as np
import polars as pl
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sincronização Cinêmica + Giroscópio (sem SciPy/pandas)", layout="wide")
st.title("🧭 Sincronização: Cinêmica (120 Hz) + Giroscópio (→100 Hz, detrend, LPF 1.5 Hz, norma) — sem SciPy/pandas")

# =========================================================
# Leitura robusta (polars) — CORRIGIDA (não quebra CSV com ',')
# =========================================================
def _bytes_to_text(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    return raw.decode("utf-8", errors="replace")


def _fix_decimal_comma(text: str) -> str:
    # Converte decimal "1,23" -> "1.23" sem mexer no separador quando ele é ','
    return re.sub(r"(?<=\d),(?=\d)", ".", text)


def _read_flexible_table(uploaded_file) -> pl.DataFrame:
    text_raw = _bytes_to_text(uploaded_file)

    # 1) Tenta ler "como está" (não mexe em vírgulas: essencial quando sep=',')
    for sep in [",", ";", "\t"]:
        try:
            df = pl.read_csv(
                io.StringIO(text_raw),
                separator=sep,
                has_header=True,
                ignore_errors=True,
                infer_schema_length=2000,
            )
            if df.width >= 3 and df.height > 0:
                return df
        except Exception:
            pass

    # 2) Se falhou, tenta corrigir decimal vírgula (caso sep=';' ou '\t')
    text_fixed = _fix_decimal_comma(text_raw)
    for sep in [";", "\t", ","]:
        try:
            df = pl.read_csv(
                io.StringIO(text_fixed),
                separator=sep,
                has_header=True,
                ignore_errors=True,
                infer_schema_length=2000,
            )
            if df.width >= 3 and df.height > 0:
                return df
        except Exception:
            pass

    # 3) fallback whitespace
    text_ws = re.sub(r"[ \t]+", " ", text_raw.strip())
    df = pl.read_csv(
        io.StringIO(text_ws),
        separator=" ",
        has_header=True,
        ignore_errors=True,
        infer_schema_length=2000,
    )
    if df.width >= 3 and df.height > 0:
        return df

    raise ValueError("Não consegui ler a tabela com separadores comuns.")


def _numeric_columns(df: pl.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        try:
            s2 = df[c].cast(pl.Float64, strict=False)
            non_null = s2.drop_nulls().len()
            if non_null >= max(5, int(0.5 * df.height)):
                cols.append(c)
        except Exception:
            pass
    return cols


def _to_numpy_numeric(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    out = []
    for c in cols:
        out.append(df[c].cast(pl.Float64, strict=False).to_numpy())
    return np.column_stack(out)


# =========================================================
# Processamento sem SciPy (detrend + filtro LPF 1.5 Hz)
# =========================================================
def infer_time_unit(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) < 2:
        return 1.0
    span = t.max() - t.min()
    dt_med = np.median(np.diff(t))
    if span > 1e5:  # microsegundos
        return 1e-6
    if span > 1e2 and dt_med > 0.5:  # milissegundos
        return 1e-3
    return 1.0


def resample_to_fs(t_s: np.ndarray, xyz: np.ndarray, fs_target: float) -> tuple[np.ndarray, np.ndarray]:
    t_s = np.asarray(t_s, dtype=float)
    xyz = np.asarray(xyz, dtype=float)

    order = np.argsort(t_s)
    t_s = t_s[order]
    xyz = xyz[order]

    keep = np.ones_like(t_s, dtype=bool)
    keep[1:] = np.diff(t_s) > 0
    t_s = t_s[keep]
    xyz = xyz[keep]

    t0, t1 = t_s[0], t_s[-1]
    dt = 1.0 / fs_target
    t_new = np.arange(t0, t1 + 0.5 * dt, dt)

    xyz_new = np.column_stack([np.interp(t_new, t_s, xyz[:, i]) for i in range(3)])
    return t_new, xyz_new


def detrend_linear(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    A = np.column_stack([t, np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    trend = A @ coef
    return y - trend


def lowpass_iir_1st(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    dt = 1.0 / fs
    RC = 1.0 / (2.0 * np.pi * fc)
    alpha = dt / (RC + dt)

    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
    return y


def zero_phase_lowpass(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    y_f = lowpass_iir_1st(x, fs, fc)
    y_b = lowpass_iir_1st(y_f[::-1], fs, fc)[::-1]
    return y_b


def preprocess_gyro(df: pl.DataFrame, fs_target=100.0, fc=1.5) -> dict:
    num_cols = _numeric_columns(df)
    if len(num_cols) < 4:
        raise ValueError(f"Giroscópio: esperado >=4 colunas numéricas (t,x,y,z). Achei {len(num_cols)}.")

    arr = _to_numpy_numeric(df, num_cols[:4])
    t = arr[:, 0]
    g = arr[:, 1:4]

    factor = infer_time_unit(t)
    t_s = (t - t[0]) * factor

    t100, g100 = resample_to_fs(t_s, g, fs_target)

    g_dt = np.column_stack([detrend_linear(g100[:, i], t100) for i in range(3)])
    g_f = np.column_stack([zero_phase_lowpass(g_dt[:, i], fs_target, fc) for i in range(3)])

    norm = np.sqrt(np.sum(g_f**2, axis=1))
    return {"t": t100, "gx": g_f[:, 0], "gy": g_f[:, 1], "gz": g_f[:, 2], "norm": norm, "fs": fs_target}


def preprocess_kinematic(df: pl.DataFrame, fs=80.0) -> dict:
    num_cols = _numeric_columns(df)
    if len(num_cols) < 3:
        raise ValueError(f"Cinêmica: esperado >=3 colunas numéricas (X,Y,Z). Achei {len(num_cols)}.")
    arr = _to_numpy_numeric(df, num_cols[:3])
    n = arr.shape[0]
    t = np.arange(n, dtype=float) / fs
    return {"t": t, "x": arr[:, 0], "y": arr[:, 1], "z": arr[:, 2], "fs": fs}


def suggest_trigger_time(t: np.ndarray, sig: np.ndarray) -> float:
    sig = np.asarray(sig, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(sig) < 5:
        return float(t[0]) if len(t) else 0.0
    d = np.abs(np.diff(sig))
    idx = int(np.argmax(d))
    return float(t[min(idx + 1, len(t) - 1)])


# =========================================================
# UI: Upload
# =========================================================
colA, colB = st.columns(2)
with colA:
    up_kin = st.file_uploader("📄 Arquivo Cinêmica (X,Y,Z) — 120 Hz", type=["csv", "txt"])
with colB:
    up_gyr = st.file_uploader("📄 Arquivo Giroscópio (t,x,y,z)", type=["csv", "txt"])

if not up_kin or not up_gyr:
    st.info("Carregue os dois arquivos para começar.")
    st.stop()

# Leitura + preprocess
try:
    df_kin = _read_flexible_table(up_kin)
    kin = preprocess_kinematic(df_kin, fs=100.0)
except Exception as e:
    st.error(f"Erro na cinêmica: {e}")
    st.stop()

try:
    df_gyr = _read_flexible_table(up_gyr)
    gyr = preprocess_gyro(df_gyr, fs_target=100.0, fc=1.5)
except Exception as e:
    st.error(f"Erro no giroscópio: {e}")
    st.stop()

# Sugestões de trigger
t0_kin_sug = suggest_trigger_time(kin["t"], kin["z"])   # salto no Z
t0_gyr_sug = suggest_trigger_time(gyr["t"], gyr["gy"])  # salto no Y

# =========================================================
# Controles
# =========================================================
st.subheader("1) Definir o trigger (tempo zero) em cada sinal")
c1, c2, c3 = st.columns([1.2, 1.2, 1.6])

with c1:
    kin_axis = st.selectbox("Cinêmica para plot (eixo)", ["y", "z"], index=1)

with c2:
    invert_kin = st.checkbox("Inverter cinêmica (× -1)", value=False)
    invert_gyr_y = st.checkbox("Inverter giroscópio Y (× -1)", value=False)

with c3:
    st.caption("Ajuste os triggers até alinhar os saltos em t=0.")

t_kin_min, t_kin_max = float(kin["t"][0]), float(kin["t"][-1])
t_gyr_min, t_gyr_max = float(gyr["t"][0]), float(gyr["t"][-1])

t0_kin = st.slider(
    "Trigger na Cinêmica (s) — referência: salto no Z",
    min_value=t_kin_min, max_value=t_kin_max,
    value=float(np.clip(t0_kin_sug, t_kin_min, t_kin_max)),
    step=1.0 / 120.0
)

t0_gyr = st.slider(
    "Trigger no Giroscópio (s) — referência: salto no Y",
    min_value=t_gyr_min, max_value=t_gyr_max,
    value=float(np.clip(t0_gyr_sug, t_gyr_min, t_gyr_max)),
    step=1.0 / 100.0
)

kin_sig = kin[kin_axis].copy()
if invert_kin:
    kin_sig = -kin_sig

gyr_y = gyr["gy"].copy()
if invert_gyr_y:
    gyr_y = -gyr_y

tkin_sync = kin["t"] - t0_kin
tgyr_sync = gyr["t"] - t0_gyr

# =========================================================
# Janela temporal
# =========================================================
st.subheader("2) Selecionar janela temporal de visualização")
tmin_common = max(float(tkin_sync.min()), float(tgyr_sync.min()))
tmax_common = min(float(tkin_sync.max()), float(tgyr_sync.max()))
if tmin_common >= tmax_common:
    st.error("Não há sobreposição temporal entre os sinais após os triggers escolhidos.")
    st.stop()

win = st.slider(
    "Janela (s) no tempo sincronizado",
    min_value=float(tmin_common),
    max_value=float(tmax_common),
    value=(max(-2.0, float(tmin_common)), min(5.0, float(tmax_common))),
    step=0.01
)
t_start, t_end = win

mask_kin = (tkin_sync >= t_start) & (tkin_sync <= t_end)
mask_gyr = (tgyr_sync >= t_start) & (tgyr_sync <= t_end)

# =========================================================
# Plot duplo
# =========================================================
st.subheader("3) Plot duplo (cinêmica vs norma do giroscópio)")
fig, ax1 = plt.subplots()

ax1.plot(tkin_sync[mask_kin], kin_sig[mask_kin]/1000,'-k')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel(f"Antero-posterior displacement (m)")
ax1.axvline(0, linestyle="--", linewidth=1)
ax1.set_xlim(t_start, t_end)

ax2 = ax1.twinx()
ax2.plot(tgyr_sync[mask_gyr], gyr["norm"][mask_gyr])
ax2.set_ylabel(" Angular velocity (rad/s)")

st.pyplot(fig, use_container_width=True)

with st.expander("🔎 Diagnóstico: sinais usados para trigger (Z cinêmica e Y giroscópio)"):
    fig2, bx1 = plt.subplots()

    z_ref = kin["z"].copy()
    if invert_kin:
        z_ref = -z_ref
    bx1.plot(tkin_sync[mask_kin], z_ref[mask_kin],'-k')
    bx1.set_xlabel("Tempo sincronizado (s)")
    bx1.set_ylabel("Cinêmica Z (ref. salto)")
    bx1.axvline(0, linestyle="--", linewidth=1)

    bx2 = bx1.twinx()
    bx2.plot(tgyr_sync[mask_gyr], gyr_y[mask_gyr])
    bx2.set_ylabel("Giroscópio Y (ref. salto)")
    

    st.pyplot(fig2, use_container_width=True)

st.success("App rodando sem SciPy/pandas (compatível com Python 3.13 no Streamlit Cloud).")
