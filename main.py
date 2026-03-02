# app.py
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import detrend, butter, filtfilt

st.set_page_config(page_title="Sincronização Cinêmática + Giroscópio", layout="wide")
st.title("🧭 Sincronização: Cinêmica (120 Hz) + Giroscópio (→100 Hz, detrend, LPF 1.5 Hz, norma)")

# -----------------------------
# Utils de leitura
# -----------------------------
def _read_flexible_csv(uploaded_file) -> pd.DataFrame:
    """Lê CSV/TXT com separador desconhecido e decimal com vírgula, tentando ser robusto."""
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8", errors="replace")

    # troca decimal vírgula por ponto quando parece número
    # (não é perfeito, mas ajuda muito em dados BR)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)

    # tenta sniff de separador: ;, \t, , ou whitespace
    # pandas com engine=python ajuda com separador regex
    for sep in [";", "\t", ","]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
            if df.shape[1] >= 3:
                return df
        except Exception:
            pass

    # fallback: whitespace
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python")
    return df


def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(how="all")
    return out


# -----------------------------
# Processamento giroscópio
# -----------------------------
def infer_time_unit(t: np.ndarray) -> float:
    """
    Retorna fator para converter t para segundos.
    Heurística:
      - se valores parecem em ms (ex.: > 1000 e duração grande), divide por 1000
      - se parecem em us (muito grandes), divide por 1e6
      - senão, assume segundos
    """
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) < 2:
        return 1.0
    dt_med = np.median(np.diff(t))
    # Se dt típico ~ 10 (ms) ou ~ 10000 (us) etc.
    # Mas melhor olhar amplitude:
    span = t.max() - t.min()
    if span > 1e5:   # muito grande -> provavelmente us
        return 1e-6
    if span > 1e2 and dt_med > 0.5:  # pode ser ms (ex: dt ~10)
        return 1e-3
    # caso comum: já em segundos
    return 1.0


def resample_to_fs(t_s: np.ndarray, xyz: np.ndarray, fs_target: float) -> tuple[np.ndarray, np.ndarray]:
    """Reamostra por interpolação linear para grade uniforme."""
    t_s = np.asarray(t_s, dtype=float)
    xyz = np.asarray(xyz, dtype=float)

    # ordena por tempo, remove repetidos
    order = np.argsort(t_s)
    t_s = t_s[order]
    xyz = xyz[order, :]

    # remove tempos iguais (keep first)
    uniq_mask = np.ones_like(t_s, dtype=bool)
    uniq_mask[1:] = np.diff(t_s) > 0
    t_s = t_s[uniq_mask]
    xyz = xyz[uniq_mask, :]

    t0, t1 = t_s[0], t_s[-1]
    dt = 1.0 / fs_target
    t_new = np.arange(t0, t1 + 0.5 * dt, dt)

    xyz_new = np.column_stack([
        np.interp(t_new, t_s, xyz[:, i]) for i in range(3)
    ])
    return t_new, xyz_new


def lowpass_filter(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    """Butterworth passa-baixa + filtfilt."""
    nyq = 0.5 * fs
    wn = fc / nyq
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)


def preprocess_gyro(df: pd.DataFrame, fs_target=100.0, fc=1.5) -> dict:
    """
    df: colunas [tempo, x, y, z] (nomes podem variar)
    retorna dict com t (s), gx, gy, gz, norm, etc.
    """
    df = _coerce_numeric_df(df)

    # pega as 4 primeiras colunas numéricas
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 4:
        raise ValueError(f"Giroscópio: esperado >= 4 colunas numéricas (t,x,y,z). Achei {len(num_cols)}.")

    t = df[num_cols[0]].to_numpy(dtype=float)
    g = df[num_cols[1:4]].to_numpy(dtype=float)

    # normaliza tempo para segundos
    factor = infer_time_unit(t)
    t_s = (t - t[0]) * factor  # zera no início do arquivo

    # interpola para 100 Hz
    t100, g100 = resample_to_fs(t_s, g, fs_target)

    # detrend (por eixo)
    g100_dt = np.column_stack([detrend(g100[:, i]) for i in range(3)])

    # passa-baixa 1.5 Hz
    g100_f = np.column_stack([lowpass_filter(g100_dt[:, i], fs_target, fc) for i in range(3)])

    # norma
    norm = np.sqrt(np.sum(g100_f**2, axis=1))

    return {
        "t": t100,
        "gx": g100_f[:, 0],
        "gy": g100_f[:, 1],
        "gz": g100_f[:, 2],
        "norm": norm,
        "fs": fs_target
    }


# -----------------------------
# Cinêmica (120 Hz)
# -----------------------------
def preprocess_kinematic(df: pd.DataFrame, fs=120.0) -> dict:
    """
    df: 3 colunas X,Y,Z (sem tempo). Se houver mais, pega 3 primeiras numéricas.
    """
    df = _coerce_numeric_df(df)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 3:
        raise ValueError(f"Cinêmica: esperado >= 3 colunas numéricas (X,Y,Z). Achei {len(num_cols)}.")

    xyz = df[num_cols[:3]].to_numpy(dtype=float)
    n = xyz.shape[0]
    t = np.arange(n) / fs
    return {"t": t, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "fs": fs}


# -----------------------------
# Detecção simples de "salto" para sugerir trigger
# -----------------------------
def suggest_trigger_time(t: np.ndarray, sig: np.ndarray) -> float:
    """
    Sugere trigger como o maior pico na derivada absoluta (mudança brusca).
    """
    sig = np.asarray(sig, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(sig) < 5:
        return float(t[0]) if len(t) else 0.0

    d = np.abs(np.diff(sig))
    idx = int(np.argmax(d))
    # idx se refere à diff; pega o tempo após a transição
    return float(t[min(idx + 1, len(t) - 1)])


# -----------------------------
# UI: upload
# -----------------------------
colA, colB = st.columns(2)
with colA:
    up_kin = st.file_uploader("📄 Arquivo Cinêmica (X,Y,Z) — 120 Hz", type=["csv", "txt"])
with colB:
    up_gyr = st.file_uploader("📄 Arquivo Giroscópio (t,x,y,z)", type=["csv", "txt"])

if not up_kin or not up_gyr:
    st.info("Carregue os dois arquivos para começar.")
    st.stop()

# leitura
df_kin = _read_flexible_csv(up_kin)
df_gyr = _read_flexible_csv(up_gyr)

# processamento
try:
    kin = preprocess_kinematic(df_kin, fs=120.0)
except Exception as e:
    st.error(f"Erro na cinêmica: {e}")
    st.stop()

try:
    gyr = preprocess_gyro(df_gyr, fs_target=100.0, fc=1.5)
except Exception as e:
    st.error(f"Erro no giroscópio: {e}")
    st.stop()

# sugestões de trigger
t0_kin_sug = suggest_trigger_time(kin["t"], kin["z"])      # salto no Z da cinemática
t0_gyr_sug = suggest_trigger_time(gyr["t"], gyr["gy"])     # salto no Y do giroscópio

# -----------------------------
# Controles
# -----------------------------
st.subheader("1) Definir o trigger (tempo zero) em cada sinal")
c1, c2, c3 = st.columns([1.2, 1.2, 1.6])

with c1:
    kin_axis = st.selectbox("Cinêmica para plot (eixo)", ["y", "z"], index=1)

with c2:
    invert_kin = st.checkbox("Inverter cinêmica (multiplicar por -1)", value=False)
    invert_gyr_jump = st.checkbox("Inverter giroscópio Y (multiplicar por -1)", value=False)

with c3:
    st.caption("Dica: o slider começa na sugestão automática baseada em mudança brusca (derivada). Ajuste até alinhar os saltos.")

t_kin_min, t_kin_max = float(kin["t"][0]), float(kin["t"][-1])
t_gyr_min, t_gyr_max = float(gyr["t"][0]), float(gyr["t"][-1])

t0_kin = st.slider(
    "Trigger na Cinêmica (s) — salto no Z (referência)",
    min_value=t_kin_min, max_value=t_kin_max, value=float(np.clip(t0_kin_sug, t_kin_min, t_kin_max)),
    step=1.0 / 120.0
)

t0_gyr = st.slider(
    "Trigger no Giroscópio (s) — salto no Y (referência)",
    min_value=t_gyr_min, max_value=t_gyr_max, value=float(np.clip(t0_gyr_sug, t_gyr_min, t_gyr_max)),
    step=1.0 / 100.0
)

# aplica inversões apenas para visual/trigger
kin_sig = kin[kin_axis].copy()
if invert_kin:
    kin_sig = -kin_sig

gyr_jump_sig = gyr["gy"].copy()
if invert_gyr_jump:
    gyr_jump_sig = -gyr_jump_sig

# tempo relativo (sincronizado)
tkin_sync = kin["t"] - t0_kin
tgyr_sync = gyr["t"] - t0_gyr

# -----------------------------
# Intervalo de visualização (sobreposição)
# -----------------------------
st.subheader("2) Selecionar janela temporal de visualização (após sincronizar)")

# calcula janela comum (onde ambos têm dados)
tmin_common = max(float(tkin_sync.min()), float(tgyr_sync.min()))
tmax_common = min(float(tkin_sync.max()), float(tgyr_sync.max()))

if tmin_common >= tmax_common:
    st.error("Não há sobreposição temporal entre os sinais após os triggers escolhidos.")
    st.stop()

# slider de janela
win = st.slider(
    "Janela (s) no tempo sincronizado",
    min_value=float(tmin_common),
    max_value=float(tmax_common),
    value=(max(float(-2.0), float(tmin_common)), min(float(5.0), float(tmax_common))),
    step=0.01
)
t_start, t_end = win

mask_kin = (tkin_sync >= t_start) & (tkin_sync <= t_end)
mask_gyr = (tgyr_sync >= t_start) & (tgyr_sync <= t_end)

# -----------------------------
# Plot duplo (eixo esquerdo e direito)
# -----------------------------
st.subheader("3) Plot duplo (cinêmica vs norma do giroscópio)")
fig, ax1 = plt.subplots()

# cinêmica no eixo esquerdo
ax1.plot(tkin_sync[mask_kin], kin_sig[mask_kin])
ax1.set_xlabel("Tempo sincronizado (s)")
ax1.set_ylabel(f"Cinêmica {kin_axis.upper()} (unid. original)")

# norma do giroscópio no eixo direito
ax2 = ax1.twinx()
ax2.plot(tgyr_sync[mask_gyr], gyr["norm"][mask_gyr])
ax2.set_ylabel("Norma do giroscópio (após LPF 1.5 Hz)")

# marca t=0
ax1.axvline(0, linestyle="--", linewidth=1)

st.pyplot(fig, use_container_width=True)

# -----------------------------
# Diagnóstico opcional: visualizar sinais do salto para ajustar trigger
# -----------------------------
with st.expander("🔎 Diagnóstico: ver sinais usados para trigger (Z cinêmica e Y giroscópio)"):
    fig2, bx1 = plt.subplots()
    bx1.plot(tkin_sync[mask_kin], (kin["z"][mask_kin] if not invert_kin else -kin["z"][mask_kin]))
    bx1.set_xlabel("Tempo sincronizado (s)")
    bx1.set_ylabel("Cinêmica Z (referência do salto)")
    bx1.axvline(0, linestyle="--", linewidth=1)

    bx2 = bx1.twinx()
    bx2.plot(tgyr_sync[mask_gyr], gyr_jump_sig[mask_gyr])
    bx2.set_ylabel("Giroscópio Y (referência do salto)")

    st.pyplot(fig2, use_container_width=True)

st.success("Pronto: giroscópio pré-processado, triggers definidos, sinais sincronizados e plot duplo com janela selecionável.")
