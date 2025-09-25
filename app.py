import io
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --------------------------- CONFIG ---------------------------
st.set_page_config(page_title="BI – UNINCCA (Streamlit MVP)", layout="wide")
TODAY = pd.Timestamp.today().normalize()  # sin tz

EXPECTED_COLS = [
    "Ciudad",
    "Programa Admitido",
    "Ingreso por",
    "Periodo Ingreso",
    "Fecha de Pago Matricula",
    "Fecha de Nacimiento",
]

# Columnas “core” (para cortar filas basura al final)
CORE_COLS = [
    "Periodo Ingreso",
    "Nombres",
    "Apellidos",
    "Ciudad",
    "Programa Admitido",
]

# Diccionario de ciudades (puedes ampliarlo)
CITY_GEO = {
    "BOGOTÁ": (4.7110, -74.0721, "Bogotá D.C."),
    "MEDELLIN": (6.2442, -75.5812, "Antioquia"),
    "MEDELLÍN": (6.2442, -75.5812, "Antioquia"),
    "CALI": (3.4516, -76.5320, "Valle del Cauca"),
    "BARRANQUILLA": (10.9685, -74.7813, "Atlántico"),
    "CARTAGENA": (10.3910, -75.4794, "Bolívar"),
    "BUCARAMANGA": (7.1193, -73.1227, "Santander"),
    "CÚCUTA": (7.8891, -72.4967, "Norte de Santander"),
    "CUCUTA": (7.8891, -72.4967, "Norte de Santander"),
    "SOACHA": (4.5857, -74.2144, "Cundinamarca"),
    "VILLAVICENCIO": (4.1420, -73.6266, "Meta"),
    "TUNJA": (5.5353, -73.3678, "Boyacá"),
    "NEIVA": (2.9273, -75.2819, "Huila"),
    "VALLEDUPAR": (10.4631, -73.2532, "Cesar"),
    "PASTO": (1.2136, -77.2811, "Nariño"),
    "IBAGUÉ": (4.4389, -75.2322, "Tolima"),
    "IBAGUE": (4.4389, -75.2322, "Tolima"),
    "PEREIRA": (4.8087, -75.6906, "Risaralda"),
    "MANIZALES": (5.0703, -75.5138, "Caldas"),
    "SANTA MARTA": (11.2408, -74.1990, "Magdalena"),
    "MONTERÍA": (8.74798, -75.88143, "Córdoba"),
    "MONTERIA": (8.74798, -75.88143, "Córdoba"),
    "POPAYÁN": (2.4448, -76.6147, "Cauca"),
    "POPAYAN": (2.4448, -76.6147, "Cauca"),
    "ARMENIA": (4.5339, -75.6811, "Quindío"),
}

# --------------------------- UTILS ---------------------------
def normalize_city(x: str):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    up = s.upper()
    if up in ["BOGOTA", "BOGOTÁ D.C.", "BOGOTA D.C.", "BOGOTÁ DC"]:
        return "BOGOTÁ"
    if up == "BOGOTÁ" or s == "Bogotá":
        return "BOGOTÁ"
    return s

def to_datetime_safe(x):
    """Parsea fechas con dayfirst y devuelve NaT si no se puede."""
    if pd.isna(x) or str(x).strip() == "":
        return pd.NaT
    try:
        return pd.to_datetime(x, dayfirst=True, errors="coerce")
    except Exception:
        return pd.NaT

def ensure_expected_columns(df: pd.DataFrame):
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    return missing

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas completamente vacías o que solo tienen espacios."""
    df2 = df.dropna(how="all")
    mask_nonblank = ~(df2.astype(str).apply(lambda r: "".join(r), axis=1).str.strip() == "")
    return df2.loc[mask_nonblank].copy()

def trim_to_core_rows(df: pd.DataFrame, core_cols: list[str]) -> pd.DataFrame:
    """
    Mantiene solo filas hasta la ÚLTIMA donde alguna columna 'core' tiene dato real.
    Ignora columnas numéricas arrastradas (ej. 'No.') y filas con fórmulas vacías.
    """
    cols = [c for c in core_cols if c in df.columns]
    if not cols:
        return df
    core = (
        df[cols]
        .apply(lambda s: s.astype(str).str.strip())
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    mask = core.notna().any(axis=1)
    if not mask.any():
        return df
    last_idx = mask[mask].index[-1]
    df = df.loc[:last_idx]
    df = df.loc[mask]
    return df.reset_index(drop=True)

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    # Normaliza headers & strings
    work.columns = [str(c).strip() for c in work.columns]
    for col in work.select_dtypes(include="object").columns:
        work[col] = work[col].astype(str).str.strip()

    # Normaliza ciudad
    if "Ciudad" in work.columns:
        work["Ciudad"] = work["Ciudad"].apply(normalize_city)

    # Fechas
    if "Fecha de Pago Matricula" in work.columns:
        work["Fecha de Pago Matricula"] = work["Fecha de Pago Matricula"].apply(to_datetime_safe)
    if "Fecha de Nacimiento" in work.columns:
        work["Fecha de Nacimiento"] = work["Fecha de Nacimiento"].apply(to_datetime_safe)

    # Derivados
    work["_PagoMatricula"] = work["Fecha de Pago Matricula"].notna() if "Fecha de Pago Matricula" in work.columns else False
    if "Fecha de Nacimiento" in work.columns:
        age_years = (TODAY - work["Fecha de Nacimiento"]).dt.days / 365.25
        # Filtra edades imposibles para no sesgar KPIs
        age_years = age_years.where((age_years >= 15) & (age_years <= 90))
        work["_Edad"] = age_years
    else:
        work["_Edad"] = np.nan

    # Textos clave
    for c in ["Programa Admitido", "Ingreso por", "Periodo Ingreso"]:
        if c in work.columns:
            work[c] = work[c].replace("", np.nan).fillna("SIN DATO")

    # Estado normalizado (si existe)
    if "Estado" in work.columns:
        work["Estado"] = work["Estado"].astype(str).str.strip().str.upper()

    return work

def top_counts(df: pd.DataFrame, col: str, n=10) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=int)
    s = df[col].fillna("SIN DATO").replace("", "SIN DATO")
    return s.value_counts().head(n)

def paid_rate_by_program(df: pd.DataFrame, program_col="Programa Admitido") -> pd.Series:
    if program_col not in df.columns or "_PagoMatricula" not in df.columns:
        return pd.Series(dtype=float)
    g = df.groupby(program_col)["_PagoMatricula"].mean().sort_values(ascending=False) * 100
    return g.round(1).head(10)

def pct_nulls(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in cols:
        if c in df.columns:
            nulls = df[c].isna().sum() + (df[c] == "").sum()
            rows.append([c, round(100 * nulls / n, 2) if n else 0.0])
    return pd.DataFrame(rows, columns=["Columna", "% Nulos"])

def build_funnel(df: pd.DataFrame) -> dict:
    total = len(df)
    admitidos = df["Programa Admitido"].notna().sum() if "Programa Admitido" in df.columns else None
    inscritos = None
    for col in ["Fecha de Pago Inscripción", "Fecha de Pago Inscripcion",
                "Recibo Inscripción - Send Status", "Recibo Inscripcion - Send Status"]:
        if col in df.columns:
            if "Fecha de Pago" in col:
                inscritos = df[col].apply(to_datetime_safe).notna().sum()
            else:
                inscritos = df[col].astype(str).str.len().gt(0).sum()
            break
    pagos = df["_PagoMatricula"].sum() if "_PagoMatricula" in df.columns else None
    return {"Leads": total, "Inscritos": inscritos, "Admitidos": admitidos, "Matriculados": pagos}

def series_by_date(df: pd.DataFrame, date_col: str, flag_col: str = None, freq="W"):
    """Serie semanal regular por resample; fillna 0."""
    if date_col not in df.columns:
        return pd.Series(dtype=int)
    d = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    if flag_col is None:
        s = pd.Series(1, index=d.dropna())
        ts = s.resample(freq).sum().astype(int)
    else:
        tmp = df.copy()
        tmp[date_col] = d
        tmp = tmp.dropna(subset=[date_col])
        fl = tmp[flag_col].astype(int) if tmp[flag_col].dtype != bool else tmp[flag_col].astype(int)
        ts = fl.set_index(tmp[date_col]).resample(freq).sum().astype(int)
    return ts.fillna(0)

def winsorize_series(y: pd.Series, q=95):
    if y.empty:
        return y
    lim = np.percentile(y.values, q)
    return y.clip(upper=lim)

# --------------------------- SIDEBAR ---------------------------
try:
    st.sidebar.image("logo.png", use_container_width=True)
except Exception:
    st.sidebar.markdown("**Universidad INCCA de Colombia**")
st.sidebar.markdown("### BI – MVP BUSINESS ANALYTICS")
st.sidebar.markdown("**Fabián Valero – Esteban Fonseca**  \nDocente: **Ivon Forero**")

uploaded = st.sidebar.file_uploader("Sube Excel (.xlsx) o CSV", type=["xlsx", "csv"])

if not uploaded:
    st.info("Sube un archivo para comenzar.")
    st.stop()

# --------------------------- LECTURA DE ARCHIVO ---------------------------
def read_any_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        sample = file_bytes[:2048].decode("utf-8", errors="ignore")
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
        df = pd.read_csv(io.BytesIO(file_bytes), delimiter=delimiter, dtype=str, keep_default_na=False)
    elif name.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", dtype=str)
    else:
        raise ValueError("Extensión no soportada")
    # Normaliza y limpieza base
    df.columns = [str(c).strip() for c in df.columns]
    df = df.applymap(lambda v: v.strip() if isinstance(v, str) else v)
    df = drop_empty_rows(df)
    # Recorta por columnas “core” (evita contar filas con fórmulas/arrastres)
    df = trim_to_core_rows(df, CORE_COLS)
    return df

try:
    raw = read_any_file(uploaded.getvalue(), uploaded.name)
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

n_raw = len(raw)
missing = ensure_expected_columns(raw)
if missing:
    st.error(f"Faltan columnas requeridas: {', '.join(missing)}")
    st.dataframe(pd.DataFrame({'Columna esperada': EXPECTED_COLS}))
    st.stop()

# --------------------------- LIMPIEZA & FEATURES ---------------------------
df = derive_features(raw)
n_after_clean = len(df)

# ---- Derivar año (para filtros y mapas por año) ----
if "Fecha de Pago Matricula" in df.columns:
    df["_Year"] = df["Fecha de Pago Matricula"].dt.year

# Si hay filas sin fecha de pago, intenta sacar el año desde "Periodo Ingreso" (ej. "2025-3")
if "_Year" not in df or df["_Year"].isna().any():
    if "Periodo Ingreso" in df.columns:
        años_pi = pd.to_numeric(df["Periodo Ingreso"].astype(str).str.extract(r"^(\d{4})")[0], errors="coerce")
        df["_Year"] = df["_Year"].fillna(años_pi)


# --------------------------- FILTROS ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")
sel_periodo  = st.sidebar.multiselect("Periodo Ingreso",
    sorted(df["Periodo Ingreso"].dropna().unique()) if "Periodo Ingreso" in df.columns else [])
sel_ciudad   = st.sidebar.multiselect("Ciudad",
    sorted(df["Ciudad"].dropna().unique()) if "Ciudad" in df.columns else [])
sel_programa = st.sidebar.multiselect("Programa Admitido",
    sorted(df["Programa Admitido"].dropna().unique()) if "Programa Admitido" in df.columns else [])

mask = pd.Series(True, index=df.index)
if sel_periodo:  mask &= df["Periodo Ingreso"].isin(sel_periodo)
if sel_ciudad:   mask &= df["Ciudad"].isin(sel_ciudad)
if sel_programa: mask &= df["Programa Admitido"].isin(sel_programa)

df = df[mask].copy()
n_after_filters = len(df)

# ---- Controles extra ----
st.sidebar.markdown("---")
st.sidebar.subheader("Pronóstico")
model_opt = st.sidebar.selectbox(
    "Modelo",
    ["Holt-Winters (simple)", "Holt-Winters estacional (3 ingresos/año)", "SARIMAX (17 semanas)"],
    index=1
)
season_weeks = st.sidebar.slider("Periodo estacional (semanas)", min_value=8, max_value=26, value=17, step=1)
winsor_q = st.sidebar.slider("Winsorizar picos (percentil)", 90, 100, 95, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Mapa")
bubble_scale = st.sidebar.slider("Escala burbujas (x)", min_value=10, max_value=150, value=50, step=5)

years_avail = sorted([int(y) for y in df["_Year"].dropna().unique()]) if "_Year" in df.columns else []
default_years = [y for y in [2025, 2026] if y in years_avail] or years_avail[-2:]
sel_years = st.sidebar.multiselect("Años (mapa)", years_avail, default=default_years)


# --------------------------- KPIs ---------------------------
st.title("Tablero ejecutivo")
st.caption(f"Registros (archivo): **{n_raw:,}**  |  Después de limpieza: **{n_after_clean:,}**  |  Después de filtros: **{n_after_filters:,}**")

total = len(df)
paid = int(df["_PagoMatricula"].sum())
tasa_pago = round(100 * paid / total, 2) if total else 0.0
edad_media = round(df["_Edad"].dropna().mean(), 1) if "_Edad" in df.columns else None
edad_mediana = round(df["_Edad"].dropna().median(), 1) if "_Edad" in df.columns else None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total registros", f"{total:,}")
c2.metric("Matrículas con fecha de pago", f"{paid:,}")
c3.metric("Tasa global de pago", f"{tasa_pago:.2f}%")
c4.metric("Edad media (años)", "—" if pd.isna(edad_media) else edad_media)
c5.metric("Edad mediana (años)", "—" if pd.isna(edad_mediana) else edad_mediana)

# Estados (si existe)
if "Estado" in df.columns and len(df):
    desistidos = (df["Estado"] == "DESISTIDO").sum()
    aplazados  = (df["Estado"] == "APLAZADO").sum()
    c6, c7 = st.columns(2)
    if total > 0:
        c6.metric("% desistimiento", f"{(desistidos/total*100):.2f}%")
        c7.metric("% aplazamiento", f"{(aplazados/total*100):.2f}%")
    st.subheader("Distribución por estado")
    st.plotly_chart(
        px.bar(
            df["Estado"].fillna("SIN DATO").value_counts().sort_values(ascending=True),
            orientation="h", labels={"value": "Cantidad", "index": ""},
            title="Estados", template="plotly_white"
        ),
        use_container_width=True
    )

# --------------------------- TOP N ---------------------------
st.subheader("Top 10 por dimensión")
top_prog = top_counts(df, "Programa Admitido", 10)
top_ciu  = top_counts(df, "Ciudad", 10)
top_can  = top_counts(df, "Ingreso por", 10)
top_per  = top_counts(df, "Periodo Ingreso", 10)
tasa_prog = paid_rate_by_program(df, "Programa Admitido")

cc1, cc2 = st.columns(2)
with cc1:
    fig = px.bar(top_prog.sort_values(ascending=True), orientation="h",
                 title="Top Programas (volumen)",
                 labels={"value": "Cantidad", "index": ""},
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
with cc2:
    fig = px.bar(top_ciu.sort_values(ascending=True), orientation="h",
                 title="Top Ciudades (volumen)",
                 labels={"value": "Cantidad", "index": ""},
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

cc3, cc4 = st.columns(2)
with cc3:
    fig = px.bar(top_can.sort_values(ascending=True), orientation="h",
                 title="Top Canales – Ingreso por (volumen)",
                 labels={"value": "Cantidad", "index": ""},
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
with cc4:
    fig = px.bar(top_per.sort_values(ascending=True), orientation="h",
                 title="Periodos de Ingreso (Top)",
                 labels={"value": "Cantidad", "index": ""},
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Tasa de pago por programa (%) – Top 10")
figtp = px.bar(tasa_prog.sort_values(ascending=True), orientation="h",
               title="Tasa de pago de matrícula (%) por programa (Top 10)",
               labels={"value": "% Pago", "index": ""},
               template="plotly_white")
st.plotly_chart(figtp, use_container_width=True)

# --------------------------- EMBUDO ---------------------------
st.header("Embudo de conversión")
funnel = build_funnel(df)
funnel_items = [(k, v) for k, v in funnel.items() if v is not None]
if len(funnel_items) >= 2:
    labels = [k for k, _ in funnel_items]
    values = [v for _, v in funnel_items]
    figf = px.funnel(x=values, y=labels,
                     title="Embudo (Leads → Inscritos → Admitidos → Matriculados)",
                     template="plotly_white")
    st.plotly_chart(figf, use_container_width=True)
else:
    st.info("No se encontraron suficientes columnas para construir el embudo completo. Se muestran los pasos disponibles.")
    st.json(funnel)

# --------------------------- SERIE HISTÓRICA ---------------------------
st.subheader("Pagos por semana (histórico)")
hist = df["Fecha de Pago Matricula"].dropna()
if len(hist):
    hist = pd.Series(1, index=hist).resample("W").sum().fillna(0)
    st.plotly_chart(px.line(hist, title="Pagos semanales (histórico)", template="plotly_white"),
                    use_container_width=True)
else:
    st.info("Sin datos de 'Fecha de Pago Matricula' para construir el histórico.")

# --------------------------- MAPA (ciudad / departamento) ---------------------------
st.header("Mapa por ciudad (scatter)")

city_group = df.groupby("Ciudad", dropna=True).agg(cantidad=("Ciudad", "count"))
if "_PagoMatricula" in df.columns:
    city_group["pagos"] = df.groupby("Ciudad")["_PagoMatricula"].sum()
    city_group["tasa_pago"] = np.where(
        city_group["cantidad"] > 0,
        (city_group["pagos"] / city_group["cantidad"]) * 100,
        np.nan
    )
else:
    city_group["pagos"] = 0
    city_group["tasa_pago"] = np.nan

geo_rows, no_match = [], []
for city, row in city_group.iterrows():
    if pd.isna(city):
        continue
    key = str(city).strip().upper()
    if key not in CITY_GEO:
        no_match.append(str(city))
        continue
    lat, lon, depto = CITY_GEO[key]
    cantidad = int(row["cantidad"])
    tasa = None if pd.isna(row.get("tasa_pago", np.nan)) else float(row["tasa_pago"])
    radius = max(1500, int(bubble_scale * math.sqrt(max(cantidad, 1))))
    t = 0.0 if tasa is None else max(0.0, min(tasa/100.0, 1.0))
    color = [int(255*(1 - t)), int(255*t), 80, 200]
    geo_rows.append({
        "Ciudad": city, "Departamento": depto, "lat": lat, "lon": lon,
        "Cantidad": cantidad, "TasaPago": tasa, "Radius": radius, "Color": color
    })

geo_df = pd.DataFrame(geo_rows)
total_ciudades = city_group.shape[0]
mapeadas = geo_df.shape[0]
if total_ciudades > 0:
    st.caption(f"Ciudades mapeadas: {mapeadas}/{total_ciudades}")
if no_match:
    st.caption("No mapeadas (añádelas a CITY_GEO): " + ", ".join(sorted(set(no_match))[:15]) + ("..." if len(set(no_match))>15 else ""))

if not geo_df.empty:
    lat0 = float(geo_df["lat"].mean())
    lon0 = float(geo_df["lon"].mean())
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v11",
        initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0, zoom=5, pitch=0),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=geo_df,
                get_position='[lon, lat]',
                get_radius="Radius",
                get_fill_color="Color",
                pickable=True,
            )
        ],
        tooltip={"text": "{Ciudad}\nDepartamento: {Departamento}\nCantidad: {Cantidad}\nTasa pago: {TasaPago}%"},
    ))
else:
    st.info("No se pudo geolocalizar ninguna ciudad. Agrega más ciudades al diccionario CITY_GEO.")

# --------------------------- PRONÓSTICOS ---------------------------
st.header("Pronósticos (Holt-Winters / SARIMAX)")

def make_weekly_counts(df, date_col):
    s = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce").dropna()
    return pd.Series(1, index=s).resample("W").sum().astype(float).fillna(0)

y_week = make_weekly_counts(df, "Fecha de Pago Matricula")

def plot_forecast(y, yhat, title):
    dfp = pd.DataFrame({"Fecha": y.index.append(yhat.index),
                        "Valor": pd.concat([y, yhat])})
    fig = px.line(dfp, x="Fecha", y="Valor", title=title, template="plotly_white")
    fig.add_vline(x=y.index.max(), line_dash="dash", line_color="orange")
    st.plotly_chart(fig, use_container_width=True)

def forecast_hw_simple(y, periods=8):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y2 = winsorize_series(y, winsor_q)
    model = ExponentialSmoothing(y2, trend="add", seasonal=None, damped_trend=True)
    fit = model.fit(optimized=True)
    return fit.forecast(periods)

def forecast_hw_seasonal(y, periods=8, sp=17):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y2 = winsorize_series(y, winsor_q)
    # estacionalidad aditiva, tendencia amortiguada
    model = ExponentialSmoothing(y2, trend="add", seasonal="add", seasonal_periods=sp, damped_trend=True)
    fit = model.fit(optimized=True)
    return fit.forecast(periods)

def forecast_sarimax(y, periods=8, sp=17):
    # SARIMAX con estacionalidad de ~17 semanas
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y2 = winsorize_series(y, winsor_q)
    # parámetros conservadores y estacionales
    model = SARIMAX(y2, order=(1,0,1), seasonal_order=(1,0,1, sp), enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    return fit.get_forecast(steps=periods).predicted_mean

# Solo si hay señal suficiente
if (y_week > 0).sum() >= 4:
    # pronóstico a 8 semanas
    if model_opt == "Holt-Winters (simple)":
        fc = forecast_hw_simple(y_week, periods=8)
        plot_forecast(y_week, fc, "Pronóstico (HW simple, semanal)")
    elif model_opt == "Holt-Winters estacional (3 ingresos/año)":
        fc = forecast_hw_seasonal(y_week, periods=8, sp=season_weeks)
        plot_forecast(y_week, fc, f"Pronóstico (HW estacional, {season_weeks} semanas)")
    else:  # SARIMAX
        fc = forecast_sarimax(y_week, periods=8, sp=season_weeks)
        plot_forecast(y_week, fc, f"Pronóstico (SARIMAX, {season_weeks} semanas)")
else:
    st.info("Poca señal histórica (menos de 4 semanas con pagos > 0). Captura más semanas para pronosticar mejor.")



# --------------------------- SCORING (probabilidad de pago) ---------------------------
st.header("Scoring: probabilidad de matrícula")
sc_df = df.copy()
y = sc_df["_PagoMatricula"].astype(int)
features = ["Programa Admitido", "Ciudad", "Ingreso por", "Periodo Ingreso", "_Edad"]
X = sc_df[features].copy()
X["_Edad"] = X["_Edad"].fillna(X["_Edad"].median())

if y.nunique() < 2 or len(sc_df) < 50:
    st.info("No hay suficientes datos/clases para entrenar un modelo de scoring. Se mostrará una puntuación trivial (cero).")
    sc_df["ScorePago"] = 0.0
else:
    cat_cols = ["Programa Admitido", "Ciudad", "Ingreso por", "Periodo Ingreso"]
    num_cols = ["_Edad"]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    pipe = Pipeline(steps=[("prep", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    try:
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        st.caption(f"AUC (validación holdout): {auc:.3f}")
    except Exception:
        pass
    sc_df["ScorePago"] = pipe.predict_proba(X)[:, 1]

st.dataframe(
    sc_df[["Ciudad", "Programa Admitido", "Ingreso por", "Periodo Ingreso", "_Edad", "Fecha de Pago Matricula", "ScorePago"]]
    .sort_values("ScorePago", ascending=False)
    .head(50),
    use_container_width=True
)

# --------------------------- PRESCRIPTIVO (sugerencias) ---------------------------
st.header("Sugerencias (prescriptivo)")
if len(df):
    eff = df.groupby("Ingreso por")["_PagoMatricula"].mean().mul(100).rename("tasa_%")
    vol = df["Ingreso por"].value_counts().rename("volumen")
    sug = pd.concat([eff, vol], axis=1).dropna()
    if not sug.empty:
        med_tasa = sug["tasa_%"].median()
        med_vol  = sug["volumen"].median()
        sug["acción"] = np.where(
            (sug["tasa_%"] >= med_tasa) & (sug["volumen"] >= med_vol), "Potenciar canal",
            np.where(sug["tasa_%"] >= med_tasa, "Escalar con cautela", "Revisar / optimizar")
        )
        st.dataframe(sug.sort_values(["tasa_%", "volumen"], ascending=[False, False]).head(15), use_container_width=True)
    else:
        st.info("No hay datos suficientes por canal para generar sugerencias.")
else:
    st.info("Sin datos tras filtros.")

# --------------------------- CALIDAD DE DATOS Y DESCARGAS ---------------------------
st.header("Calidad de datos")
st.dataframe(pct_nulls(df, EXPECTED_COLS), use_container_width=True)

st.header("Descargas")
kpis = {
    "total_registros": int(total),
    "matriculas_con_fecha_pago": int(paid),
    "tasa_pago_global_pct": float(tasa_pago),
    "edad_media": None if pd.isna(edad_media) else float(edad_media),
    "edad_mediana": None if pd.isna(edad_mediana) else float(edad_mediana),
    "top_programas": top_prog.to_dict(),
    "top_ciudades": top_ciu.to_dict(),
    "top_canales": top_can.to_dict(),
    "top_periodos": top_per.to_dict(),
    "tasa_pago_por_programa_pct": tasa_prog.to_dict(),
}
st.download_button(
    "Descargar KPIs (JSON)",
    data=json.dumps(kpis, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name="kpis.json",
    mime="application/json"
)

clean_export = df.copy()
if "Fecha de Pago Matricula" in clean_export.columns:
    clean_export["Fecha de Pago Matricula"] = clean_export["Fecha de Pago Matricula"].dt.strftime("%Y-%m-%d")
if "Fecha de Nacimiento" in clean_export.columns:
    clean_export["Fecha de Nacimiento"] = clean_export["Fecha de Nacimiento"].dt.strftime("%Y-%m-%d")
st.download_button(
    "Descargar dataset limpiado (CSV)",
    data=clean_export.to_csv(index=False).encode("utf-8-sig"),
    file_name="dataset_limpiado.csv",
    mime="text/csv"
)

# Exportar PDF (opcional: requiere `pip install weasyprint`)
st.caption("Sugerencias: ampliar CITY_GEO, agregar costos por canal para CAC, y crear diccionario de normalización institucional.")
try:
    from weasyprint import HTML  # pesado; solo si lo instalas
    if st.button("Exportar PDF (KPIs)"):
        html = f"""
        <h1>KPIs UNINCCA</h1>
        <p>Total: {total:,}</p>
        <p>Matrículas con pago: {paid:,}</p>
        <p>Tasa pago: {tasa_pago:.2f}%</p>
        <p>Edad media: {edad_media}</p>
        <p>Edad mediana: {edad_mediana}</p>
        """
        HTML(string=html).write_pdf("reporte.pdf")
        with open("reporte.pdf", "rb") as f:
            st.download_button("Descargar reporte.pdf", f, file_name="reporte.pdf", mime="application/pdf")
except Exception:
    st.caption("Para exportar a PDF instala weasyprint:  pip install weasyprint")

