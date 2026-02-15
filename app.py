import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px

st.set_page_config(page_title="Informe Estadístico - Ejercicio 10", layout="wide")

# =========================
# 1) DATA DEL EJERCICIO 10
# =========================
data = [
    {"ID": 1,  "Edad": 18, "Nota_Mat": 3.5, "Nota_Est": 3.8, "Horas_Estudio": 10, "Nivel_Estres": 6},
    {"ID": 2,  "Edad": 19, "Nota_Mat": 4.0, "Nota_Est": 4.2, "Horas_Estudio": 12, "Nivel_Estres": 5},
    {"ID": 3,  "Edad": 20, "Nota_Mat": 2.8, "Nota_Est": 3.1, "Horas_Estudio": 6,  "Nivel_Estres": 8},
    {"ID": 4,  "Edad": 21, "Nota_Mat": 3.2, "Nota_Est": 3.6, "Horas_Estudio": 8,  "Nivel_Estres": 7},
    {"ID": 5,  "Edad": 22, "Nota_Mat": 4.5, "Nota_Est": 4.8, "Horas_Estudio": 15, "Nivel_Estres": 4},
    {"ID": 6,  "Edad": 20, "Nota_Mat": 3.7, "Nota_Est": 3.5, "Horas_Estudio": 9,  "Nivel_Estres": 6},
    {"ID": 7,  "Edad": 19, "Nota_Mat": 2.9, "Nota_Est": 3.2, "Horas_Estudio": 7,  "Nivel_Estres": 7},
    {"ID": 8,  "Edad": 21, "Nota_Mat": 3.8, "Nota_Est": 3.9, "Horas_Estudio": 11, "Nivel_Estres": 5},
    {"ID": 9,  "Edad": 22, "Nota_Mat": 4.1, "Nota_Est": 4.3, "Horas_Estudio": 13, "Nivel_Estres": 4},
    {"ID": 10, "Edad": 18, "Nota_Mat": 2.5, "Nota_Est": 2.8, "Horas_Estudio": 5,  "Nivel_Estres": 9},
    {"ID": 11, "Edad": 23, "Nota_Mat": 4.6, "Nota_Est": 4.7, "Horas_Estudio": 16, "Nivel_Estres": 3},
    {"ID": 12, "Edad": 20, "Nota_Mat": 3.3, "Nota_Est": 3.4, "Horas_Estudio": 8,  "Nivel_Estres": 7},
]
df = pd.DataFrame(data)

numeric_cols = ["Edad", "Nota_Mat", "Nota_Est", "Horas_Estudio", "Nivel_Estres"]

# =========================
# 2) FUNCIONES DE CÁLCULO
# =========================
def moda_usable(series: pd.Series):
    s = series.dropna()
    vc = s.value_counts()

    if vc.empty:
        return ""

    max_f = vc.max()

    # Si el máximo es 1, nadie se repite => NO hay moda
    if max_f == 1:
        return "No hay moda"

    # Si sí hay repetidos, puede haber 1 o varias modas
    modas = vc[vc == max_f].index.tolist()
    if len(modas) == 1:
        return str(modas[0])

    return ", ".join(str(v) for v in modas)



def resumen_tendencia_central(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna()
        rows.append({
            "Variable": c,
            "Media": round(s.mean(), 4),
            "Mediana": round(s.median(), 4),
            "Moda": moda_usable(s)
        })
    return pd.DataFrame(rows)

def resumen_dispersion(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna()
        media = s.mean()
        var_muestral = s.var(ddof=1)     # muestral
        std_muestral = s.std(ddof=1)     # muestral
        rango = s.max() - s.min()
        cv = (std_muestral / media) * 100 if media != 0 else np.nan
        rows.append({
            "Variable": c,
            "Min": round(s.min(), 4),
            "Max": round(s.max(), 4),
            "Rango": round(rango, 4),
            "Varianza (muestral)": round(var_muestral, 4),
            "Desv.Est (muestral)": round(std_muestral, 4),
            "Coef. Variación (%)": round(cv, 2)
        })
    return pd.DataFrame(rows)

def cuartiles_y_percentiles(series: pd.Series) -> dict:
    s = series.dropna()
    # método típico (pandas = interpolación lineal)
    q1 = s.quantile(0.25)
    q2 = s.quantile(0.50)
    q3 = s.quantile(0.75)
    p25 = s.quantile(0.25)
    p50 = s.quantile(0.50)
    p75 = s.quantile(0.75)
    return {
        "Q1": round(q1, 4),
        "Q2": round(q2, 4),
        "Q3": round(q3, 4),
        "P25": round(p25, 4),
        "P50": round(p50, 4),
        "P75": round(p75, 4),
    }

def pearson_r(x: pd.Series, y: pd.Series) -> float:
    return float(x.corr(y, method="pearson"))

def interpretar_r(r: float) -> str:
    ar = abs(r)
    if ar >= 0.80:
        fuerza = "fuerte"
    elif ar >= 0.50:
        fuerza = "moderada"
    elif ar >= 0.30:
        fuerza = "débil"
    else:
        fuerza = "muy débil o casi nula"

    sentido = "positiva" if r > 0 else "negativa" if r < 0 else "nula"
    return f"Correlación {sentido} {fuerza} (r={r:.3f})."

# =========================
# 3) UI
# =========================
st.title("Informe Estadístico - Ejercicio 10")

st.subheader("Base de datos (12 estudiantes)")
st.dataframe(df, use_container_width=True)

# =========================
# 4) 1. Tendencia central
# =========================
st.header("1) Medidas de tendencia central")
df_central = resumen_tendencia_central(df, numeric_cols)
df_central["Moda"] = df_central["Moda"].astype(str)
st.dataframe(df_central, use_container_width=True)

# =========================
# 5) 2. Dispersión
# =========================
st.header("2) Medidas de dispersión")
df_disp = resumen_dispersion(df, numeric_cols)
st.dataframe(df_disp, use_container_width=True)

# =========================
# 6) 3. Posición (Mat y Est)
# =========================
st.header("3) Medidas de posición (cuartiles y percentiles 25, 50, 75)")
pos_mat = cuartiles_y_percentiles(df["Nota_Mat"])
pos_est = cuartiles_y_percentiles(df["Nota_Est"])

df_pos = pd.DataFrame([
    {"Variable": "Nota_Mat", **pos_mat},
    {"Variable": "Nota_Est", **pos_est},
])
st.dataframe(df_pos, use_container_width=True)

# =========================
# 7) 4. Correlación (Pearson)
# =========================
st.header("4) Correlación (Pearson)")

r_horas_est = pearson_r(df["Horas_Estudio"], df["Nota_Est"])
r_estres_mat = pearson_r(df["Nivel_Estres"], df["Nota_Mat"])

colA, colB = st.columns(2)
with colA:
    st.subheader("a) Horas de estudio vs Nota en Estadística")
    st.write(interpretar_r(r_horas_est))
    fig1 = px.scatter(
        df, x="Horas_Estudio", y="Nota_Est",
        title="Horas de Estudio vs Nota en Estadística",
        trendline="ols"
    )
    st.plotly_chart(fig1, use_container_width=True)

with colB:
    st.subheader("b) Nivel de estrés vs Nota en Matemáticas")
    st.write(interpretar_r(r_estres_mat))
    fig2 = px.scatter(
        df, x="Nivel_Estres", y="Nota_Mat",
        title="Nivel de Estrés vs Nota en Matemáticas",
        trendline="ols"
    )
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# 8) GRÁFICAS (barras + caja)
# =========================
st.header("Gráficas solicitadas")

st.subheader("Gráficas de barras (frecuencia por valor)")
graf_cols = st.columns(3)
for i, c in enumerate(numeric_cols):
    with graf_cols[i % 3]:
        freq = df[c].value_counts().sort_index().reset_index()
        freq.columns = [c, "Frecuencia"]
        fig_bar = px.bar(freq, x=c, y="Frecuencia", title=f"Frecuencia - {c}")
        st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Diagramas de caja (boxplot)")
box_cols = st.columns(3)
for i, c in enumerate(numeric_cols):
    with box_cols[i % 3]:
        fig_box = px.box(df, y=c, title=f"Boxplot - {c}")
        st.plotly_chart(fig_box, use_container_width=True)

# =========================
# 9) 5. Interpretación / Conclusiones
# =========================
st.header("5) Interpretación (conclusiones)")
st.write(
    """
- Con la tabla de tendencia central puedes describir el “nivel típico” del grupo en cada variable.
- Con la dispersión puedes decir si el grupo es homogéneo o si hay mucha variación.
- Los cuartiles y percentiles de las notas (Mat y Est) te ayudan a ubicar a un estudiante dentro del grupo.
- Las correlaciones te dicen si se ve relación entre estudiar y la nota, y entre estrés y desempeño.
"""
)

# =========================
# 10) DESCARGA (Excel)
# =========================
st.header("Descargar resultados (Excel)")

def to_excel_bytes():
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Datos", index=False)
        df_central.to_excel(writer, sheet_name="Tendencia_Central", index=False)
        df_disp.to_excel(writer, sheet_name="Dispersion", index=False)
        df_pos.to_excel(writer, sheet_name="Posicion", index=False)

        corr_df = pd.DataFrame([
            {"Relacion": "Horas_Estudio vs Nota_Est", "r": round(r_horas_est, 6), "Interpretacion": interpretar_r(r_horas_est)},
            {"Relacion": "Nivel_Estres vs Nota_Mat", "r": round(r_estres_mat, 6), "Interpretacion": interpretar_r(r_estres_mat)},
        ])
        corr_df.to_excel(writer, sheet_name="Correlacion", index=False)

    output.seek(0)
    return output.getvalue()

st.download_button(
    "Descargar Excel con tablas",
    data=to_excel_bytes(),
    file_name="Informe_Ejercicio_10.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)