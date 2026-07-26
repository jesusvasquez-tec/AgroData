"""
app.py
======
AgroData Perú — Dashboard de predicción de producción de Plátano (Ica).

Punto de entrada de la aplicación Streamlit. Se encarga únicamente de:
  1. la interfaz de carga de archivo,
  2. mostrar la barra de progreso mientras corre el pipeline,
  3. organizar el resultado en pestañas.

Toda la lógica de negocio vive en utils/ y models/, nunca aquí.
"""

import pandas as pd
import streamlit as st

import config
from utils.pipeline_runner import ejecutar_pipeline
from utils.data_pipeline import PipelineError
from components import tabs

st.set_page_config(
    page_title="AgroData Perú",
    page_icon="🌾",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Estilos mínimos (paleta idéntica al dashboard HTML del proyecto)
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
.block-container {{ padding-top: 1.5rem; }}
h1, h2, h3 {{ color: {config.COLOR_VERDE_OSCURO}; }}
[data-testid="stMetricValue"] {{ color: {config.COLOR_VERDE_OSCURO}; }}
</style>
""", unsafe_allow_html=True)


def _leer_archivo(uploaded_file) -> pd.DataFrame:
    nombre = uploaded_file.name.lower()
    if nombre.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    xls = pd.ExcelFile(uploaded_file)
    if len(xls.sheet_names) == 1:
        return pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    # Varias hojas (p. ej. "2016_2020" y "2021_2025"): se combinan todas las
    # que tengan el esquema esperado de SISAGRI en un único DataFrame.
    partes = []
    hojas_omitidas = []
    for sheet in xls.sheet_names:
        df_hoja = pd.read_excel(xls, sheet_name=sheet)
        faltantes = [c for c in config.COLUMNAS_ORIGINALES_REQUERIDAS if c not in df_hoja.columns]
        if faltantes:
            hojas_omitidas.append(sheet)
            continue
        partes.append(df_hoja)

    if not partes:
        raise ValueError(
            f"Ninguna de las hojas del Excel ({', '.join(xls.sheet_names)}) tiene el "
            "esquema esperado de SISAGRI."
        )
    if hojas_omitidas:
        st.warning(f"Se omitieron estas hojas por no tener el esquema de SISAGRI: {', '.join(hojas_omitidas)}")

    st.info(f"Se combinaron {len(partes)} hojas del Excel: {', '.join(xls.sheet_names)}")
    return pd.concat(partes, ignore_index=True)


def main():
    st.title("🌾 AgroData Perú")
    st.caption("Sistema de predicción de producción agrícola · Plátano · Ica (2021–2025) · CRISP-DM")

    if "resultado" not in st.session_state:
        st.session_state.resultado = None

    with st.sidebar:
        st.header("📂 Datos de entrada")
        uploaded_file = st.file_uploader(
            "Importar base de datos SISAGRI (Excel o CSV)",
            type=["xlsx", "xls", "csv"],
            help="El archivo debe contener las columnas originales de SISAGRI: "
                 "anho, mes, Dpto, Prov, Dist, dsc_Cultivo, PRODUCCION(t), "
                 "COSECHA (ha), MTO_PRECCHAC (S/ x kg).",
        )
        forzar = st.checkbox(
            "Forzar reentrenamiento",
            value=False,
            help="Por defecto, si vuelves a cargar el mismo archivo se reutiliza el "
                 "modelo ya entrenado (no se recalculan las predicciones). "
                 "Marca esto solo si quieres reentrenar desde cero.",
        )
        st.divider()
        st.caption(
            f"Alcance: {config.CULTIVO_OBJETIVO.title()} · {config.DEPARTAMENTO_OBJETIVO.title()}\n\n"
            f"Escenarios: A (11 var), B (8 var), C (6 var)\n\n"
            f"Modelos comparados: Random Forest, LightGBM, XGBoost"
        )

    if uploaded_file is None:
        st.info(
            "👈 Carga un archivo Excel/CSV con el formato de SISAGRI para ejecutar el "
            "pipeline completo (limpieza → features → entrenamiento → predicciones → dashboard)."
        )
        return

    ejecutar = st.session_state.resultado is None or forzar or st.session_state.get("_last_file") != uploaded_file.name

    if ejecutar:
        try:
            df_raw = _leer_archivo(uploaded_file)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            return

        progress_bar = st.progress(0, text="Iniciando...")

        def on_progress(pct, msg):
            progress_bar.progress(pct / 100, text=msg)

        try:
            resultado = ejecutar_pipeline(df_raw, on_progress=on_progress, forzar_reentrenamiento=forzar)
        except PipelineError as e:
            progress_bar.empty()
            st.error(f"⚠️ {e}")
            return
        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ Ocurrió un error inesperado procesando el archivo: {e}")
            raise

        progress_bar.empty()
        st.session_state.resultado = resultado
        st.session_state._last_file = uploaded_file.name

        if resultado["usar_cache"]:
            st.success("✅ Archivo procesado. Se reutilizó el modelo ya entrenado (mismo dataset detectado).")
        else:
            st.success("✅ Archivo procesado y modelo entrenado desde cero.")

    resultado = st.session_state.resultado
    if resultado is None:
        return

    tabs.render_header_meta(resultado)
    st.divider()

    tab_labels = [
        "🏆 Rankings", "🔮 Predicción próximo mes", "📅 Últimos 12 meses",
        "📈 Producción Anual", "⚖️ Comparativa Anual", "📜 Histórico Completo",
        "🧬 Variables del Modelo",
    ]
    t1, t2, t3, t4, t5, t6, t7 = st.tabs(tab_labels)

    with t1:
        tabs.render_tab_rankings(resultado)
    with t2:
        tabs.render_tab_prediccion(resultado)
    with t3:
        tabs.render_tab_historico(resultado, ultimos_n_meses=12, key_suffix="12m")
    with t4:
        tabs.render_tab_anual(resultado)
    with t5:
        tabs.render_tab_comparativa(resultado)
    with t6:
        tabs.render_tab_historico(resultado, ultimos_n_meses=None, key_suffix="full")
    with t7:
        tabs.render_tab_variables(resultado)


if __name__ == "__main__":
    main()
