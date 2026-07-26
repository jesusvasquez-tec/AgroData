"""
config.py
=========
Configuración centralizada de AgroData Perú.
Todas las constantes replican EXACTAMENTE los valores usados en el
notebook V2_PLAN_DE_TESIS_Cordova,_Vasquez.ipynb (Fases 2 a 6 del CRISP-DM).

Si necesitas ajustar un parámetro del pipeline (pesos del scoring, años de
corte train/valid/test, lags, etc.) hazlo AQUÍ y no dentro del código de
procesamiento, para mantener un único punto de verdad.
"""

# ---------------------------------------------------------------------------
# Filtros de alcance del proyecto (Fase 2.3 del notebook)
# ---------------------------------------------------------------------------
CULTIVO_OBJETIVO = "PLATANO"
DEPARTAMENTO_OBJETIVO = "ICA"

# ---------------------------------------------------------------------------
# Columnas originales esperadas en el Excel/CSV de SISAGRI
# ---------------------------------------------------------------------------
COLUMNAS_ORIGINALES_REQUERIDAS = [
    "anho", "mes", "Dpto", "Prov", "Dist", "dsc_Cultivo",
    "PRODUCCION(t)", "COSECHA (ha)", "MTO_PRECCHAC (S/ x kg)",
]

# Selección de variables representativas (Fase 3.1.2)
VARIABLES_SELECCIONADAS = [
    "anho", "mes", "Dpto", "Prov", "Dist", "dsc_Cultivo",
    "PRODUCCION(t)", "COSECHA (ha)", "MTO_PRECCHAC (S/ x kg)",
]

# Renombrado de variables (Fase 3.2.1)
RENOMBRE_COLUMNAS = {
    "anho": "anio",
    "Dpto": "departamento",
    "Prov": "provincia",
    "Dist": "distrito",
    "dsc_Cultivo": "cultivo",
    "COSECHA (ha)": "cosecha",
    "PRODUCCION(t)": "produccion",
    "MTO_PRECCHAC (S/ x kg)": "precio_chacra",
}

CLAVES_SERIE = ["departamento", "provincia", "distrito", "cultivo"]

# ---------------------------------------------------------------------------
# Parámetros de filtrado y validación de series (Fase 3.4)
# ---------------------------------------------------------------------------
START_YEAR = 2020
MIN_MESES_POR_ANIO = 12
MIN_MESES_PRODUCCION = 12
MIN_TOTAL_MESES_SERIE = 24
USAR_FILTRO_PRODUCCION = True
VENTANA_MESES_RECIENTES = 6
MIN_MESES_RECIENTES = 4

# ---------------------------------------------------------------------------
# Construcción del panel temporal (Fase 3.5)
# ---------------------------------------------------------------------------
USAR_RANGO_REAL_SERIE = True
USAR_IMPUTACION_PRECIO = True
USAR_BACKFILL_PRECIO = False

# ---------------------------------------------------------------------------
# Feature engineering (Fase 3.6)
# ---------------------------------------------------------------------------
LAGS = [1, 2, 3, 6, 12]
ROLL_WINDOWS = [3, 6, 12]
FORECAST_HORIZON = 1

# ---------------------------------------------------------------------------
# Escenarios experimentales (Fase 4.4)
# ---------------------------------------------------------------------------
FEATURES_ESCENARIO_A = [
    "distrito_enc", "mes_sin", "mes_cos", "indice_tiempo", "precio_chacra",
    "lag_prod_1", "lag_prod_3", "lag_prod_12", "lag_cosecha_1",
    "roll_mean_prod_3", "actividad_agricola",
]

FEATURES_ESCENARIO_B = [
    "distrito_enc", "mes_sin", "mes_cos", "indice_tiempo", "precio_chacra",
    "lag_prod_1", "lag_cosecha_1", "actividad_agricola",
]

FEATURES_ESCENARIO_C = [
    "distrito_enc", "mes_sin", "mes_cos", "indice_tiempo",
    "lag_prod_1", "actividad_agricola",
]

ESCENARIOS_DEF = {
    "A": {"nombre": "Escenario A (11 variables - Completo)", "features": FEATURES_ESCENARIO_A},
    "B": {"nombre": "Escenario B (8 variables - Reducido)", "features": FEATURES_ESCENARIO_B},
    "C": {"nombre": "Escenario C (6 variables - Optimizado)", "features": FEATURES_ESCENARIO_C},
}

# ---------------------------------------------------------------------------
# División temporal train / validation / test (Fase 4.1)
# ---------------------------------------------------------------------------
ANIO_VALID_START = 2023
ANIO_TEST = 2024

RANDOM_STATE = 42
N_ESTIMATORS = 300

# ---------------------------------------------------------------------------
# Scoring de inversión por distrito (Fase 5.9)
# ---------------------------------------------------------------------------
PESO_PRODUCCION = 0.70
PESO_CONFIABILIDAD = 0.30

UMBRALES_SCORE = {"Alta": 65, "Media": 40, "Baja": 0}

# ---------------------------------------------------------------------------
# Auditoría de series (Fase 6.1)
# ---------------------------------------------------------------------------
UMBRAL_REGISTROS_TEST = 12

# ---------------------------------------------------------------------------
# Paleta de colores (idéntica al dashboard HTML)
# ---------------------------------------------------------------------------
COLOR_VERDE_OSCURO = "#2d5016"
COLOR_VERDE_MEDIO = "#4a7c23"
COLOR_VERDE = "#2e7d32"
COLOR_GRIS_VERDE = "#4a6c4a"
COLOR_NARANJA = "#e65100"
COLOR_AZUL = "#1565c0"
COLOR_ALTA = "#2e7d32"
COLOR_MEDIA = "#f9a825"
COLOR_BAJA = "#c62828"

# ---------------------------------------------------------------------------
# Rutas de persistencia de artefactos (evita reentrenar en cada carga)
# ---------------------------------------------------------------------------
MODELS_DIR = "models_cache"
MODEL_PATH = f"{MODELS_DIR}/modelo_final.joblib"
ENCODERS_PATH = f"{MODELS_DIR}/encoders.joblib"
METADATA_PATH = f"{MODELS_DIR}/metadata.joblib"
DATA_HASH_PATH = f"{MODELS_DIR}/data_hash.txt"

ETAPAS_PIPELINE = [
    ("carga", "📂 Cargando datos", 5),
    ("exploracion", "🔍 Exploración de datos", 10),
    ("limpieza", "🧹 Limpieza de datos", 20),
    ("preparacion", "⚙️ Preparación y transformación", 35),
    ("features", "📊 Ingeniería de características", 50),
    ("entrenamiento", "🤖 Entrenamiento del modelo", 75),
    ("evaluacion", "📈 Evaluación del modelo", 85),
    ("prediccion", "🔮 Generación de predicciones", 92),
    ("dashboard", "📊 Construcción del dashboard", 98),
    ("fin", "✅ Proceso finalizado", 100),
]
