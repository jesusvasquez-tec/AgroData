"""
utils/pipeline_runner.py
=========================
Orquesta el pipeline completo (Fases 2 a 6) y reporta progreso a la UI vía
un callback `on_progress(pct: int, mensaje: str)`.

Regla de oro (ver Fase 6 del notebook y el registro de decisiones del
proyecto): si el archivo cargado es EXACTAMENTE el mismo que el usado la
última vez (mismo hash), no se vuelve a entrenar el modelo — se reutiliza
el artefacto guardado en `models_cache/`. Esto evita recalcular
"Producción Predicha" de formas distintas entre sesiones.
"""

import hashlib
import os
import joblib
import pandas as pd

import config
from utils import data_pipeline as dp
from utils import feature_engineering as fe
from models import train as mtrain
from models import predict as mpredict


def _hash_dataframe(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def _tiene_cache_valido(data_hash: str) -> bool:
    if not os.path.exists(config.MODEL_PATH):
        return False
    if not os.path.exists(config.DATA_HASH_PATH):
        return False
    with open(config.DATA_HASH_PATH, "r") as f:
        cached_hash = f.read().strip()
    return cached_hash == data_hash


def _guardar_cache(data_hash, modelo_final, encoders, metadata):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(modelo_final, config.MODEL_PATH)
    joblib.dump(encoders, config.ENCODERS_PATH)
    joblib.dump(metadata, config.METADATA_PATH)
    with open(config.DATA_HASH_PATH, "w") as f:
        f.write(data_hash)


def _cargar_cache():
    modelo_final = joblib.load(config.MODEL_PATH)
    encoders = joblib.load(config.ENCODERS_PATH)
    metadata = joblib.load(config.METADATA_PATH)
    return modelo_final, encoders, metadata


def ejecutar_pipeline(df_raw: pd.DataFrame, on_progress=None, forzar_reentrenamiento=False):
    """
    Ejecuta el pipeline completo y devuelve un dict con todos los artefactos
    que el dashboard necesita. Es la ÚNICA función que el dashboard debe
    llamar; nunca debe reimplementar limpieza, features o predicción por su
    cuenta en las pestañas.
    """

    def progreso(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    # ---------------------------------------------------------------
    # Fase 2: Carga y validación
    # ---------------------------------------------------------------
    progreso(5, "📂 Cargando datos")
    df_raw = dp.cargar_y_validar(df_raw)
    data_hash = _hash_dataframe(df_raw)

    usar_cache = (not forzar_reentrenamiento) and _tiene_cache_valido(data_hash)

    progreso(10, "🔍 Exploración de datos")
    df_filtrado = dp.filtrar_cultivo_departamento(df_raw)

    # ---------------------------------------------------------------
    # Fase 3: Preparación de datos (siempre se recalcula: es barato y
    # garantiza que las tablas del dashboard reflejen el archivo actual)
    # ---------------------------------------------------------------
    progreso(20, "🧹 Limpieza de datos")
    df_clean = dp.seleccionar_y_renombrar(df_filtrado)
    df_clean = dp.estandarizar_tipos(df_clean)

    progreso(35, "⚙️ Preparación y transformación")
    df_paso1 = dp.filtrar_series_validas(df_clean)
    df_completo = dp.construir_panel_temporal(df_paso1)

    progreso(50, "📊 Ingeniería de características")
    df_features, df_prediccion_futura, ultimo_mes = fe.construir_features(df_completo)

    if usar_cache:
        progreso(75, "🤖 Reutilizando modelo ya entrenado (mismo archivo detectado)")
        modelo_final, encoders, metadata = _cargar_cache()
        features_finales = metadata["features_finales"]
        modelo_final_nombre = metadata["modelo_final_nombre"]
        escenario_ganador_id = metadata["escenario_ganador_id"]
        tabla_resultados = metadata["tabla_resultados"]
        metricas_finales = metadata["metricas_finales"]

        # Los encoders deben re-aplicarse al dataset recién procesado
        df_features["distrito_enc"] = encoders["distrito"].transform(df_features[["distrito"]]).astype(int)
        df_features["provincia_enc"] = encoders["provincia"].transform(df_features[["provincia"]]).astype(int)
        df_prediccion_futura["distrito_enc"] = encoders["distrito"].transform(df_prediccion_futura[["distrito"]]).astype(int)
        df_prediccion_futura["provincia_enc"] = encoders["provincia"].transform(df_prediccion_futura[["provincia"]]).astype(int)

        _, train_df, valid_df, test_df = mtrain.dividir_train_valid_test(df_features)
    else:
        df_features, df_prediccion_futura, encoders = fe.codificar_categoricas(df_features, df_prediccion_futura)

        progreso(60, "🤖 Entrenamiento del modelo (comparando escenarios)")
        _, train_df, valid_df, test_df = mtrain.dividir_train_valid_test(df_features)
        tabla_resultados, config_ganadora = mtrain.evaluar_escenarios(train_df, valid_df)

        progreso(75, "🤖 Entrenando el modelo ganador sobre Train+Validation")
        modelo_final, metricas_finales, _ = mtrain.entrenar_modelo_final(
            train_df, valid_df, test_df, config_ganadora
        )

        features_finales = config_ganadora["features"]
        modelo_final_nombre = config_ganadora["modelo"]
        escenario_ganador_id = config_ganadora["escenario"]

        metadata = {
            "features_finales": features_finales,
            "modelo_final_nombre": modelo_final_nombre,
            "escenario_ganador_id": escenario_ganador_id,
            "tabla_resultados": tabla_resultados,
            "metricas_finales": metricas_finales,
        }
        _guardar_cache(data_hash, modelo_final, encoders, metadata)

    progreso(85, "📈 Evaluación del modelo")
    df_imp = mtrain.feature_importance(modelo_final, features_finales)

    progreso(92, "🔮 Generación de predicciones")
    df_predicciones = mpredict.construir_dataset_predicciones(
        test_df, modelo_final, features_finales, modelo_final_nombre, escenario_ganador_id
    )
    df_proximo_mes, proximo_periodo = mpredict.predecir_proximo_mes(
        df_prediccion_futura, modelo_final, features_finales
    )
    scoring = mpredict.construir_scoring_distritos(df_predicciones)

    progreso(98, "📊 Construyendo el dashboard")

    resultado = {
        "data_hash": data_hash,
        "usar_cache": usar_cache,
        "df_model": df_features,
        "df_completo": df_completo,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "modelo_final": modelo_final,
        "modelo_final_nombre": modelo_final_nombre,
        "escenario_ganador_id": escenario_ganador_id,
        "features_finales": features_finales,
        "tabla_resultados": tabla_resultados,
        "metricas_finales": metricas_finales,
        "df_importancia": df_imp,
        "df_predicciones": df_predicciones,
        "df_proximo_mes": df_proximo_mes,
        "proximo_periodo": proximo_periodo,
        "ultimo_periodo": ultimo_mes,
        "scoring": scoring,
        "encoders": encoders,
    }

    progreso(100, "✅ Proceso finalizado")
    return resultado
