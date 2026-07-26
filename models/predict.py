"""
models/predict.py
==================
Replica exacta de las Fases 5.7 (Dataset final de predicciones) y 5.9
(Scoring de distritos para inversión) del notebook, más la predicción
"próximo mes" (Fase 6) usando la última fila disponible de cada serie.

Todo lo que consume el dashboard sale de aquí: es el ÚNICO lugar donde se
llama a modelo_final.predict(). Ninguna pestaña del dashboard debe volver
a invocar el modelo ni recalcular producción predicha por su cuenta.
"""

import numpy as np
import pandas as pd

import config


def construir_dataset_predicciones(test_df, modelo_final, features_finales, modelo_final_nombre, escenario_ganador_id):
    """Fase 5.7.1: dataset final de predicciones sobre el conjunto Test."""
    X_test = test_df[features_finales].fillna(0)
    pred_test = np.maximum(np.expm1(modelo_final.predict(X_test)), 0)

    df_predicciones = test_df[[
        "departamento", "provincia", "distrito", "cultivo", "anio", "mes", "periodo", "produccion",
    ]].copy()
    df_predicciones["produccion_predicha"] = pred_test
    df_predicciones["error_abs"] = np.abs(df_predicciones["produccion"] - df_predicciones["produccion_predicha"])
    df_predicciones["error_pct"] = np.where(
        df_predicciones["produccion"] > 0,
        df_predicciones["error_abs"] / df_predicciones["produccion"] * 100,
        np.nan,
    )
    df_predicciones["modelo"] = modelo_final_nombre
    df_predicciones["escenario"] = f"Escenario {escenario_ganador_id}"
    return df_predicciones


def predecir_proximo_mes(df_prediccion_futura, modelo_final, features_finales):
    """
    Genera la predicción del mes siguiente al último dato disponible, por serie
    (distrito). Se apoya en la fila 'df_prediccion_futura' calculada durante el
    feature engineering (última fila real de cada serie, con sus lags ya listos).
    """
    df_pred_futura = df_prediccion_futura.copy()
    X_futuro = df_pred_futura[features_finales].fillna(0)
    pred_log = modelo_final.predict(X_futuro)
    df_pred_futura["prod_pred_mes_actual"] = np.maximum(np.expm1(pred_log), 0)

    proximo_periodo = df_pred_futura["periodo"].max() + config.FORECAST_HORIZON

    resultado = df_pred_futura[[
        "departamento", "provincia", "distrito", "cultivo", "periodo",
        "produccion", "precio_chacra", "prod_pred_mes_actual",
    ]].copy()
    resultado = resultado.rename(columns={"periodo": "periodo_base", "produccion": "ultima_produccion_real"})
    resultado["periodo_predicho"] = str(proximo_periodo)
    return resultado, proximo_periodo


def construir_scoring_distritos(df_predicciones: pd.DataFrame) -> pd.DataFrame:
    """Fase 5.9.1-5.9.2: scoring 70% producción / 30% confiabilidad (MAE)."""
    peso_prod = config.PESO_PRODUCCION
    peso_conf = config.PESO_CONFIABILIDAD
    assert abs(peso_prod + peso_conf - 1.0) < 0.01

    scoring = (
        df_predicciones.groupby("distrito", as_index=False)
        .agg({"produccion_predicha": "mean", "error_abs": "mean"})
    )
    scoring.columns = ["distrito", "produccion_promedio_predicha", "mae_promedio"]

    scoring["score_produccion"] = (
        scoring["produccion_promedio_predicha"] / scoring["produccion_promedio_predicha"].max()
    ) * 100

    scoring["score_confiabilidad"] = (
        1 - (scoring["mae_promedio"] / scoring["mae_promedio"].max())
    ) * 100

    scoring["score_final"] = peso_prod * scoring["score_produccion"] + peso_conf * scoring["score_confiabilidad"]

    def clasificar_zona(score):
        if score >= config.UMBRALES_SCORE["Alta"]:
            return "Alta"
        elif score >= config.UMBRALES_SCORE["Media"]:
            return "Media"
        return "Baja"

    scoring["zona"] = scoring["score_final"].apply(clasificar_zona)
    scoring = scoring.sort_values("score_final", ascending=False).reset_index(drop=True)
    scoring.insert(0, "ranking", range(1, len(scoring) + 1))
    return scoring
