"""
models/train.py
================
Replica exacta de las Fases 4 (Modelado) y 5 (Evaluación) del notebook:

- División temporal Train / Validation / Test (por año de corte).
- Entrenamiento y comparación de 3 escenarios de variables x 3 algoritmos
  (Random Forest, LightGBM, XGBoost) sobre Validation.
- Selección automática del ganador por menor WMAPE.
- Reentrenamiento del modelo ganador sobre Train+Validation.
- Evaluación final sobre Test (MAE, RMSE, R², WMAPE).

IMPORTANTE (single source of truth): estas funciones se ejecutan UNA sola
vez por dataset cargado (ver utils/pipeline_runner.py, que cachea el
resultado). El dashboard nunca reentrena ni recalcula predicciones: solo
lee lo que esta capa produjo.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
import lightgbm as lgb
import xgboost as xgb

import config


def calcular_metricas(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    total_real = np.sum(y_real)
    wmape = (np.abs(y_real - y_pred).sum() / total_real * 100) if total_real > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "WMAPE": wmape}


def dividir_train_valid_test(df_features: pd.DataFrame):
    lag_maximo = max(config.LAGS)
    df_model = df_features.dropna(subset=[f"lag_prod_{lag_maximo}"]).copy()

    train_df = df_model[df_model["anio"] < config.ANIO_VALID_START].copy()
    valid_df = df_model[df_model["anio"] == config.ANIO_VALID_START].copy()
    test_df = df_model[df_model["anio"] == config.ANIO_TEST].copy()

    if len(train_df) == 0 or len(valid_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "La división Train/Validation/Test quedó vacía en algún conjunto. "
            f"Verifica que el archivo cargado cubra hasta el año {config.ANIO_TEST} "
            f"(ANIO_VALID_START={config.ANIO_VALID_START}, ANIO_TEST={config.ANIO_TEST} en config.py)."
        )

    return df_model, train_df, valid_df, test_df


def _instanciar_modelos():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=config.N_ESTIMATORS, learning_rate=0.05, max_depth=6,
            random_state=config.RANDOM_STATE, verbose=-1,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=config.N_ESTIMATORS, learning_rate=0.05, max_depth=6,
            random_state=config.RANDOM_STATE, objective="reg:squarederror", n_jobs=-1,
        ),
    }


def evaluar_escenarios(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    """Fase 5.3: entrena 3 escenarios x 3 modelos sobre Train, evalúa en Validation."""
    modelos = _instanciar_modelos()
    todos_los_resultados = []
    mejor_wmape = float("inf")
    config_ganadora = None

    for id_escenario, cfg in config.ESCENARIOS_DEF.items():
        features = cfg["features"]
        X_train_esc = train_df[features].fillna(0)
        y_train_esc = train_df["target_log"]
        X_valid_esc = valid_df[features].fillna(0)
        y_valid_real = np.expm1(valid_df["target_log"])

        for nombre_modelo, modelo in modelos.items():
            modelo.fit(X_train_esc, y_train_esc)
            pred_log = modelo.predict(X_valid_esc)
            pred_real = np.maximum(np.expm1(pred_log), 0)
            metricas = calcular_metricas(y_valid_real, pred_real)

            resultado = {
                "Escenario": id_escenario,
                "Modelo": nombre_modelo,
                "MAE": round(metricas["MAE"], 2),
                "RMSE": round(metricas["RMSE"], 2),
                "R2": round(metricas["R2"], 3),
                "WMAPE": round(metricas["WMAPE"], 2),
                "num_features": len(features),
            }
            todos_los_resultados.append(resultado)

            if metricas["WMAPE"] < mejor_wmape:
                mejor_wmape = metricas["WMAPE"]
                config_ganadora = {
                    "escenario": id_escenario,
                    "features": features.copy(),
                    "modelo": nombre_modelo,
                    "wmape": metricas["WMAPE"],
                }

    tabla_resultados = pd.DataFrame(todos_los_resultados)
    return tabla_resultados, config_ganadora


def entrenar_modelo_final(train_df, valid_df, test_df, config_ganadora):
    """Fase 5.5: reentrena el ganador sobre Train+Validation y evalúa en Test."""
    features_finales = config_ganadora["features"]
    modelo_final_nombre = config_ganadora["modelo"]

    train_final = pd.concat([train_df, valid_df], ignore_index=True)
    X_train_final = train_final[features_finales].fillna(0)
    y_train_final = train_final["target_log"]

    X_test = test_df[features_finales].fillna(0)
    y_test = np.expm1(test_df["target_log"])

    modelos = _instanciar_modelos()
    modelo_final = clone(modelos[modelo_final_nombre])
    modelo_final.fit(X_train_final, y_train_final)

    pred_test = np.maximum(np.expm1(modelo_final.predict(X_test)), 0)
    metricas_finales = calcular_metricas(y_test, pred_test)

    return modelo_final, metricas_finales, pred_test


def feature_importance(modelo_final, features_finales):
    if not hasattr(modelo_final, "feature_importances_"):
        return pd.DataFrame(columns=["variable", "importancia"])
    df_imp = pd.DataFrame({
        "variable": features_finales,
        "importancia": modelo_final.feature_importances_,
    }).sort_values("importancia", ascending=False).reset_index(drop=True)
    return df_imp
