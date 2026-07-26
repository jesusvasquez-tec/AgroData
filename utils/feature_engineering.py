"""
utils/feature_engineering.py
=============================
Replica exacta de la Fase 3.6-3.8 del notebook: lags, ventanas móviles,
rachas de inactividad, codificación ordinal de distrito/provincia y la
separación del último mes disponible para la predicción futura ("next month").
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import config


def _calcular_racha(df, grupo_por, variable_binaria, nombre_columna):
    valor_anterior = df.groupby(grupo_por)[variable_binaria].shift(1)
    cambio_estado = df[variable_binaria].ne(valor_anterior)
    id_racha = cambio_estado.cumsum()
    racha = df.groupby([*grupo_por, id_racha]).cumcount() + 1
    df[nombre_columna] = np.where(df[variable_binaria] == 1, racha, 0)
    return df


def _meses_desde_ultima_prod(serie):
    contador = 0
    resultado = []
    for valor in serie:
        contador = 0 if valor > 0 else contador + 1
        resultado.append(contador)
    return resultado


def construir_features(df_completo: pd.DataFrame):
    """
    Devuelve (df_features, df_prediccion_futura, ultimo_mes_features).

    df_features         -> registros con target conocido, listos para modelar.
    df_prediccion_futura -> última fila de cada serie, para predecir el próximo mes.
    """
    target_real = "produccion"
    target_model = "produccion_log"

    df_features = df_completo.copy().sort_values(config.CLAVES_SERIE + ["periodo"]).reset_index(drop=True)
    grupo_series = df_features.groupby(config.CLAVES_SERIE)

    df_features["sin_produccion"] = (df_features["produccion"] == 0).astype(int)
    df_features = _calcular_racha(df_features, config.CLAVES_SERIE, "sin_produccion", "racha_inactiva")

    df_features["meses_desde_ultima_prod"] = grupo_series["produccion"].transform(_meses_desde_ultima_prod)

    for lag in config.LAGS:
        df_features[f"lag_prod_{lag}"] = grupo_series[target_model].shift(lag)
        df_features[f"lag_cosecha_{lag}"] = grupo_series["cosecha_log"].shift(lag)
        df_features[f"lag_precio_{lag}"] = grupo_series["precio_chacra"].shift(lag)

    for window in config.ROLL_WINDOWS:
        df_features[f"roll_mean_prod_{window}"] = grupo_series[target_model].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df_features[f"roll_std_prod_{window}"] = (
            grupo_series[target_model]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).std())
            .fillna(0)
        )
        df_features[f"roll_max_prod_{window}"] = grupo_series[target_model].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        )

    df_features["target_log"] = grupo_series[target_model].shift(-config.FORECAST_HORIZON)
    df_features["target_real"] = grupo_series[target_real].shift(-config.FORECAST_HORIZON)

    ultimo_mes_features = df_features["periodo"].max()
    df_prediccion_futura = df_features[df_features["periodo"] == ultimo_mes_features].copy()

    df_features = df_features.dropna(subset=["target_log"]).copy()

    return df_features, df_prediccion_futura, ultimo_mes_features


def codificar_categoricas(df_features: pd.DataFrame, df_prediccion_futura: pd.DataFrame):
    """Fase 3.7: OrdinalEncoder robusto a distritos/provincias no vistos (-1)."""
    encoder_distrito = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder_provincia = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    df_features = df_features.copy()
    df_prediccion_futura = df_prediccion_futura.copy()

    df_features["distrito_enc"] = encoder_distrito.fit_transform(df_features[["distrito"]]).astype(int)
    df_features["provincia_enc"] = encoder_provincia.fit_transform(df_features[["provincia"]]).astype(int)

    df_prediccion_futura["distrito_enc"] = encoder_distrito.transform(df_prediccion_futura[["distrito"]]).astype(int)
    df_prediccion_futura["provincia_enc"] = encoder_provincia.transform(df_prediccion_futura[["provincia"]]).astype(int)

    encoders = {"distrito": encoder_distrito, "provincia": encoder_provincia}
    return df_features, df_prediccion_futura, encoders
