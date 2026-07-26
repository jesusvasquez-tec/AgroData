"""
utils/data_pipeline.py
======================
Replica exacta de las Fases 2 (Comprensión de datos) y 3 (Preparación de
datos) del notebook original: carga, filtrado por cultivo/departamento,
selección y renombrado de variables, validación de series y construcción
del panel temporal mensual sin huecos.

Cada función es pura (recibe DataFrame, devuelve DataFrame) para poder
testearla de forma aislada y para que el orquestador (pipeline_runner.py)
controle el progreso mostrado al usuario.
"""

import numpy as np
import pandas as pd

import config


class PipelineError(Exception):
    """Error de negocio del pipeline (columnas faltantes, datos vacíos, etc.)."""


def cargar_y_validar(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Valida que el archivo cargado tenga las columnas mínimas de SISAGRI."""
    faltantes = [c for c in config.COLUMNAS_ORIGINALES_REQUERIDAS if c not in df_raw.columns]
    if faltantes:
        raise PipelineError(
            "El archivo cargado no tiene el formato esperado de SISAGRI. "
            f"Faltan las columnas: {faltantes}. "
            f"Columnas encontradas: {list(df_raw.columns)}"
        )
    return df_raw.copy()


def filtrar_cultivo_departamento(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Fase 2.3: filtra únicamente Plátano en Ica (parametrizable en config.py)."""
    df = df_raw.copy()
    df = df[df["dsc_Cultivo"].str.upper().str.strip() == config.CULTIVO_OBJETIVO]
    df = df[df["Dpto"].str.upper().str.strip() == config.DEPARTAMENTO_OBJETIVO]
    if df.empty:
        raise PipelineError(
            f"No se encontraron registros de '{config.CULTIVO_OBJETIVO}' en "
            f"'{config.DEPARTAMENTO_OBJETIVO}' dentro del archivo cargado."
        )
    return df


def seleccionar_y_renombrar(df_filtrado: pd.DataFrame) -> pd.DataFrame:
    """Fase 3.1.2 + 3.2.1: selección de variables representativas y renombrado."""
    df_clean = df_filtrado[config.VARIABLES_SELECCIONADAS].copy()
    df_clean = df_clean.rename(columns=config.RENOMBRE_COLUMNAS)
    return df_clean


def estandarizar_tipos(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Fase 3.2.2: construcción del período mensual y tipado numérico/categórico."""
    df = df_clean.copy()
    df["periodo"] = pd.to_datetime(
        df["anio"].astype(str) + "-" + df["mes"].astype(str), format="%Y-%m"
    ).dt.to_period("M")

    for col in ["departamento", "provincia", "distrito", "cultivo"]:
        df[col] = df[col].astype(str)

    for col in ["cosecha", "produccion", "precio_chacra"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def filtrar_series_validas(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Fase 3.4: filtrado paramétrico y validación de series temporales."""
    df_clean = df_clean.copy()
    df_clean["precio_chacra"] = df_clean["precio_chacra"].replace(0, np.nan)

    df_paso1 = df_clean[df_clean["anio"] >= config.START_YEAR].copy()

    for col in config.CLAVES_SERIE:
        df_paso1[col] = df_paso1[col].astype(str).str.strip().str.upper()

    ultimo_anio_datos = df_paso1["anio"].max()
    ultimo_periodo_ds = df_paso1["periodo"].max()

    cov_anual = (
        df_paso1[df_paso1["anio"] < ultimo_anio_datos]
        .groupby(config.CLAVES_SERIE + ["anio"])
        .agg(
            meses_registrados=("mes", "nunique"),
            meses_con_produccion=("produccion", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    resumen_series = (
        cov_anual.groupby(config.CLAVES_SERIE)
        .agg(
            anios_completos=("anio", "nunique"),
            minimo_meses_anio=("meses_registrados", "min"),
            promedio_meses_anio=("meses_registrados", "mean"),
            minimo_meses_produccion=("meses_con_produccion", "min"),
            promedio_meses_produccion=("meses_con_produccion", "mean"),
            total_meses_historicos=("meses_registrados", "sum"),
        )
        .reset_index()
    )

    filtro_historico = (
        resumen_series["minimo_meses_anio"] >= config.MIN_MESES_POR_ANIO
    ) & (resumen_series["total_meses_historicos"] >= config.MIN_TOTAL_MESES_SERIE)

    if config.USAR_FILTRO_PRODUCCION:
        filtro_historico &= (
            resumen_series["promedio_meses_produccion"] >= config.MIN_MESES_PRODUCCION
        )

    series_validas = resumen_series[filtro_historico].copy()

    periodo_inicio_ventana = ultimo_periodo_ds - config.VENTANA_MESES_RECIENTES
    meses_en_ventana = (
        df_paso1[
            (df_paso1["periodo"] > periodo_inicio_ventana)
            & (df_paso1["periodo"] <= ultimo_periodo_ds)
        ]
        .groupby(config.CLAVES_SERIE)["mes"]
        .nunique()
        .reset_index()
        .rename(columns={"mes": "meses_en_ventana"})
    )

    series_activas = meses_en_ventana[
        meses_en_ventana["meses_en_ventana"] >= config.MIN_MESES_RECIENTES
    ][config.CLAVES_SERIE].copy()

    series_validas_final = series_validas.merge(series_activas, on=config.CLAVES_SERIE, how="inner")

    if series_validas_final.empty:
        raise PipelineError(
            "Ninguna serie distrito/cultivo cumple los criterios mínimos de "
            "cobertura histórica y actividad reciente definidos en config.py. "
            "Revisa START_YEAR, MIN_TOTAL_MESES_SERIE y VENTANA_MESES_RECIENTES."
        )

    df_paso1 = df_paso1.merge(series_validas_final[config.CLAVES_SERIE], on=config.CLAVES_SERIE, how="inner")
    return df_paso1


def construir_panel_temporal(df_paso1: pd.DataFrame) -> pd.DataFrame:
    """Fase 3.5: panel mensual continuo por serie, con imputación causal de precio."""
    df_paso1 = df_paso1.sort_values(config.CLAVES_SERIE + ["periodo"]).reset_index(drop=True)

    duplicados = df_paso1.duplicated(subset=config.CLAVES_SERIE + ["periodo"]).sum()
    if duplicados > 0:
        raise PipelineError(
            f"Existen {duplicados:,} registros duplicados por serie-período en el "
            "archivo cargado. Revisa el archivo antes de continuar."
        )

    rangos_series = (
        df_paso1.groupby(config.CLAVES_SERIE)
        .agg(periodo_inicio=("periodo", "min"), periodo_fin=("periodo", "max"))
        .reset_index()
    )

    def construir_panel_serie(fila):
        if config.USAR_RANGO_REAL_SERIE:
            rango = pd.period_range(start=fila["periodo_inicio"], end=fila["periodo_fin"], freq="M")
        else:
            rango = pd.period_range(start=f"{config.START_YEAR}-01", end=f"{fila['periodo_fin']}", freq="M")
        panel = pd.DataFrame({"periodo": rango})
        for col in config.CLAVES_SERIE:
            panel[col] = fila[col]
        return panel

    paneles = [construir_panel_serie(fila) for _, fila in rangos_series.iterrows()]
    esqueleto = pd.concat(paneles, ignore_index=True)
    esqueleto["anio"] = esqueleto["periodo"].dt.year
    esqueleto["mes"] = esqueleto["periodo"].dt.month

    df_completo = esqueleto.merge(df_paso1, on=config.CLAVES_SERIE + ["anio", "mes", "periodo"], how="left")

    df_completo["dato_original"] = df_completo["produccion"].notna().astype(int)
    df_completo["fila_creada"] = (df_completo["dato_original"] == 0).astype(int)

    for col in ["cosecha", "produccion"]:
        df_completo[col] = pd.to_numeric(df_completo[col], errors="coerce").fillna(0)

    df_completo["actividad_agricola"] = (
        (df_completo["produccion"] > 0) | (df_completo["cosecha"] > 0)
    ).astype(int)

    df_completo["precio_chacra"] = pd.to_numeric(df_completo["precio_chacra"], errors="coerce")
    df_completo.loc[
        (df_completo["actividad_agricola"] == 0) & (df_completo["precio_chacra"] == 0),
        "precio_chacra",
    ] = np.nan

    if config.USAR_IMPUTACION_PRECIO:
        df_completo["precio_chacra"] = df_completo.groupby(config.CLAVES_SERIE)["precio_chacra"].transform(
            lambda x: x.ffill()
        )
        if config.USAR_BACKFILL_PRECIO:
            df_completo["precio_chacra"] = df_completo.groupby(config.CLAVES_SERIE)["precio_chacra"].transform(
                lambda x: x.bfill()
            )

    df_completo["mes_sin"] = np.sin(2 * np.pi * df_completo["mes"] / 12)
    df_completo["mes_cos"] = np.cos(2 * np.pi * df_completo["mes"] / 12)
    df_completo["trimestre"] = ((df_completo["mes"] - 1) // 3) + 1
    df_completo["indice_tiempo"] = (df_completo["anio"] - df_completo["anio"].min()) * 12 + (
        df_completo["mes"] - 1
    )

    df_completo["produccion_log"] = np.log1p(df_completo["produccion"])
    df_completo["cosecha_log"] = np.log1p(df_completo["cosecha"])

    df_completo = df_completo.sort_values(config.CLAVES_SERIE + ["periodo"]).reset_index(drop=True)
    return df_completo
