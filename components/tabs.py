"""
components/tabs.py
===================
Cada función `render_tab_xxx(resultado)` dibuja una pestaña del dashboard
usando ÚNICAMENTE los DataFrames que ya vienen calculados en `resultado`
(el dict que devuelve utils.pipeline_runner.ejecutar_pipeline).

Regla de single source of truth: ninguna función de este archivo llama a
modelo_final.predict(), reentrena nada, ni recalcula "producción predicha"
con una fórmula distinta a la de models/predict.py. Solo filtran/agrupan
para visualizar.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config

MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}


def _kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        with col:
            st.metric(label, value, help=sub)


def render_header_meta(resultado):
    m = resultado["metricas_finales"]
    _kpi_row([
        ("Último registro", str(resultado["ultimo_periodo"]), None),
        ("Predicción para", str(resultado["proximo_periodo"]), None),
        ("Modelo ganador", resultado["modelo_final_nombre"], f"Escenario {resultado['escenario_ganador_id']}"),
        ("MAE (Test)", f"{m['MAE']:.2f} t", None),
        ("R² (Test)", f"{m['R2']:.3f}", None),
        ("WMAPE (Test)", f"{m['WMAPE']:.2f}%", None),
    ])


# ---------------------------------------------------------------------------
# TAB 1: Rankings de inversión
# ---------------------------------------------------------------------------
def render_tab_rankings(resultado):
    st.subheader("🏆 Ranking de distritos por potencial productivo")
    st.caption(
        f"Score = {config.PESO_PRODUCCION*100:.0f}% producción predicha + "
        f"{config.PESO_CONFIABILIDAD*100:.0f}% confiabilidad del modelo (MAE inverso). "
        f"Zonas: Alta ≥ {config.UMBRALES_SCORE['Alta']}, Media ≥ {config.UMBRALES_SCORE['Media']}."
    )

    scoring = resultado["scoring"]
    color_map = {"Alta": config.COLOR_ALTA, "Media": config.COLOR_MEDIA, "Baja": config.COLOR_BAJA}

    fig = px.bar(
        scoring.sort_values("score_final"),
        x="score_final", y="distrito", color="zona",
        color_discrete_map=color_map, orientation="h",
        labels={"score_final": "Score Final", "distrito": "Distrito"},
        title="Score Final por Distrito",
    )
    fig.update_layout(height=420, xaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True, key="chart_rankings_score")

    # Producción real vs predicha agregada por distrito (mismo test set)
    df_pred = resultado["df_predicciones"]
    agg = df_pred.groupby("distrito", as_index=False).agg(
        produccion_total=("produccion", "sum"),
        produccion_predicha_total=("produccion_predicha", "sum"),
    )
    agg = agg.merge(scoring[["distrito", "score_final"]], on="distrito").sort_values("score_final")

    fig2 = go.Figure()
    fig2.add_bar(y=agg["distrito"], x=agg["produccion_total"], name="Producción Real (t)",
                 orientation="h", marker_color=config.COLOR_VERDE_MEDIO)
    fig2.add_bar(y=agg["distrito"], x=agg["produccion_predicha_total"], name="Producción Predicha (t)",
                 orientation="h", marker_color=config.COLOR_NARANJA)
    fig2.update_layout(barmode="group", height=420, title="Producción Total: Real vs. Predicha (conjunto Test)")
    st.plotly_chart(fig2, use_container_width=True, key="chart_rankings_produccion")

    tabla = scoring.rename(columns={
        "ranking": "Ranking", "distrito": "Distrito",
        "produccion_promedio_predicha": "Producción prom. predicha (t)",
        "mae_promedio": "MAE promedio (t)",
        "score_produccion": "Score Producción",
        "score_confiabilidad": "Score Confiabilidad",
        "score_final": "Score Final", "zona": "Zona",
    })
    st.dataframe(tabla.round(2), use_container_width=True, hide_index=True, key="tabla_rankings")

    csv = tabla.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Descargar ranking (CSV)", csv, "ranking_distritos.csv", "text/csv", key="dl_rankings")


# ---------------------------------------------------------------------------
# TAB 2: Predicción del próximo mes
# ---------------------------------------------------------------------------
def render_tab_prediccion(resultado):
    st.subheader(f"🔮 Predicción para {resultado['proximo_periodo']}")
    st.caption(
        "Comparación entre la producción predicha para el próximo mes y la producción "
        "real del mismo mes en el año anterior, por distrito."
    )

    df_prox = resultado["df_proximo_mes"].copy()
    df_prox["periodo_base"] = df_prox["periodo_base"].astype(str)
    df_prox["mes_predicho"] = df_prox["periodo_predicho"].astype(str).str[-2:]

    fig = go.Figure()
    fig.add_bar(
        x=df_prox["distrito"], y=df_prox["ultima_produccion_real"],
        name="Última producción real conocida (t)", marker_color=config.COLOR_VERDE_MEDIO,
    )
    fig.add_bar(
        x=df_prox["distrito"], y=df_prox["prod_pred_mes_actual"],
        name=f"Predicción {resultado['proximo_periodo']} (t)", marker_color=config.COLOR_NARANJA,
    )
    fig.update_layout(barmode="group", height=420, title="Producción: última real vs. próxima predicha, por distrito")
    st.plotly_chart(fig, use_container_width=True, key="chart_prediccion_proximo_mes")

    tabla = df_prox.rename(columns={
        "distrito": "Distrito", "periodo_base": "Período base",
        "ultima_produccion_real": "Última producción real (t)",
        "precio_chacra": "Precio chacra (S/)",
        "prod_pred_mes_actual": "Predicción próximo mes (t)",
        "periodo_predicho": "Período predicho",
    })[["Distrito", "Período base", "Última producción real (t)", "Precio chacra (S/)",
        "Predicción próximo mes (t)", "Período predicho"]]
    st.dataframe(tabla.round(2), use_container_width=True, hide_index=True, key="tabla_prediccion_proximo_mes")


# ---------------------------------------------------------------------------
# TAB 3: Últimos 12 meses / histórico con filtros
# ---------------------------------------------------------------------------
def _tabla_historica(resultado):
    df = resultado["df_predicciones"].copy()
    df["periodo_str"] = df["periodo"].astype(str)
    df["mes_nombre"] = df["mes"].map(MESES_ES)
    return df


def render_tab_historico(resultado, ultimos_n_meses=None, key_suffix=""):
    df = _tabla_historica(resultado)

    if ultimos_n_meses:
        periodos_ordenados = sorted(df["periodo"].unique())[-ultimos_n_meses:]
        df = df[df["periodo"].isin(periodos_ordenados)]
        st.subheader(f"📅 Últimos {ultimos_n_meses} meses (conjunto Test)")
    else:
        st.subheader("📜 Histórico completo (conjunto Test)")

    st.caption("Producción real vs. predicha. El error se calcula como WMAPE ponderado por volumen.")

    col1, col2 = st.columns(2)
    with col1:
        distritos_sel = st.multiselect(
            "Distrito", sorted(df["distrito"].unique()), key=f"dist_{key_suffix}"
        )
    with col2:
        meses_sel = st.multiselect(
            "Mes", sorted(df["mes"].unique()), format_func=lambda m: MESES_ES[m], key=f"mes_{key_suffix}"
        )

    if distritos_sel:
        df = df[df["distrito"].isin(distritos_sel)]
    if meses_sel:
        df = df[df["mes"].isin(meses_sel)]

    total_error_abs = df["error_abs"].sum()
    total_real = df["produccion"].sum()
    wmape = (total_error_abs / total_real * 100) if total_real > 0 else 0

    _kpi_row([
        ("Registros", f"{len(df):,}", None),
        ("Producción total", f"{df['produccion'].sum():,.0f} t", None),
        ("WMAPE ponderado", f"{wmape:.2f}%", None),
        ("Distritos", f"{df['distrito'].nunique()}", None),
    ])

    agg_periodo = df.groupby("periodo_str", as_index=False).agg(
        produccion_real=("produccion", "sum"), produccion_predicha=("produccion_predicha", "sum")
    ).sort_values("periodo_str")

    fig = go.Figure()
    fig.add_scatter(x=agg_periodo["periodo_str"], y=agg_periodo["produccion_real"],
                     name="Producción Real (t)", fill="tozeroy", line_color=config.COLOR_VERDE_MEDIO)
    fig.add_scatter(x=agg_periodo["periodo_str"], y=agg_periodo["produccion_predicha"],
                     name="Producción Predicha (t)", fill="tozeroy", line_color=config.COLOR_NARANJA)
    fig.update_layout(height=380, title="Tendencia de Producción Real vs Predicha")
    st.plotly_chart(fig, use_container_width=True, key=f"chart_historico_{key_suffix}")

    tabla = df[["distrito", "periodo_str", "produccion", "produccion_predicha", "error_abs", "error_pct"]].rename(
        columns={"distrito": "Distrito", "periodo_str": "Período", "produccion": "Real (t)",
                 "produccion_predicha": "Predicha (t)", "error_abs": "Error (t)", "error_pct": "Error (%)"}
    ).sort_values("Período")
    st.dataframe(tabla.round(2), use_container_width=True, hide_index=True, key=f"tabla_historico_{key_suffix}")

    csv = tabla.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Descargar tabla (CSV)", csv, f"historico_{key_suffix or 'completo'}.csv",
                        "text/csv", key=f"dl_{key_suffix}")


# ---------------------------------------------------------------------------
# TAB 4: Producción anual por distrito
# ---------------------------------------------------------------------------
def render_tab_anual(resultado):
    st.subheader("📈 Producción total por año y distrito")
    st.caption("Agregado anual de la producción histórica dentro del conjunto Test evaluado por el modelo.")

    df = resultado["df_predicciones"].copy()

    distrito_sel = st.multiselect("Distrito", sorted(df["distrito"].unique()), key="anual_dist")
    if distrito_sel:
        df = df[df["distrito"].isin(distrito_sel)]

    agg = df.groupby(["anio", "distrito", "departamento", "provincia"], as_index=False).agg(
        registros=("produccion", "count"),
        produccion_total=("produccion", "sum"),
        mae_promedio=("error_abs", "mean"),
    )

    fig = px.bar(
        agg.groupby("anio", as_index=False)["produccion_total"].sum(),
        x="anio", y="produccion_total", labels={"anio": "Año", "produccion_total": "Producción Total (t)"},
        title="Total de Producción por Año", color_discrete_sequence=[config.COLOR_VERDE_MEDIO],
    )
    st.plotly_chart(fig, use_container_width=True, key="chart_anual")

    tabla = agg.rename(columns={
        "anio": "Año", "distrito": "Distrito", "departamento": "Departamento", "provincia": "Provincia",
        "registros": "Registros", "produccion_total": "Producción Total (t)", "mae_promedio": "MAE Prom. (t)",
    })
    st.dataframe(tabla.round(2), use_container_width=True, hide_index=True, key="tabla_anual")


# ---------------------------------------------------------------------------
# TAB 5: Comparativa anual (real vs predicha)
# ---------------------------------------------------------------------------
def render_tab_comparativa(resultado):
    st.subheader("⚖️ Comparativa anual: real vs. predicha")

    df = resultado["df_predicciones"].copy()
    agg = df.groupby("anio", as_index=False).agg(
        registros=("produccion", "count"),
        produccion_real=("produccion", "sum"),
        produccion_predicha=("produccion_predicha", "sum"),
        distritos=("distrito", "nunique"),
    )
    agg["error_pct"] = np.where(
        agg["produccion_real"] > 0,
        np.abs(agg["produccion_real"] - agg["produccion_predicha"]) / agg["produccion_real"] * 100,
        np.nan,
    )

    _kpi_row([
        ("Producción total", f"{agg['produccion_real'].sum():,.0f} t", None),
        ("Error promedio", f"{agg['error_pct'].mean():.2f}%", None),
        ("Años registrados", f"{len(agg)}", None),
    ])

    col1, col2 = st.columns(2)
    with col1:
        fig1 = go.Figure()
        fig1.add_bar(x=agg["anio"], y=agg["produccion_real"], name="Real (t)", marker_color=config.COLOR_VERDE_MEDIO)
        fig1.add_bar(x=agg["anio"], y=agg["produccion_predicha"], name="Predicha (t)", marker_color=config.COLOR_NARANJA)
        fig1.update_layout(title="Producción por Año (Real vs Predicha)", height=380)
        st.plotly_chart(fig1, use_container_width=True, key="chart_comparativa_barras")
    with col2:
        fig2 = px.line(agg.sort_values("anio"), x="anio", y="produccion_real", markers=True,
                        title="Evolución Anual de la Producción Real",
                        color_discrete_sequence=[config.COLOR_VERDE_MEDIO])
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True, key="chart_comparativa_linea")

    tabla = agg.rename(columns={
        "anio": "Año", "registros": "Registros", "produccion_real": "Real (t)",
        "produccion_predicha": "Predicha (t)", "error_pct": "Error (%)", "distritos": "Distritos",
    })
    st.dataframe(tabla.round(2), use_container_width=True, hide_index=True, key="tabla_comparativa")


# ---------------------------------------------------------------------------
# TAB 6: Variables del modelo
# ---------------------------------------------------------------------------
def render_tab_variables(resultado):
    st.subheader("🧬 Variables del modelo ganador")
    st.caption(
        f"Modelo: **{resultado['modelo_final_nombre']}** · "
        f"Escenario: **{resultado['escenario_ganador_id']}** · "
        f"{len(resultado['features_finales'])} variables predictoras."
    )

    df_imp = resultado["df_importancia"]
    if not df_imp.empty:
        fig = px.bar(
            df_imp.sort_values("importancia"), x="importancia", y="variable", orientation="h",
            title="Importancia de variables (feature_importances_)",
            color_discrete_sequence=[config.COLOR_AZUL],
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_variables_importancia")
    else:
        st.info("El modelo ganador no expone `feature_importances_` de forma nativa.")

    st.markdown("**Tabla comparativa de escenarios evaluados (Validation):**")
    tabla_resultados = resultado["tabla_resultados"].rename(columns={
        "Escenario": "Escenario", "Modelo": "Modelo", "MAE": "MAE", "RMSE": "RMSE",
        "R2": "R²", "WMAPE": "WMAPE (%)", "num_features": "N° variables",
    })
    st.dataframe(
        tabla_resultados.sort_values("WMAPE (%)").round(3), use_container_width=True, hide_index=True,
        key="tabla_variables_escenarios",
    )

    with st.expander("Ver definición completa de variables por escenario"):
        for id_esc, cfg in config.ESCENARIOS_DEF.items():
            st.markdown(f"**{cfg['nombre']}**")
            st.write(cfg["features"])
