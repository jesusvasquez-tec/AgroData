# AgroData Perú — Dashboard de Predicción de Producción Agrícola

Aplicación web (Streamlit) que reemplaza al notebook `V2_PLAN_DE_TESIS_Cordova,_Vasquez.ipynb`
para uso final, manteniendo **exactamente** la misma lógica de datos, entrenamiento y scoring.

## Qué hace

1. El usuario sube un Excel/CSV con el formato de SISAGRI desde la barra lateral.
2. La app ejecuta automáticamente todo el pipeline (Fases 2–6 del CRISP-DM), mostrando
   una barra de progreso con cada etapa.
3. El dashboard se actualiza con 7 pestañas: Rankings de inversión, Predicción del
   próximo mes, Últimos 12 meses, Producción Anual, Comparativa Anual, Histórico
   Completo y Variables del Modelo.
4. Si vuelves a subir el **mismo archivo**, la app reutiliza el modelo ya entrenado
   (no reentrena ni recalcula predicciones) — ver "Single Source of Truth" abajo.

## Estructura del proyecto

```
agrodata_app/
├── app.py                      # Punto de entrada: UI, upload, progreso, tabs
├── config.py                   # ÚNICA fuente de parámetros (pesos, años, features, colores)
├── requirements.txt
├── utils/
│   ├── data_pipeline.py        # Fases 2-3: carga, filtrado, limpieza, panel temporal
│   ├── feature_engineering.py  # Fase 3.6-3.8: lags, rolling windows, encoding
│   └── pipeline_runner.py      # Orquestador + caché de modelo (evita reentrenar)
├── models/
│   ├── train.py                # Fase 4-5: escenarios A/B/C, selección automática, modelo final
│   └── predict.py               # Fase 5.7 y 5.9: dataset de predicciones y scoring
├── components/
│   └── tabs.py                  # Cada pestaña del dashboard (solo lectura de resultados)
├── models_cache/                 # (se genera en runtime) modelo .joblib + hash del dataset
└── tests/
    └── generar_datos_sinteticos.py   # Genera un CSV de prueba con el esquema de SISAGRI
```

## Cómo correrla localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre `http://localhost:8501`, sube tu Excel/CSV de SISAGRI y espera a que la barra de
progreso llegue a 100%.

### Formato de archivo esperado

El archivo debe tener estas columnas originales de SISAGRI (no hace falta renombrarlas,
la app lo hace internamente):

`anho, mes, Dpto, Prov, Dist, dsc_Cultivo, PRODUCCION(t), COSECHA (ha), MTO_PRECCHAC (S/ x kg)`

## Decisiones de arquitectura y mejoras aplicadas sobre el notebook original

Antes de migrar identifiqué estos puntos, corregidos en esta versión:

1. **Un solo lugar para llamar al modelo.** En el notebook, `modelo_final.predict()` se
   invocaba en varias celdas distintas (5.5.4, 5.7.1, gráficos de la 5.8). Aquí solo
   `models/predict.py` llama al modelo; todo el dashboard lee de ahí. Esto es lo que
   garantiza que "Producción Predicha" sea idéntica en todas las pestañas.
2. **No reentrenar en cada carga.** Se guarda un hash del DataFrame subido; si coincide
   con el de la última ejecución, se reutiliza el modelo `.joblib` guardado en
   `models_cache/` en lugar de repetir la búsqueda de escenarios A/B/C x 3 algoritmos
   (que es la parte costosa del pipeline).
3. **Selección de escenario ganador desacoplada de nombres de columna hardcodeados.**
   El notebook buscaba el escenario ganador comparando listas de features por
   referencia (`if config['features'] == VARIABLES_FINALES`); aquí `train.py` devuelve
   directamente el id del escenario ganador (`"A"`, `"B"` o `"C"`) sin depender de esa
   comparación fragil.
4. **Validación explícita de columnas y de tamaño de conjuntos**: si el archivo subido
   no tiene el esquema de SISAGRI, o si no cubre años suficientes para construir
   Train/Validation/Test (2020–2024 con la configuración por defecto), la app muestra
   un mensaje de error claro en vez de fallar con un traceback de pandas.
5. **`config.py` centraliza todos los parámetros** (pesos del scoring, años de corte,
   lags, ventanas, umbrales de zona) para que cambiar un valor no implique tocar
   código de negocio en varios archivos.

## Nota sobre el modelo ganador

El notebook define un mecanismo de selección automática entre 3 escenarios de
variables (A: 11, B: 8, C: 6) y 3 algoritmos (Random Forest, LightGBM, XGBoost),
eligiendo la combinación con menor WMAPE sobre el conjunto de validación. Esta app
replica ese mecanismo tal cual — el "modelo ganador" mostrado en el dashboard puede
variar según los datos que cargues (no está fijado a Random Forest + Escenario C).

## Despliegue gratuito

### Opción recomendada: Streamlit Community Cloud

Es la opción más simple para tu caso (proyecto de tesis, poco tráfico esperado,
prioridad en facilidad de mantenimiento):

1. Sube este proyecto a un repositorio de GitHub (puede ser privado).
2. Entra a https://share.streamlit.io con tu cuenta de GitHub.
3. "New app" → selecciona el repo, la rama y `app.py` como archivo principal.
4. Streamlit Cloud instala `requirements.txt` automáticamente y te da una URL pública
   (`https://tu-app.streamlit.app`).
5. Cada vez que hagas `git push`, la app se redespliega sola.

Limitaciones a tener en cuenta: 1 GB de RAM en el plan gratuito, la app "duerme" tras
un tiempo sin uso (tarda ~30s en despertar), y el almacenamiento no es persistente
entre reinicios — por eso `models_cache/` debe tratarse como una caché, no como base
de datos definitiva (si el contenedor se reinicia, la primera carga del día
reentrena una vez y listo).

### Alternativas y cuándo tendría sentido usarlas

| Opción | Gratis | Curva de aprendizaje | Rendimiento típico | Cuándo elegirla |
|---|---|---|---|---|
| **Streamlit Community Cloud** | Sí | Baja (ya lo tienes) | Bueno para dashboards de este tamaño | Tu caso: tesis, pocos usuarios simultáneos, quieres algo simple de mantener |
| **Hugging Face Spaces (Streamlit o Gradio)** | Sí | Baja-media | Similar o mejor (más RAM en el free tier en algunos casos) | Si quieres un link permanente sin que la app "duerma" tan agresivamente, y no te importa que quede alojado en HF |
| **Gradio** (en HF Spaces) | Sí | Baja | Muy bueno para demos ML puntuales | Si el foco es solo "sube un archivo, mira una predicción", con menos necesidad de tabs/dashboards complejos |
| **Render / Railway (Docker)** | Nivel gratuito limitado | Media | Configurable | Si más adelante necesitas un backend separado (API) del frontend |
| **Dash (Plotly) autohospedado** | Sí, pero necesitas dónde correrlo | Media-alta | Muy bueno para dashboards complejos | Si el proyecto crece y necesitas control total del layout, no si estás empezando con desarrollo web |

**Recomendación concreta:** quédate con Streamlit Community Cloud para la entrega de
tesis. Es gratuito, no requiere Docker ni servidores propios, y el código que armamos
ya está listo para subir tal cual. Si en el futuro el dashboard necesita atender
muchos usuarios simultáneos o requiere autenticación por cooperativa, ahí sí
valdría la pena migrar a un backend con Dash o FastAPI + un frontend separado.

## Solución de problemas comunes

- **"No se encontraron registros de PLATANO en ICA"**: revisa que las columnas
  `dsc_Cultivo` y `Dpto` en tu archivo usen esos textos (la app ya ignora mayúsculas/
  espacios, pero el cultivo/departamento deben existir en el archivo).
- **"La división Train/Validation/Test quedó vacía"**: tu archivo no cubre el rango
  2020–2024 configurado en `config.py` (`ANIO_VALID_START`, `ANIO_TEST`). Ajusta esos
  años si tu ventana de datos es distinta.
- **Quiero forzar que reentrene aunque sea el mismo archivo**: marca la casilla
  "Forzar reentrenamiento" en la barra lateral.
