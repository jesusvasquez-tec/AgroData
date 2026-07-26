"""Genera un CSV sintético con el esquema de SISAGRI para probar el pipeline
end-to-end sin depender del archivo real del usuario."""
import numpy as np
import pandas as pd

np.random.seed(0)

DISTRITOS = ["ALTO LARAN", "SUNAMPE", "GROCIO PRADO", "EL CARMEN", "CHINCHA ALTA", "CHINCHA BAJA"]
PROVINCIA = "CHINCHA"
DEPARTAMENTO = "ICA"
CULTIVO = "PLATANO"

filas = []
for anio in range(2020, 2025):
    for mes in range(1, 13):
        if anio == 2024 and mes > 12:
            continue
        for dist in DISTRITOS:
            base = {"ALTO LARAN": 40, "SUNAMPE": 30, "GROCIO PRADO": 25,
                    "EL CARMEN": 35, "CHINCHA ALTA": 20, "CHINCHA BAJA": 15}[dist]
            estacional = 5 * np.sin(2 * np.pi * mes / 12)
            tendencia = (anio - 2020) * 1.5
            ruido = np.random.normal(0, 3)
            produccion = max(0, base + estacional + tendencia + ruido)
            cosecha = max(0, produccion / 8 + np.random.normal(0, 0.5))
            precio = max(0.5, 1.2 + 0.05 * np.sin(2 * np.pi * mes / 12) + np.random.normal(0, 0.05))
            filas.append({
                "anho": anio, "mes": mes, "Dpto": DEPARTAMENTO, "Prov": PROVINCIA, "Dist": dist,
                "dsc_Cultivo": CULTIVO, "PRODUCCION(t)": round(produccion, 2),
                "COSECHA (ha)": round(cosecha, 2), "MTO_PRECCHAC (S/ x kg)": round(precio, 3),
            })

df = pd.DataFrame(filas)
df.to_csv("/home/claude/agrodata_app/tests/sisagri_sintetico.csv", index=False)
print("Generado:", df.shape)
print(df.head())
