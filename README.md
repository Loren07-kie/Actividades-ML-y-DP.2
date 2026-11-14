# MotorGuard SENATI — detección de anomalías en motores

**Autora:** Lorena Abigail Coca Ccorpa · **Año:** 2026

Aplicación Flask de aprendizaje **no supervisado** con Isolation Forest. Inspirada en los archivos de la actividad sobre anomalías en motores; nueva interfaz y documentación propia.

## Descripción
- 318 observaciones **simuladas**: 300 habituales + 18 casos sintéticos atípicos.
- Variables: RPM, temperatura (°C), vibración (mm/s), corriente (A).
- Estandarización + Isolation Forest (200 árboles, contamination=0.06).
- Formulario web y API `POST /api/predict`, además de `GET /health`.
- Cuaderno: `notebooks/MotorGuard_Documentacion.ipynb`.
- Logotipo SENATI incluido como recurso SVG identificativo; uso académico.

## Uso local
```bash
python -m pip install -r requirements.txt
python app.py
```
Visitar http://localhost:5000

## Render
Servicio web Python, build `pip install -r requirements.txt`, start `gunicorn app:app --bind 0.0.0.0:$PORT`.

## Limitaciones
Datos ficticios para docencia: **no** sustituye una inspección de ingeniería ni es diagnóstico de falla. Las salidas son relativas al conjunto generado, no probabilidades clínicas ni industriales.
