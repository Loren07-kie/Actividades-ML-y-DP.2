"""MotorGuard SENATI — detector educativo de anomalías, datos simulados."""
import math
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template_string
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
FEATURES = ["rpm", "temperatura_c", "vibracion_mm_s", "corriente_a"]
LIMITS = [(0, 3600), (0, 150), (0, 30), (0, 60)]

def generate_data(seed=42):
    rng = np.random.default_rng(seed)
    n = 300
    rpm = rng.uniform(900, 1800, n)
    usual = pd.DataFrame({
        "rpm": rpm,
        "temperatura_c": 40 + 0.018*(rpm-900) + rng.normal(0, 3, n),
        "vibracion_mm_s": 1.2 + 0.0012*(rpm-900) + rng.normal(0, .25, n),
        "corriente_a": 4 + .004*(rpm-900) + rng.normal(0, .5, n)})
    unusual = pd.DataFrame({
        "rpm": rng.uniform(900,1800,18),
        "temperatura_c": np.r_[rng.uniform(78,95,6), rng.uniform(42,60,6),rng.uniform(70,85,6)],
        "vibracion_mm_s": np.r_[rng.uniform(1.5,3,6),rng.uniform(5.5,8.5,6),rng.uniform(2,3.5,6)],
        "corriente_a": np.r_[rng.uniform(5,8,6),rng.uniform(4,7,6),rng.uniform(11,15,6)]})
    return pd.concat([usual, unusual], ignore_index=True).sample(frac=1,random_state=1).reset_index(drop=True).round(2)

data = generate_data()
model = make_pipeline(StandardScaler(), IsolationForest(n_estimators=200,contamination=.06,random_state=42))
model.fit(data[FEATURES])
flagged = int((model.predict(data[FEATURES]) == -1).sum())

HTML = r"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MotorGuard | SENATI · Lorena Coca</title><link rel="icon" href="/static/logo.svg">
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#07101d;--panel:#111c2d;--line:#2c3c52;--accent:#27ddbd;--ink:#edf5ff;--muted:#a6b5ca}
*{box-sizing:border-box}body{margin:0;color:var(--ink);font-family:'DM Sans',sans-serif;background:radial-gradient(circle at 85% 0,#163b4c,transparent 40%),var(--bg)}
header{display:flex;align-items:center;justify-content:space-between;gap:15px;padding:22px max(5vw,20px);border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:16px}.brand img{width:120px;background:#fff;border-radius:10px;padding:6px}
small,.muted{color:var(--muted)}.tag{color:var(--accent);letter-spacing:.14em;font-size:12px;font-weight:700}
main{max-width:1200px;margin:auto;padding:54px 24px}h1,h2,h3{font-family:'Space Grotesk',sans-serif}h1{font-size:clamp(42px,6vw,76px);line-height:1.05;letter-spacing:-.055em;margin:16px 0 18px}h1 em{font-style:normal;color:var(--accent)}
.lead{color:#bfcede;font-size:18px;max-width:680px;line-height:1.6}.grid{display:grid;grid-template-columns:1.15fr .85fr;gap:24px;margin-top:36px}
.card{background:#101d2eeb;border:1px solid var(--line);border-radius:24px;padding:28px;box-shadow:0 16px 50px #0003}.card h2{margin-top:0}
.formgrid{display:grid;grid-template-columns:1fr 1fr;gap:17px}label{display:block;color:#c4d0df;font-size:13px;font-weight:700;margin-bottom:8px}
input{width:100%;padding:16px;background:#081323;border:1px solid #3b4e62;border-radius:12px;color:#fff;font-size:18px;outline:none}input:focus{border-color:var(--accent)}
button{background:var(--accent);color:#06241f;border:0;padding:16px 28px;border-radius:12px;font-size:15px;font-weight:800;cursor:pointer;margin-top:24px;width:100%}button:hover{filter:brightness(1.12)}
.statgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.stat{border-radius:15px;background:#17283a;padding:18px}.stat strong{font-size:31px;display:block;color:var(--accent)}
#result{margin-top:20px;border:1px dashed #49627a;padding:20px;border-radius:14px;min-height:120px;color:#b7c9da}.good{color:#4be2b4}.bad{color:#ff9c87}
footer{padding:25px;text-align:center;color:var(--muted);border-top:1px solid var(--line)}
@media(max-width:780px){.grid{grid-template-columns:1fr}.formgrid{grid-template-columns:1fr}header{flex-wrap:wrap}main{padding-top:25px}}
</style></head><body>
<header><div class="brand"><img src="/static/logo.svg" alt="SENATI"><div><b>MOTORGUARD</b><br><small>Laboratorio de Machine Learning</small></div></div><div class="tag">PROYECTO ACADÉMICO · 2026</div></header>
<main><span class="tag">INTELIGENCIA ARTIFICIAL · APRENDIZAJE NO SUPERVISADO</span>
<h1>Descubre lo <em>inusual</em><br>antes de que pase inadvertido.</h1>
<p class="lead">Explora un sistema educativo de detección de anomalías en motores con Isolation Forest. Introduce los valores del equipo y analiza su similitud con los patrones de una base simulada.</p>
<section class="grid"><div class="card"><h2>Analizar un motor</h2><p class="muted">Completa las cuatro variables de operación.</p><form id="motor"><div class="formgrid">
<div><label for="rpm">Velocidad · RPM</label><input id="rpm" name="rpm" type="number" step="any" min="0" max="3600" value="1450" required></div>
<div><label for="temperatura_c">Temperatura · °C</label><input id="temperatura_c" name="temperatura_c" type="number" step="any" min="0" max="150" value="55" required></div>
<div><label for="vibracion_mm_s">Vibración · mm/s</label><input id="vibracion_mm_s" name="vibracion_mm_s" type="number" step="any" min="0" max="30" value="2.0" required></div>
<div><label for="corriente_a">Corriente · A</label><input id="corriente_a" name="corriente_a" type="number" step="any" min="0" max="60" value="6.5" required></div>
</div><button type="submit">ANALIZAR CON IA →</button></form><div id="result" aria-live="polite">El diagnóstico exploratorio aparecerá aquí.</div></div>
<div class="card"><span class="tag">PANEL DEL MODELO</span><h2>MotorGuard / estadísticas</h2>
<div class="statgrid"><div class="stat"><strong>{{ total }}</strong><small>registros sintéticos</small></div><div class="stat"><strong>{{ flagged }}</strong><small>registros señalados</small></div><div class="stat"><strong>4</strong><small>variables estudiadas</small></div><div class="stat"><strong>200</strong><small>árboles del modelo</small></div></div>
<h3>¿Qué significa el resultado?</h3><p class="muted">La etiqueta «poco habitual» señala un patrón estadísticamente aislado. No es una confirmación de avería ni un diagnóstico de seguridad.</p>
<h3>Autora</h3><p><b>Lorena Abigail Coca Ccorpa</b><br><span class="muted">Proyecto académico · SENATI</span></p></div></section></main>
<footer>MotorGuard SENATI · Lorena Abigail Coca Ccorpa · Datos simulados para enseñanza · 2026</footer>
<script>
document.getElementById('motor').addEventListener('submit',async e=>{
e.preventDefault();const box=document.getElementById('result');box.textContent='Analizando registros…';
const payload=Object.fromEntries(new FormData(e.target).entries());
try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
const d=await r.json();if(!r.ok)throw Error(d.error||'No se pudo analizar');
box.innerHTML='<h3 class="'+(d.anomalia?'bad':'good')+'">'+(d.anomalia?'⚠ Patrón poco habitual':'✓ Patrón habitual')+'</h3><p>Índice del modelo: <b>'+d.puntaje.toFixed(4)+'</b></p><small>'+d.mensaje+'</small>';
}catch(err){box.textContent='Error: '+err.message;}
});
</script></body></html>"""

@app.get("/")
def home():
    return render_template_string(HTML,total=len(data),flagged=flagged)

@app.get("/health")
def health():
    return jsonify(status="ok",registros=len(data),modelo="IsolationForest")

@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload,dict): return jsonify(error="Envía un objeto JSON."),400
    try:
        values=[float(payload[name]) for name in FEATURES]
        if any(not math.isfinite(v) or v<lo or v>hi for v,(lo,hi) in zip(values,LIMITS)):
            raise ValueError("Variables fuera del rango admitido.")
    except (KeyError,ValueError,TypeError) as exc:
        return jsonify(error="Datos inválidos: "+str(exc)),400
    row=pd.DataFrame([values],columns=FEATURES)
    unusual=bool(model.predict(row)[0]==-1)
    score=float(model.decision_function(row)[0])
    return jsonify(anomalia=unusual,puntaje=round(score,5),
                   mensaje=("Revisar las condiciones del motor con un técnico." if unusual else "Medición compatible con patrones habituales del conjunto simulado."),
                   autor="Lorena Abigail Coca Ccorpa")
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)
