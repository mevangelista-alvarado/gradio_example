import os
import base64
import json
import uuid
import random
from copy import deepcopy
from datetime import datetime
import time

# Importaciones requeridas
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px

# 🧠 IA y Procesamiento
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

# 🔒 Cifrado
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

# 🔥 PyTorch (RNN)
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------------
# 🔹 Base de datos (Memoria/JSON)
# --------------------------------------------------------

JSON_PATH = "aspirantes.json"

def save_json(data):
    if isinstance(data, dict) and "aspirantes" in data:
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            data["ultima_actualizacion"] = datetime.now().isoformat()
            json.dump(data, f, ensure_ascii=False, indent=2)

def load_json():
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"aspirantes": []}
    else:
        return {"aspirantes": []}

class InMemoryCollection:
    def __init__(self):
        data = load_json()
        self._docs = data.get("aspirantes", [])
        for doc in self._docs:
            if '_id' not in doc: doc['_id'] = str(uuid.uuid4())
        print(f"✅ Sistema inicializado. {len(self._docs)} candidatos cargados.")

    def insert_one(self, doc):
        doc["_id"] = str(uuid.uuid4())
        self._docs.append(deepcopy(doc))
        save_json({"aspirantes": self._docs})

    def find(self, q=None):
        if q and "_id" in q:
             return [d for d in self._docs if d.get('_id') == q['_id']]
        return list(self._docs)

    def find_one(self, q):
        for doc in self._docs:
            if all(doc.get(k) == v for k, v in q.items()):
                return deepcopy(doc)
        return None

col_applicants = InMemoryCollection()

# --------------------------------------------------------
# 🔒 Cifrado RSA
# --------------------------------------------------------
KEY_FOLDER = './keys'
os.makedirs(KEY_FOLDER, exist_ok=True)
PUB_PATH, PRIV_PATH = f"{KEY_FOLDER}/public.pem", f"{KEY_FOLDER}/private.pem"

if not os.path.exists(PRIV_PATH):
    key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    with open(PRIV_PATH, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    with open(PUB_PATH, "wb") as f:
        f.write(key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))

public_key = serialization.load_pem_public_key(open(PUB_PATH, "rb").read())
private_key = serialization.load_pem_private_key(open(PRIV_PATH, "rb").read(), password=None)

def rsa_encrypt(txt):
    if not txt: return ""
    try:
        return base64.b64encode(public_key.encrypt(
            str(txt).encode('utf-8'),
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )).decode('utf-8')
    except Exception: return str(txt)

def rsa_decrypt(txt):
    if not txt: return ""
    try:
        raw = base64.b64decode(txt, validate=True)
        return private_key.decrypt(
            raw,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        ).decode('utf-8')
    except Exception: return "[Cifrado]"

# --------------------------------------------------------
# 🔥 RNN MODULE
# --------------------------------------------------------
class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModule, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

# --------------------------------------------------------
# 🧠 Controlador Principal
# --------------------------------------------------------
class SistemaReclutamiento:
    def __init__(self):
        self.vectorizer = None
        self.modelo_rnn = None
        self.scaler_rnn = StandardScaler()
        self.rnn_input_size = 0

    # --- GENERADOR DE DATOS SINTÉTICOS ---
    def simular_datos_masivos(self, n=50):
        nombres = ["Ana", "Carlos", "Elena", "David", "Sofia", "Jorge", "Lucia", "Miguel", "Maria", "Pedro", "Laura", "Pablo"]
        apellidos = ["García", "López", "Martínez", "Rodriguez", "Hernandez", "Smith", "Johnson", "Williams", "Brown", "Jones"]
        puestos = ["Ingeniero", "Ventas", "Marketing"]

        skills_dev = ["Python", "C++", "Java", "AWS", "Docker", "React", "Node", "SQL"]
        skills_sales = ["Ventas", "CRM", "Negociación", "Clientes", "Liderazgo"]
        skills_mkt = ["Marketing", "SEO", "Redes", "Diseño", "Copywriting"]
        skills_basic = ["Word", "Excel", "PowerPoint", "Email", "Internet"]

        count = 0
        for _ in range(n):
            puesto = random.choice(puestos)
            es_buen_candidato = random.random() > 0.4

            if es_buen_candidato:
                exp = random.randint(4, 12)
                edad = random.randint(26, 50)
                if puesto == "Ingeniero": lista_skills = random.sample(skills_dev, 3)
                elif puesto == "Ventas": lista_skills = random.sample(skills_sales, 3)
                else: lista_skills = random.sample(skills_mkt, 3)
                habilidades = ", ".join(lista_skills + ["Proactivo", "Inglés Avanzado"])
            else:
                exp = random.randint(0, 3)
                edad = random.randint(20, 30)
                habilidades = ", ".join(random.sample(skills_basic, 3))

            nombre_full = f"{random.choice(nombres)} {random.choice(apellidos)}"
            score_real = self.evaluar(habilidades, puesto)
            prob_rec = 0.85 if score_real > 70 and exp > 3 else random.uniform(0.1, 0.5)

            data = {
                "nombre_enc": rsa_encrypt(nombre_full), "edad": edad,
                "pais": random.choice(["México", "España", "Colombia", "Argentina", "Chile"]),
                "genero": random.choice(["Masculino", "Femenino"]),
                "puesto": puesto, "experiencia": exp,
                "habilidades_enc": rsa_encrypt(habilidades), "score": score_real,
                "recomendacion_prob": prob_rec, "estado": "Simulado", "fecha": datetime.now().isoformat()
            }
            col_applicants.insert_one(data)
            count += 1
        return f"✅ {count} candidatos simulados generados."

    def _preparar_datos_completos(self):
        raw_data = col_applicants.find()
        if not raw_data: return pd.DataFrame(), pd.DataFrame()
        df = pd.DataFrame(raw_data)

        df['edad'] = pd.to_numeric(df['edad'], errors='coerce').fillna(0)
        df['experiencia'] = pd.to_numeric(df['experiencia'], errors='coerce').fillna(0)
        df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
        df['habilidades_descifradas'] = df['habilidades_enc'].apply(rsa_decrypt)
        df_encoded = pd.get_dummies(df, columns=['pais', 'genero', 'puesto'], drop_first=False)

        if not self.vectorizer:
            self.vectorizer = CountVectorizer(max_features=10, stop_words='spanish')
            try: X_hab = self.vectorizer.fit_transform(df_encoded['habilidades_descifradas']).toarray()
            except: X_hab = np.zeros((len(df), 10))
        else:
            try: X_hab = self.vectorizer.transform(df_encoded['habilidades_descifradas']).toarray()
            except: X_hab = np.zeros((len(df), 10))

        X_hab_df = pd.DataFrame(X_hab, columns=[f"skill_{i}" for i in range(X_hab.shape[1])])
        features = ['edad', 'experiencia', 'score'] + [col for col in df_encoded.columns if col.startswith(('pais_', 'genero_', 'puesto_'))]
        X_final = pd.concat([df_encoded[features].reset_index(drop=True), X_hab_df], axis=1).fillna(0)
        return df, X_final

    def evaluar(self, texto, puesto):
        keywords = {
            'ingeniero': ['python','c++','tecnología','desarrollo','proyectos','aws','docker','sql'],
            'ventas': ['clientes','ventas','negociación','estrategia','crm'],
            'marketing': ['publicidad','redes','estrategia','creativo','seo']
        }
        rel = keywords.get(puesto.lower(), [])
        if not rel: return 40.0
        count = sum(1 for w in texto.lower().replace(',','').split() if w in rel)
        return round(min(40 + 60 * (count / (len(rel) or 1) * 2), 100.0), 2)

    # --- RNN TRAINING CON MATRIZ DE CONFUSIÓN ---
    def entrenar_rnn_desempeno(self):
        df, X_final = self._preparar_datos_completos()
        if X_final.empty or len(df) < 5:
            return "⚠️ Faltan datos. Simula candidatos primero.", px.scatter(title="Sin datos")

        X_numpy = self.scaler_rnn.fit_transform(X_final.to_numpy())
        inputs = torch.tensor(X_numpy, dtype=torch.float32).unsqueeze(1)
        # Regla de éxito para entrenamiento: Exp >= 4 y Score > 65
        y_simulado = [1 if (r['experiencia'] >= 4 and r['score'] > 65) else 0 for _, r in df.iterrows()]
        targets = torch.tensor(y_simulado, dtype=torch.float32).unsqueeze(1)

        self.rnn_input_size = X_numpy.shape[1]
        self.modelo_rnn = RNNModule(input_size=self.rnn_input_size, hidden_size=32, output_size=1)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.modelo_rnn.parameters(), lr=0.01)

        self.modelo_rnn.train()
        loss_vals = []

        # 1. Entrenamiento
        for _ in range(200):
            optimizer.zero_grad()
            loss = criterion(self.modelo_rnn(inputs), targets)
            loss.backward()
            optimizer.step()
            loss_vals.append(loss.item())

        # 2. Generar Matriz de Confusión sobre los datos de entrenamiento
        self.modelo_rnn.eval()
        with torch.no_grad():
            outputs = self.modelo_rnn(inputs)
            predicted = (outputs > 0.5).float()

            # Convertir a numpy para sklearn y plotly
            y_true = targets.numpy().flatten()
            y_pred = predicted.numpy().flatten()

            cm = confusion_matrix(y_true, y_pred)

        # 3. Crear Gráfico de Matriz
        fig = px.imshow(cm,
                        text_auto=True,
                        color_continuous_scale='Blues',
                        labels=dict(x="Predicción IA", y="Realidad", color="Cantidad"),
                        x=['Rechazar', 'Contratar'],
                        y=['Rechazar', 'Contratar'],
                        title="Matriz de Confusión del Modelo"
                       )
        fig.update_layout(width=400, height=300)

        log_txt = f"🚀 FIN ENTRENAMIENTO\n-----------------\n• Datos: {len(df)}\n• Epochs: 200\n• Loss Final: {loss_vals[-1]:.4f}\n\nLa Matriz muestra qué tan bien aprendió la IA sus reglas."

        return log_txt, fig

    # --- PREDICCIÓN ---
    def predecir_con_rnn(self, id_candidato):
        if not self.modelo_rnn: return "⚠️ Primero debes presionar 'Iniciar Entrenamiento Neuronal'."
        if not id_candidato: return "⚠️ Selecciona un ID de la tabla."

        doc = col_applicants.find_one({"_id": id_candidato.strip()})
        if not doc: return "❌ ID no encontrado."

        df, X_final = self._preparar_datos_completos()
        idx = df.index[df['_id'] == id_candidato.strip()].tolist()
        if not idx: return "❌ Error de índice."

        vector = X_final.iloc[idx[0]].to_numpy().reshape(1, -1)
        prob = self.modelo_rnn(torch.tensor(self.scaler_rnn.transform(vector), dtype=torch.float32).unsqueeze(1)).item()

        emoji = "🌟" if prob > 0.75 else ("⚠️" if prob < 0.4 else "🤔")
        veredicto = "CONTRATAR" if prob > 0.75 else ("DESCARTAR" if prob < 0.4 else "EVALUAR")

        return f"Candidato: {rsa_decrypt(doc.get('nombre_enc'))}\n---------------------------\n📊 Probabilidad de Éxito: {prob:.1%}\n{emoji} Veredicto IA: {veredicto}"

    def registrar(self, nombre, edad, pais, genero, puesto, experiencia, habilidades):
        if not nombre: return "⚠️ Falta nombre."
        score = self.evaluar(habilidades, puesto)
        data = {
            "nombre_enc": rsa_encrypt(nombre), "edad": edad, "pais": pais, "genero": genero,
            "puesto": puesto, "experiencia": experiencia,
            "habilidades_enc": rsa_encrypt(habilidades), "score": score,
            "recomendacion_prob": 0.5, "estado": "Manual", "fecha": datetime.now().isoformat()
        }
        col_applicants.insert_one(data)
        return f"✅ {nombre} registrado."

    def ver_candidatos(self):
        data = col_applicants.find()
        if not data: return pd.DataFrame()
        res = []
        for d in data:
            item = deepcopy(d)
            item['ID'] = item.get('_id')
            item['Nombre'] = rsa_decrypt(item.get('nombre_enc'))
            item['Habilidades'] = rsa_decrypt(item.get('habilidades_enc'))
            res.append(item)
        return pd.DataFrame(res)[['ID', 'Nombre', 'puesto', 'score', 'experiencia']]

    def graficar_estrella(self):
        data = col_applicants.find()
        if not data: return px.line_polar(title="Sin datos")
        df = pd.DataFrame(data).tail(5)
        df['Nombre'] = df['nombre_enc'].apply(rsa_decrypt)
        df['Experiencia'] = df['experiencia'].apply(lambda x: min(float(x)*10, 100))
        df['Aptitud'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
        df['Potencial'] = df.apply(lambda r: r['score'] if r['experiencia'] > 3 else r['score']*0.5, axis=1)
        df_melted = df[['Nombre','Experiencia','Aptitud','Potencial']].melt(id_vars=['Nombre'], var_name='Métrica', value_name='Valor')
        fig = px.line_polar(df_melted, r='Valor', theta='Métrica', color='Nombre', line_close=True, title="Radar: Últimos 5 Candidatos", range_r=[0,100])
        fig.update_traces(fill='toself')
        return fig

sistema = SistemaReclutamiento()

# --------------------------------------------------------
# 🎨 UI Gradio
# --------------------------------------------------------
def click_tabla(evt: gr.SelectData, df):
    try: return str(df.iloc[evt.index[0], 0])
    except: return ""

with gr.Blocks() as app:
    gr.Markdown("# 🤖 HumanX: Sistema Neural de Reclutamiento")
    df_state = gr.State()

    with gr.Tab("👤 Registro"):
        with gr.Row():
            nombre = gr.Textbox(label="Nombre")
            edad = gr.Number(label="Edad", value=25)
            pais = gr.Textbox(label="País", value="México")
            genero = gr.Dropdown(["Masculino","Femenino"], label="Género")
            puesto = gr.Dropdown(["Ingeniero","Ventas","Marketing"], label="Puesto")
            experiencia = gr.Number(label="Experiencia (Años)", value=2)
        habilidades = gr.Textbox(label="Habilidades", lines=2)
        btn_reg = gr.Button("Registrar")
        out_reg = gr.Textbox(label="Status")
        btn_reg.click(sistema.registrar, [nombre,edad,pais,genero,puesto,experiencia,habilidades], out_reg)

    with gr.Tab("📊 Radar Chart"):
        btn_g = gr.Button("Ver Radar")
        plot = gr.Plot()
        btn_g.click(sistema.graficar_estrella, outputs=plot)

    with gr.Tab("🛠️ Admin & Brain"):
        with gr.Row():
            # COLUMNA IZQUIERDA: Datos
            with gr.Column(scale=1):
                gr.Markdown("### 🗂️ Gestión de Datos")
                with gr.Group():
                    with gr.Row():
                        btn_simular = gr.Button("🎲 Simular 50 Datos", variant="secondary")
                        btn_ref = gr.Button("🔄 Refrescar Tabla")
                    out_simulacion = gr.Markdown("")
                    table = gr.DataFrame(interactive=False, label="Base de Datos (Click para seleccionar)")

            # COLUMNA DERECHA: Cerebro IA
            with gr.Column(scale=1):
                # SECCIÓN 1: ENTRENAMIENTO + MATRIZ
                with gr.Group():
                    gr.Markdown("## 🧠 Entrenamiento Neuronal (RNN)")

                    btn_train = gr.Button("🔥 INICIAR ENTRENAMIENTO", variant="primary", size="lg")

                    with gr.Row():
                        # Consola de Texto
                        out_train = gr.Textbox(
                            label="Log de Entrenamiento",
                            lines=8,
                            placeholder="Resultados aquí...",
                            show_copy_button=True
                        )
                        # 🔥 NUEVO: Matriz de Confusión
                        plot_cm = gr.Plot(label="Matriz de Confusión")

                gr.Markdown("---")

                # SECCIÓN 2: PREDICCIÓN
                with gr.Group():
                    gr.Markdown("## 🔮 Predicción de Futuro Desempeño")
                    with gr.Row():
                        id_in = gr.Textbox(label="ID Seleccionado", placeholder="Selecciona en la tabla...", scale=3)
                        btn_pred = gr.Button("ANALIZAR", variant="stop", scale=1)

                    out_pred = gr.Textbox(
                        label="Veredicto de la Inteligencia Artificial",
                        lines=3,
                        text_align="center"
                    )

        # Eventos
        btn_simular.click(sistema.simular_datos_masivos, outputs=out_simulacion)
        btn_ref.click(sistema.ver_candidatos, outputs=table)
        table.change(lambda x: x, inputs=[table], outputs=[df_state])

        # Evento actualizado: Ahora devuelve texto Y gráfico
        btn_train.click(sistema.entrenar_rnn_desempeno, outputs=[out_train, plot_cm])

        btn_pred.click(sistema.predecir_con_rnn, inputs=id_in, outputs=out_pred)
        table.select(click_tabla, inputs=[df_state], outputs=[id_in])

# Configuración para Render
if __name__ == "__main__":
    # Render usa la variable de entorno PORT
    port = int(os.environ.get("PORT", 7860))
    # Render requiere server_name="0.0.0.0" para aceptar conexiones externas
    app.launch(server_name="0.0.0.0", server_port=port, share=False)

