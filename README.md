# 🤖 HumanX: Sistema Neural de Reclutamiento

Sistema de reclutamiento inteligente con IA basado en RNN (Red Neuronal Recurrente) usando Gradio.

## 🚀 Despliegue en Render

### Configuración en Render

1. **Crear un nuevo Web Service** en Render
2. **Configuración del servicio:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
   - **Environment:** Python 3
   - **Plan:** Free o Starter (según tus necesidades)

3. **Variables de entorno (opcionales):**
   - `PORT`: Render lo configura automáticamente, pero puedes dejarlo por defecto

### 📁 Estructura del Proyecto

```
gradio_example/
├── app.py              # Aplicación principal
├── requirements.txt    # Dependencias Python
├── README.md          # Este archivo
├── aspirantes.json    # Base de datos (se crea automáticamente)
└── keys/              # Claves RSA (se crean automáticamente)
    ├── public.pem
    └── private.pem
```

### 🔧 Características

- ✅ Registro de candidatos con cifrado RSA
- ✅ Simulación de datos masivos
- ✅ Entrenamiento de modelo RNN
- ✅ Matriz de confusión para evaluación
- ✅ Predicción de desempeño futuro
- ✅ Visualizaciones con Plotly (Radar Chart)

### 📝 Notas Importantes

- Los archivos `aspirantes.json` y `keys/` se crean automáticamente
- En Render, los archivos se persisten durante el ciclo de vida del servicio
- Para producción, considera usar una base de datos externa (PostgreSQL, MongoDB, etc.)

### 🛠️ Desarrollo Local

```bash
pip install -r requirements.txt
python app.py
```

La aplicación estará disponible en `http://localhost:7860`

