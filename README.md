# CarBrain

CarBrain es una aplicación web hecha con **Streamlit** para ayudar a evaluar autos seminuevos.  
Analiza el historial de quejas por marca, modelo y año, calcula un nivel de riesgo y usa **OpenAI** para explicar la recomendación en lenguaje claro.

## Qué hace

- Carga y limpia la base `df_final.csv`.
- Calcula métricas por vehículo, como:
  - total de reportes
  - choques
  - incendios
  - casos con lesionados
  - severidad general
- Genera un ranking de marcas.
- Agrupa vehículos por perfiles de riesgo con `KMeans`.
- Muestra gráficos interactivos con `Plotly`.
- Permite chatear con un asistente que responde solo sobre el vehículo seleccionado.

## Tecnologías usadas

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Plotly
- OpenAI API
- python-dotenv

## Estructura principal

- `app.py`: interfaz principal de Streamlit.
- `carbrain_data.py`: carga, limpieza, métricas, clusters y contexto para el chat.
- `df_final.csv`: base de datos usada por la app.
- `requirements.txt`: dependencias del proyecto.
- `.streamlit/secrets.toml.example`: ejemplo de configuración para secretos.

## Requisitos

- Python 3.10 o superior
- Cuenta y API key de OpenAI

## Instalación

1. Clona el repositorio.
2. Instala las dependencias:

```powershell
pip install -r requirements.txt
```

## Configuración

Puedes configurar las credenciales de dos formas:

### Opción 1: archivo `.env`

Crea un archivo `.env` en la raíz del proyecto con este contenido:

```ini
OPENAI_API_KEY=tu_api_key
OPENAI_MODEL=gpt-5
```

### Opción 2: Streamlit secrets

Si vas a desplegar en Streamlit Community Cloud, usa un archivo `secrets.toml` con este formato:

```toml
OPENAI_API_KEY = "tu_api_key"
OPENAI_MODEL = "gpt-5"
```

Puedes partir del ejemplo ubicado en:

- [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)

## Cómo ejecutar la app

```powershell
streamlit run app.py
```

## Uso

1. Selecciona una marca, modelo y año en la barra lateral.
2. Revisa el resumen del vehículo.
3. Compara contra otros vehículos desde la pestaña **Comparador**.
4. Haz preguntas en el chat para recibir una explicación en lenguaje sencillo.

## Qué muestra la app

- Veredicto general: `Recomendado`, `Con precaución` o `No recomendado`.
- Nivel de atención del vehículo.
- Comparación contra autos similares y contra el promedio de su marca.
- Gráficas de perfil general y distribución de tipos de falla.
- Lista de revisiones sugeridas antes de comprar.

## Despliegue en Streamlit Community Cloud

1. Sube el proyecto a GitHub.
2. Crea una nueva app en Streamlit Community Cloud.
3. Apunta el archivo principal a `app.py`.
4. Configura las variables `OPENAI_API_KEY` y `OPENAI_MODEL` en los secretos de Streamlit.
5. Verifica que `df_final.csv` esté incluido en el repositorio o disponible en la ruta esperada.

## Notas

- La app usa el contexto calculado localmente y la API de OpenAI para redactar respuestas.
- Si no encuentra `OPENAI_API_KEY` o `OPENAI_MODEL`, mostrará un mensaje de error en pantalla.
- El archivo `.gitignore` ya excluye `.env`, `__pycache__` y `.streamlit/secrets.toml`.

