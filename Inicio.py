import os
import streamlit as st
import base64
from openai import OpenAI
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

# -----------------------------
# Session state
# -----------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "emotion_result" not in st.session_state:
    st.session_state.emotion_result = ""

# -----------------------------
# Función convertir imagen
# -----------------------------
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# -----------------------------
# Configuración app
# -----------------------------
st.set_page_config(page_title="🎨 Trazos con Emoción")
st.title("🎨 Trazos con Emoción")

with st.sidebar:
    st.subheader("🧠 Acerca de")
    st.write(
        "Dibuja libremente y la inteligencia artificial "
        "interpretará la emoción transmitida por tus trazos."
    )

    st.divider()
    st.subheader("🎛️ Propiedades del Tablero")

    # Dimensiones
    st.subheader("Dimensiones")
    canvas_width = st.slider("Ancho", 300, 700, 500, 50)
    canvas_height = st.slider("Alto", 200, 600, 350, 50)

    # Herramienta
    drawing_mode = st.selectbox(
        "Herramienta de dibujo",
        (
            "freedraw",
            "line",
            "rect",
            "circle",
            "transform",
            "polygon",
            "point",
        ),
    )

    # Grosor
    stroke_width = st.slider(
        "Ancho del trazo", 1, 30, 5
    )

    # Colores
    stroke_color = st.color_picker(
        "Color de trazo", "#000000"
    )
    bg_color = st.color_picker(
        "Color de fondo", "#FFFFFF"
    )

# -----------------------------
# Canvas con controles
# -----------------------------
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=canvas_height,
    width=canvas_width,
    drawing_mode=drawing_mode,
    key=f"canvas_{canvas_width}_{canvas_height}_{drawing_mode}",
)

# -----------------------------
# API KEY
# -----------------------------
api_key = st.text_input("🔑 Ingresa tu API Key", type="password")

if api_key:
    client = OpenAI(api_key=api_key)

# -----------------------------
# Botón análisis
# -----------------------------
analyze_button = st.button("🧠 Interpretar emoción")

if canvas_result.image_data is not None and api_key and analyze_button:
    with st.spinner("Analizando emoción del dibujo..."):
        try:
            input_numpy_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(
                input_numpy_array.astype("uint8")
            ).convert("RGBA")

            input_image.save("emocion.png")

            base64_image = encode_image_to_base64("emocion.png")

            prompt_text = """
            Analiza este dibujo hecho a mano e interpreta la emoción que transmite.
            Responde en español con:
            - Emoción principal
            - Explicación basada en trazos, formas y colores
            - Sensación que genera
            - Un título artístico para la obra
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )

            result = response.choices[0].message.content
            st.session_state.emotion_result = result
            st.session_state.analysis_done = True

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

# -----------------------------
# Resultado
# -----------------------------
if st.session_state.analysis_done:
    st.divider()
    st.subheader("✨ Resultado emocional")
    st.write(st.session_state.emotion_result)

# -----------------------------
# Aviso
# -----------------------------
if not api_key:
    st.info("Ingresa tu API key para activar el análisis.")
