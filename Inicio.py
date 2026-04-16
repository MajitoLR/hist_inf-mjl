import os
import streamlit as st
import base64
import openai
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
# Función base64
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
        "Esta aplicación interpreta la emoción transmitida a través "
        "de tus trazos, formas y composición visual."
    )

st.subheader("Dibuja libremente en el panel y presiona analizar")

# -----------------------------
# Canvas
# -----------------------------
stroke_width = st.sidebar.slider(
    "Selecciona el ancho de línea", 1, 30, 5
)

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=350,
    width=500,
    drawing_mode="freedraw",
    key="canvas",
)

# -----------------------------
# API KEY
# -----------------------------
ke = st.text_input("🔑 Ingresa tu API Key", type="password")

if ke:
    client = OpenAI(api_key=ke)

# -----------------------------
# Botón analizar
# -----------------------------
analyze_button = st.button("🧠 Interpretar emoción")

if canvas_result.image_data is not None and ke and analyze_button:
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
            1. Emoción principal
            2. Qué elementos visuales te llevan a esa emoción
            3. Qué sensación transmite al espectador
            4. Un título creativo para la obra
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
# Mostrar resultado
# -----------------------------
if st.session_state.analysis_done:
    st.divider()
    st.subheader("✨ Interpretación emocional")
    st.write(st.session_state.emotion_result)

# -----------------------------
# Warning
# -----------------------------
if not ke:
    st.warning("Por favor ingresa tu API key.")
