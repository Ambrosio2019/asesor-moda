
# FUNCIONA perfecto LOS DOS PRIMEROS FLUJOS, FALTA EL TERCERO pero en teoria funcionaria 
# Al combinar ambos flujos, obtendrás recomendaciones más precisas y adaptadas a ti. El modelo tendrá en cuenta cómo las prendas pueden favorecer tus  
#proporciones y cómo combinarlas adecuadamente para la ocasión y el clima especificados

import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import tempfile
import requests
import io
import base64
import requests
from PIL import Image
import matplotlib.pyplot as plt
api_key = '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'
import os
import openai
os.environ['OPENAI_API_KEY'] = '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'
OPENAI_API_KEY='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'
openai.api_key = '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'
from openai import OpenAI
client = OpenAI()
import io  # Asegúrate de importar io
# Configuración de MediaPipe

import streamlit as st
import io
import base64
import requests
from PIL import Image
import numpy as np
import mediapipe as mp

# Configuración de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# OpenAI API Key
api_key = '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'

def encode_image(image):
    """Convierte la imagen a base64."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')



# Función para llamar a la API de OpenAI GPT-4 Vision
def call_gpt4_vision(image):
    base64_image = encode_image(image)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe exclusivamente las prendas de vestir que se observan en esta imagen."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    # Manejar la respuesta para evitar errores
    try:
        return response.json()['choices'][0]['message']['content']
    except KeyError:
        st.error("Error en la respuesta de la API. Respuesta completa:")
        st.json(response.json())
        return None

# Función para generar recomendaciones de outfits con análisis de postura
def generate_outfit_advisor_with_posture(images, occasion, weather, posture_data=None):
    prendas_descriptions = []

    # Procesar imágenes de ropa, si existen
    if images:
        for uploaded_file in images[:4]:  # Limitar a un máximo de 4 imágenes
            description = call_gpt4_vision(uploaded_file)
            if description:
                prendas_descriptions.append(description)

    # Crear el prompt adaptativo
    if prendas_descriptions and posture_data:
        # Caso: Con imágenes de ropa y datos de postura
        prendas_text = ', '.join(prendas_descriptions)
        prompt = f"""
        El usuario ha subido las siguientes prendas: {prendas_text}.
        La ocasión es una {occasion} y el clima es {weather}.
        Las proporciones corporales detectadas son:
        - Altura del torso: {posture_data.get('altura_torso', 'desconocido')}
        - Longitud de brazos: {posture_data.get('longitud_brazos', 'desconocido')}
        - Longitud de piernas: {posture_data.get('longitud_piernas', 'desconocido')}
        
        Proporciona recomendaciones claras sobre cómo combinar estas prendas de manera que favorezcan las proporciones corporales detectadas. 
        Ofrece dos opciones de combinación y selecciona la mejor para la ocasión. 
        Formatea la respuesta en Markdown.
        """
    elif not prendas_descriptions and posture_data:
        # Caso: Solo datos de postura
        prompt = f"""
        El usuario no ha subido imágenes de prendas específicas.
        La ocasión es una {occasion} y el clima es {weather}.
        Las proporciones corporales detectadas son:
        - Altura del torso: {posture_data.get('altura_torso', 'desconocido')}
        - Longitud de brazos: {posture_data.get('longitud_brazos', 'desconocido')}
        - Longitud de piernas: {posture_data.get('longitud_piernas', 'desconocido')}
        
        Basándote únicamente en las proporciones corporales, genera recomendaciones generales de ropa que favorezcan esta complexión. 
        Ofrece dos opciones de combinación y selecciona la mejor para la ocasión.
        Formatea la respuesta en Markdown.
        """
    elif prendas_descriptions and not posture_data:
        # Caso: Solo imágenes de ropa
        prendas_text = ', '.join(prendas_descriptions)
        prompt = f"""
        El usuario ha subido las siguientes prendas: {prendas_text}.
        La ocasión es una {occasion} y el clima es {weather}.
        
        Proporciona recomendaciones claras sobre cómo combinar estas prendas. Debes ofrecer dos opciones de combinación, 
        y al final seleccionar la mejor opción para la ocasión. Solo céntrate en las prendas mostradas y no inventes ni sugieras accesorios, calzado u otras prendas que no estén presentes en las imágenes.
        Formatea la respuesta en Markdown.
        """
    else:
        # Caso: No hay datos suficientes
        return "Por favor, sube imágenes de prendas o una imagen de postura para generar recomendaciones."

    # Llamada a la API de OpenAI
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "Eres un asesor de moda experto en combinaciones de ropa y proporciones corporales."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Manejar la respuesta para evitar errores
    try:
        return response.json()['choices'][0]['message']['content']
    except KeyError:
        st.error("Error en la respuesta de la API para el asesoramiento.")
        st.json(response.json())
        return None

def extract_posture_data(image):
    """Procesa una imagen con MediaPipe Pose y extrae proporciones corporales."""
    image_np = np.array(image)
    posture_data = {}

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        annotated_image = image_np.copy()
        if results.pose_landmarks:
            # Dibujar la pose en la imagen
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
            )
            # Extraer proporciones básicas
            landmarks = results.pose_landmarks.landmark
            posture_data = {
                "altura_torso": round(abs(landmarks[11].y - landmarks[23].y), 2),
                "longitud_brazos": round(abs(landmarks[15].y - landmarks[11].y), 2),
                "longitud_piernas": round(abs(landmarks[23].y - landmarks[27].y), 2)
            }
    return posture_data, annotated_image

# Función para redimensionar las imágenes
def resize_image(uploaded_file, max_size=(512, 512)):
    img = Image.open(uploaded_file)
    img.thumbnail(max_size)
    return img

# Función para cargar y redimensionar imágenes desde Streamlit
def process_uploaded_images(uploaded_files):
    resized_images = []
    for uploaded_file in uploaded_files:
        resized_img = resize_image(uploaded_file)
        resized_images.append(resized_img)
    return resized_images

# Interfaz de usuario principal
st.title("Recomendador de Outfits con IA y Análisis de Postura")

# Subir imagen de postura
st.header("Análisis de Postura")
uploaded_posture_file = st.file_uploader("Sube una imagen clara de tu cuerpo completo (de pie) para analizar tus proporciones:", type=["jpg", "jpeg", "png"])

posture_data = None
if uploaded_posture_file:
    image = Image.open(uploaded_posture_file)
    st.image(image, caption="Imagen de postura subida", use_column_width=True)

    posture_data, annotated_image = extract_posture_data(image)
    if posture_data:
        st.subheader("Datos extraídos de la postura:")
        st.write(posture_data)
        st.image(annotated_image, caption="Detección de postura", use_column_width=True)
    else:
        st.error("No se detectaron proporciones corporales en la imagen subida.")

# Subir imágenes de ropa
st.header("Imágenes de Prendas")
uploaded_clothes_files = st.file_uploader("Sube imágenes de tus prendas:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

resized_images = []
if uploaded_clothes_files:
    st.subheader("Imágenes redimensionadas:")
    resized_images = process_uploaded_images(uploaded_clothes_files)

    # Mostrar las imágenes redimensionadas
    for resized_img in resized_images:
        st.image(resized_img, caption="Imagen redimensionada", use_column_width=True)

# Seleccionar ocasión y clima
st.header("Configuración del Outfit")
occasion = st.text_input("Describe la ocasión (ej. comida informal con los suegros, trabajo, etc.)", "comida informal con los suegros")
weather = st.text_input("Describe el clima (ej. calor, fresco, frío, etc.)", "calor")

# Botón para generar recomendaciones
if st.button("Generar recomendaciones"):
    st.subheader("Recomendaciones de Outfits")
    recommendations = generate_outfit_advisor_with_posture(resized_images, occasion, weather, posture_data)
    st.markdown(recommendations)