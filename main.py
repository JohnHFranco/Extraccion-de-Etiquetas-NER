import gradio as gr
from transformers import pipeline
import os

# Lee el ID del modelo desde la variable de entorno definida en start.sh
model_id = os.getenv("MODEL_ID", "mrm8488/bert-spanish-cased-finetuned-ner")

# Cargar el pipeline de "ner" (Named Entity Recognition) una sola vez
print("Cargando el modelo NER...")
ner_pipeline = pipeline(
    "ner",
    model=model_id,
    grouped_entities=True
)
print(f"¡Modelo '{model_id}' cargado!")

# 1. Define una función que procese el texto de entrada
def encontrar_entidades(texto):
    """
    Esta función toma un texto como entrada, utiliza el pipeline de NER para
    encontrar entidades y devuelve los resultados formateados como un string.
    """
    if not texto:
        return "Por favor, ingresa un texto para analizar."
    
    print("\nAnalizando el texto para encontrar entidades...")
    entidades = ner_pipeline(texto)
    
    # Formatear la salida para que sea legible
    resultados = ""
    if not entidades:
        return "No se encontraron entidades en el texto."
        
    for entidad in entidades:
        resultados += f"Texto: '{entidad['word']}'\n"
        resultados += f"Categoría: {entidad['entity_group']} (Confianza: {entidad['score']:.2f})\n\n"
    
    return resultados

# 2. Crea la interfaz de Gradio
iface = gr.Interface(
    fn=encontrar_entidades,  # La función que se ejecutará
    inputs=gr.Textbox(lines=10, placeholder="Escribe o pega aquí el texto que quieres analizar..."),
    outputs=gr.Textbox(label="Entidades Encontradas"),
    title="🔎 Reconocimiento de Entidades Nombradas (NER)",
    description="Este modelo identifica personas, organizaciones, lugares y otras entidades en texto en español. Está basado en 'mrm8488/bert-spanish-cased-finetuned-ner'."
)

# 3. Lanza la interfaz
print("Lanzando la interfaz de Gradio...")
iface.launch(server_name="0.0.0.0")