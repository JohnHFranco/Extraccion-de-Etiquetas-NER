import gradio as gr
from transformers import pipeline
import os
import time

# Lee el ID del modelo desde la variable de entorno definida en start.sh
model_id = os.getenv("MODEL_ID", "mrm8488/bert-spanish-cased-finetuned-ner")

# Cargar el pipeline de "ner" (Named Entity Recognition) una sola vez
print("Cargando el modelo NER...")
ner_pipeline = pipeline(
    "ner",
    model=model_id,
    aggregation_strategy="simple"
)
print(f"¡Modelo '{model_id}' cargado!")

# 1. Define una función que procese el texto de entrada
def encontrar_entidades(texto):
    # --- Métrica 1: Tiempo de Respuesta (Inicio) ---
    start_time = time.time()

    if not texto:
        return "Por favor, ingresa un texto para analizar."
    
    print("\nAnalizando el texto para encontrar entidades...")
    entidades = ner_pipeline(texto)
    
    # --- Métrica 1: Tiempo de Respuesta (Fin) ---
    end_time = time.time()
    tiempo_respuesta = end_time - start_time

    # --- Métrica 2: Tokens de Entrada y Salida ---
    tokenizador = ner_pipeline.tokenizer
    tokens_entrada = len(tokenizador.tokenize(texto))

    # Sumamos los tokens de solo las palabras que fueron identificadas como entidades
    tokens_salida = sum(len(tokenizador.tokenize(e['word'])) for e in entidades)

    # --- Métrica 3: Fiabilidad de Respuesta (Confianza Promedio) ---
    confianza_promedio = 0
    if entidades:
        # Calculamos el promedio de la confianza de todas las entidades encontradas
        confianza_promedio = sum(e['score'] for e in entidades) / len(entidades)
    
    # --- Formatear la Salida para Incluir las Métricas ---
    # Creamos una cabecera con el resumen de las métricas
    resumen_metricas = (
        f"--- Métricas ---\n"
        f"Tiempo de Respuesta: {tiempo_respuesta:.2f} segundos\n"
        f"Tokens de Entrada: {tokens_entrada}\n"
        f"Tokens de Salida (Entidades): {tokens_salida}\n"
        f"Confianza Promedio: {confianza_promedio:.2%}\n"
        f"----------------\n\n"
    )

    resultados = ""
    if not entidades:
        resultados = "No se encontraron entidades en el texto."
    else:
        for entidad in entidades:
            resultados += f"Texto: '{entidad['word']}'\n"
            resultados += f"Categoría: {entidad['entity_group']} (Confianza: {entidad['score']:.2%})\n\n"
    
    # Devolvemos el resumen de métricas junto con los resultados
    return resumen_metricas + resultados

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