import gradio as gr
from transformers import pipeline
import os
import time
import logging

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Lee el ID del modelo desde la variable de entorno
model_id = os.getenv("MODEL_ID", "mrm8488/bert-spanish-cased-finetuned-ner")

# Cargar el pipeline de "ner" una sola vez
logging.info("Cargando el modelo NER...")
ner_pipeline = pipeline(
    "ner",
    model=model_id,
    aggregation_strategy="simple"
)
logging.info(f"¡Modelo '{model_id}' cargado!")


# --- NUEVA FUNCIÓN DE SEGMENTACIÓN BASADA EN TOKENS ---
def segment_text(text, tokenizer, max_tokens=510):
    """
    Divide el texto en fragmentos que no excedan max_tokens según el tokenizador.
    Devuelve una lista de strings (los fragmentos) y el conteo total de tokens.
    """
    # 1. Convierte todo el texto a una secuencia de IDs de tokens
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    total_tokens = len(input_ids)
    
    segments = []
    start = 0
    
    # 2. Recorre la secuencia de IDs y córtala en fragmentos
    while start < total_tokens:
        # Define el final del fragmento, sin pasarse de la longitud total
        end = min(start + max_tokens, total_tokens)
        segment_ids = input_ids[start:end]
        
        # 3. Convierte los IDs del fragmento de vuelta a texto
        segment_text = tokenizer.decode(segment_ids, skip_special_tokens=True)
        segments.append(segment_text)
        
        # Avanza al siguiente fragmento
        start = end
        
    logging.info(f"Texto segmentado en {len(segments)} chunk(s) para un total de {total_tokens} tokens.")
    return segments, total_tokens


# --- FUNCIÓN PRINCIPAL MODIFICADA PARA USAR LA NUEVA SEGMENTACIÓN ---
def encontrar_entidades(texto):
    start_time = time.time()
    logging.info("Iniciando análisis de entidades...")

    if not texto:
        logging.warning("Se recibió una entrada de texto vacía.")
        return "Por favor, ingresa un texto para analizar."
    
    entidades_totales = []
    
    try:
        # 1. Usa la nueva función para segmentar el texto por tokens
        tokenizador = ner_pipeline.tokenizer
        chunks_de_texto, tokens_entrada_total = segment_text(texto, tokenizer)
        
        # 2. Procesa cada fragmento de texto por separado
        for chunk in chunks_de_texto:
            entidades = ner_pipeline(chunk)
            if entidades:
                entidades_totales.extend(entidades)

        end_time = time.time()
        tiempo_respuesta = end_time - start_time

        # 3. Calcula las métricas con los resultados totales
        tokens_salida = sum(len(tokenizador.tokenize(e['word'])) for e in entidades_totales)
        confianza_promedio = sum(e['score'] for e in entidades_totales) / len(entidades_totales) if entidades_totales else 0

        logging.info(f"Análisis completado en {tiempo_respuesta:.2f} segundos. Entidades encontradas: {len(entidades_totales)}")

        resumen_metricas = (
            f"--- Métricas ---\n"
            f"⏱️ Tiempo de Respuesta: {tiempo_respuesta:.2f} segundos\n"
            f"🎟️ Tokens de Entrada: {tokens_entrada_total}\n"
            f"🏷️ Tokens de Salida (Entidades): {tokens_salida}\n"
            f"🎯 Confianza Promedio: {confianza_promedio:.2%}\n"
            f"----------------\n\n"
        )

        resultados = ""
        if not entidades_totales:
            resultados = "No se encontraron entidades en el texto."
        else:
            # Eliminar duplicados si una entidad aparece en el borde de dos chunks
            entidades_unicas = [dict(t) for t in {tuple(d.items()) for d in entidades_totales}]
            for entidad in sorted(entidades_unicas, key=lambda x: x['start']):
                resultados += f"Texto: '{entidad['word']}'\n"
                resultados += f"Categoría: {entidad['entity_group']} (Confianza: {entidad['score']:.2%})\n\n"
        
        return resumen_metricas + resultados

    except Exception as e:
        logging.error(f"Ocurrió un error al procesar el texto: {e}", exc_info=True)
        return "Error: La aplicación encontró un problema inesperado al procesar el texto."


# 2. Crea la interfaz de Gradio (sin cambios aquí)
iface = gr.Interface(
    fn=encontrar_entidades,
    inputs=gr.Textbox(lines=10, placeholder="Escribe o pega aquí el texto que quieres analizar..."),
    outputs=gr.Textbox(label="Entidades Encontradas"),
    title="🔎 Reconocimiento de Entidades Nombradas (NER)",
    description="Este modelo identifica personas, organizaciones, lugares y otras entidades en texto en español."
)

# 3. Lanza la interfaz (sin cambios aquí)
logging.info("Lanzando la interfaz de Gradio...")
iface.launch(server_name="0.0.0.0")