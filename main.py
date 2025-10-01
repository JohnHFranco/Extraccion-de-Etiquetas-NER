import gradio as gr
from transformers import pipeline
import os
import time
import logging

# --- 1. Configuración Inicial ---

# Configuración del logging para ver el estado y los errores en la consola de Docker
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Lee el ID del modelo desde la variable de entorno (definida en start.sh)
model_id = os.getenv("MODEL_ID", "mrm8488/bert-spanish-cased-finetuned-ner")

# Cargar el pipeline de "ner" (Named Entity Recognition) una sola vez al iniciar
logging.info(f"Cargando el modelo NER: {model_id}...")
ner_pipeline = pipeline(
    "ner",
    model=model_id,
    aggregation_strategy="simple" # Parámetro actualizado para agrupar entidades
)
logging.info(f"¡Modelo '{model_id}' cargado!")


# --- 2. Función de Ayuda para Segmentar Texto ---

def segment_text(text, tokenizer, max_tokens=500):
    """
    Divide el texto en fragmentos que no excedan max_tokens según el tokenizador.
    Devuelve una lista de strings (los fragmentos) y el conteo total de tokens.
    """
    # Convierte todo el texto a una secuencia de IDs de tokens
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    total_tokens = len(input_ids)
    
    segments = []
    start = 0
    
    # Recorre la secuencia de IDs y la corta en fragmentos
    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        segment_ids = input_ids[start:end]
        
        # Convierte los IDs del fragmento de vuelta a texto
        segment_text = tokenizer.decode(segment_ids, skip_special_tokens=True)
        segments.append(segment_text)
        
        start = end
        
    logging.info(f"Texto segmentado en {len(segments)} chunk(s) para un total de {total_tokens} tokens.")
    return segments, total_tokens


# --- 3. Función Principal de Procesamiento ---

def encontrar_entidades(texto):
    start_time = time.time()
    logging.info("Iniciando análisis de entidades...")

    if not texto:
        logging.warning("Se recibió una entrada de texto vacía.")
        return "Por favor, ingresa un texto para analizar."
    
    # Filtro inicial para textos excesivamente largos
    LIMITE_PALABRAS = 2000
    conteo_palabras = len(texto.split())
    if conteo_palabras > LIMITE_PALABRAS:
        logging.warning(f"El texto de entrada ({conteo_palabras} palabras) excede el límite de {LIMITE_PALABRAS} palabras.")
        return (
            f"⚠️ ADVERTENCIA: El texto ingresado es demasiado largo ({conteo_palabras} palabras).\n"
            f"Por favor, reduce el texto a menos de {LIMITE_PALABRAS} palabras."
        )

    entidades_totales = []
    
    try:
        # Segmenta el texto en fragmentos (chunks) usando el tokenizador del modelo
        tokenizador = ner_pipeline.tokenizer
        chunks_de_texto, tokens_entrada_total = segment_text(texto, tokenizador)
        
        # Procesa cada fragmento por separado
        for chunk in chunks_de_texto:
            entidades = ner_pipeline(chunk)
            if entidades:
                entidades_totales.extend(entidades)

        end_time = time.time()
        tiempo_respuesta = end_time - start_time

        # Calcula las métricas con los resultados totales
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
            # Lógica para eliminar duplicados y preservar el orden original del texto
            entidades_vistas = set()
            entidades_unicas_ordenadas = []
            
            for entidad in sorted(entidades_totales, key=lambda x: x['start']):
                identificador_entidad = (entidad['word'], entidad['entity_group'], entidad['start'], entidad['end'])
                if identificador_entidad not in entidades_vistas:
                    entidades_unicas_ordenadas.append(entidad)
                    entidades_vistas.add(identificador_entidad)
            
            for entidad in entidades_unicas_ordenadas:
                resultados += f"Texto: '{entidad['word']}'\n"
                resultados += f"Categoría: {entidad['entity_group']} (Confianza: {entidad['score']:.2%})\n\n"
        
        return resumen_metricas + resultados

    except Exception as e:
        logging.error(f"Ocurrió un error al procesar el texto: {e}", exc_info=True)
        return "Error: La aplicación encontró un problema inesperado al procesar el texto."


# --- 4. Creación y Lanzamiento de la Interfaz ---

iface = gr.Interface(
    fn=encontrar_entidades,
    inputs=gr.Textbox(lines=10, placeholder="Escribe o pega aquí el texto que quieres analizar..."),
    outputs=gr.Textbox(label="Entidades Encontradas"),
    title="🔎 Reconocimiento de Entidades Nombradas (NER)",
    description="Este modelo identifica personas, organizaciones, lugares y otras entidades en texto en español."
)

logging.info("Lanzando la interfaz de Gradio...")
iface.launch(server_name="0.0.0.0")