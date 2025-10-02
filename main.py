import gradio as gr
from transformers import pipeline
import os
import time
import logging
import pandas as pd

# --- 1. Configuración Inicial ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
model_id = os.getenv("MODEL_ID", "mrm8488/bert-spanish-cased-finetuned-ner")

logging.info(f"Cargando el modelo NER: {model_id}...")
ner_pipeline = pipeline("ner", model=model_id, aggregation_strategy="simple")
logging.info(f"¡Modelo '{model_id}' cargado!")


# --- 2. Función de Ayuda para Segmentar Texto ---

def segment_text(text, tokenizer, max_tokens=500):
    input_ids = tokenizer(text, return_tensors="pt")['input_ids'][0]
    total_tokens = len(input_ids)
    segments = []
    start = 0
    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        segment_ids = input_ids[start:end]
        segment_text = tokenizer.decode(segment_ids, skip_special_tokens=True)
        segments.append(segment_text)
        start = end
    logging.info(f"Texto segmentado en {len(segments)} chunk(s) para un total de {total_tokens} tokens.")
    return segments, total_tokens


# --- 3. Función Principal de Procesamiento ---

def encontrar_entidades(texto):
    start_time = time.time()
    logging.info("Iniciando análisis de entidades...")

    # Valores de salida por defecto
    metricas_vacias = "--- Métricas de Procesamiento ---\n" \
                      "⏱️ Tiempo de Respuesta: N/A\n" \
                      "..."
    lista_vacia = "_Sin resultados._"

    if not texto:
        logging.warning("Se recibió una entrada de texto vacía.")
        return metricas_vacias, lista_vacia

    entidades_totales = []
    try:
        tokenizador = ner_pipeline.tokenizer
        chunks_de_texto, tokens_entrada_total = segment_text(texto, tokenizador)
        
        for chunk in chunks_de_texto:
            entidades = ner_pipeline(chunk)
            if entidades:
                entidades_totales.extend(entidades)

        end_time = time.time()
        tiempo_respuesta = end_time - start_time
        
        if not entidades_totales:
            logging.info("Análisis completado. No se encontraron entidades.")
            return metricas_vacias, "_No se encontraron entidades en el texto._"

        # Agrupar entidades únicas
        entidades_unicas = {}
        for entidad in entidades_totales:
            key = (entidad['word'].strip().lower(), entidad['entity_group'])
            if key not in entidades_unicas or entidad['score'] > entidades_unicas[key]['score']:
                entidades_unicas[key] = entidad

        df_unicas = pd.DataFrame(list(entidades_unicas.values()))
        
        # --- NUEVO FORMATO DE SALIDA PARA LAS LISTAS ---
        texto_final_entidades = ""
        mapeo_titulos = {
            "PER": "Person",
            "ORG": "Organization",
            "LOC": "Location",
            "MISC": "Miscellaneous"
        }

        for categoria, titulo in mapeo_titulos.items():
            sub_df = df_unicas[df_unicas['entity_group'] == categoria]
            texto_final_entidades += f"### {titulo}\n" # Añade el título de la sección
            if not sub_df.empty:
                # Crea la lista con guiones y el formato solicitado
                lista_entidades = "\n".join(
                    sorted([f"- {row['word']} (confidence: {row['score']:.1%})" 
                            for _, row in sub_df.iterrows()])
                )
                texto_final_entidades += lista_entidades + "\n\n"
            else:
                texto_final_entidades += f"_No se encontraron entidades de tipo '{categoria}'._\n\n"

        # Calcular métricas
        tokens_salida = sum(len(tokenizador.tokenize(e['word'])) for e in entidades_totales)
        confianza_promedio = sum(e['score'] for e in entidades_totales) / len(entidades_totales)

        logging.info(f"Análisis completado en {tiempo_respuesta:.2f} segundos.")

        resumen_metricas = (
            f"--- Processing Metrics ---\n"
            f"⏱️ Response Time: {tiempo_respuesta:.2f} seconds\n"
            f"🎟️ Input Tokens: {tokens_entrada_total}\n"
            f"🏷️ Output Tokens (Entities): {tokens_salida}\n"
            f"🎯 Average Confidence: {confianza_promedio:.2%}\n"
            f"----------------\n\n"
        )
        
        return resumen_metricas, texto_final_entidades

    except Exception as e:
        logging.error(f"Ocurrió un error al procesar el texto: {e}", exc_info=True)
        error_msg = "Error: The application encountered an unexpected problem."
        return error_msg, lista_vacia


# --- 4. Creación y Lanzamiento de la Interfaz con Tema y Layout Corregidos ---

# Se usa un tema claro (Soft) para que el texto negro sea visible
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_text_color="#111827", # Color de texto principal en negro
)

# Función de ayuda para limpiar todas las salidas (CORREGIDA)
def clear_all_outputs():
    # Devuelve un valor vacío para cada componente de salida (3 en total)
    return "", "", ""

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Named Entity Recognition (NER) Extractor</h1>")
    gr.Markdown("<p style='text-align: center;'>This model identifies persons (PER), organizations (ORG), locations (LOC), and other miscellaneous entities (MISC) in Spanish text.</p>")

    # Layout principal de dos columnas
    with gr.Row(equal_height=False):
        # Columna Izquierda
        with gr.Column(scale=1, variant='panel'):
            input_text = gr.Textbox(lines=28, placeholder="Pega aquí el texto que quieres analizar...", label="Texto de Entrada")
            with gr.Row():
                clear_button = gr.Button("Limpiar")
                submit_button = gr.Button("Analizar Texto", variant="primary")
        
        # Columna Derecha
        with gr.Column(scale=1, variant='panel'):
            # El Markdown único para todas las listas de entidades
            detailed_list_output = gr.Markdown(label="Listas Detalladas de Entidades Únicas")

    # Bloque de Métricas separado en la parte inferior
    with gr.Row(variant='panel'):
        metrics_output = gr.Textbox(label="Métricas de Procesamiento", lines=6, interactive=False)

    # Lógica de los botones
    outputs_list = [metrics_output, detailed_list_output]
    submit_button.click(fn=encontrar_entidades, inputs=input_text, outputs=outputs_list)
    
    # CORRECCIÓN: La función para limpiar ahora apunta al número correcto de componentes
    clear_button.click(fn=clear_all_outputs, inputs=None, outputs=[input_text] + outputs_list)

logging.info("Lanzando la interfaz de Gradio...")
demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)))