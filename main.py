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

    metricas_vacias = "--- Processing Metrics ---\n" \
                      "⏱️ Response Time: N/A\n" \
                      "🔠 Word Count: N/A\n" \
                      "🎟️ Input Tokens: N/A\n" \
                      "🏷️ Output Tokens (Entities): N/A\n" \
                      "🎯 Average Confidence: N/A\n" \
                      "----------------\n\n"
    lista_vacia = "_Sin resultados._"

    if not texto:
        logging.warning("Se recibió una entrada de texto vacía.")
        return metricas_vacias, lista_vacia

    conteo_palabras = len(texto.split())
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
            return metricas_vacias, "_No entities found in the text._"

        entidades_unicas = {}
        for entidad in entidades_totales:
            key = (entidad['word'].strip().lower(), entidad['entity_group'])
            if key not in entidades_unicas or entidad['score'] > entidades_unicas[key]['score']:
                entidades_unicas[key] = entidad

        df_unicas = pd.DataFrame(list(entidades_unicas.values()))
        
        texto_final_entidades = ""
        mapeo_titulos = {"PER": "Person", "ORG": "Organization", "LOC": "Location", "MISC": "Miscellaneous"}

        for categoria, titulo in mapeo_titulos.items():
            sub_df = df_unicas[df_unicas['entity_group'] == categoria]
            texto_final_entidades += f"### {titulo}\n"
            if not sub_df.empty:
                lista_entidades = "\n".join(
                    sorted([f"- {row['word']} (confidence: {row['score']:.1%})" 
                            for _, row in sub_df.iterrows()])
                )
                texto_final_entidades += lista_entidades + "\n\n"
            else:
                texto_final_entidades += f"_No entities of type '{categoria}' found._\n\n"

        tokens_salida = sum(len(tokenizador.tokenize(e['word'])) for e in entidades_totales)
        confianza_promedio = sum(e['score'] for e in entidades_totales) / len(entidades_totales)
        logging.info(f"Análisis completado en {tiempo_respuesta:.2f} segundos.")

        resumen_metricas = (
            f"--- Processing Metrics ---\n"
            f"⏱️ Response Time: {tiempo_respuesta:.2f} seconds\n"
            f"🔠 Word Count: {conteo_palabras}\n"
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

# Se mantiene el tema oscuro para toda la aplicación
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace", "sans-serif"],
).set(
    body_background_fill="#111827",
    body_text_color="#e5e7eb",
    button_primary_background_fill="#4f46e5",
    button_primary_text_color="#ffffff",
    background_fill_primary="#1f2937",
    block_background_fill="#1f2937",
    block_border_width="1px",
    block_shadow="*shadow_md",
    block_label_background_fill="#111827",
    block_label_text_color="#ffffff",
    input_background_fill="#374151",
)

# --- CORRECCIÓN CON CSS ---
# CSS para el fondo y para aplicar el estilo específico al cuadro de salida
css = """
body {
    background-image: radial-gradient(circle at top, #1e3a8a 10%, #111827);
    background-attachment: fixed;
}
#title { text-align: center; display: block; }
#subtitle { text-align: center; display: block; color: #9ca3af; margin-bottom: 20px; }

/* Estilos específicos para el bloque de resultados */
#entity_output {
    color: #000000 !important; /* Letra negra */
    background-color: #f9fafb !important; /* Fondo blanco */
    padding: 1rem;
    border-radius: 8px;
}
#entity_output h3 { /* Estilo para los títulos (Person, Organization, etc.) */
    color: #1e3a8a !important; /* Títulos en azul oscuro */
}
"""

with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown("<h1 style='color: #e5e7eb;'>Named Entity Recognition (NER) Extractor</h1>", elem_id="title")
    gr.Markdown("<p>This model identifies persons (PER), organizations (ORG), locations (LOC), and other miscellaneous entities (MISC) in Spanish text.</p>", elem_id="subtitle")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, variant='panel'):
            input_text = gr.Textbox(lines=28, placeholder="Paste the text you want to analyze here...", label="Input Text")
            submit_button = gr.Button("Analyze Text", variant="primary")
        
        with gr.Column(scale=1, variant='panel'):
            # Se añade un ID a este componente para que el CSS pueda seleccionarlo
            detailed_list_output = gr.Markdown(label="Detailed Entity Lists", elem_id="entity_output")

    with gr.Row(variant='panel'):
        metrics_output = gr.Textbox(label="Processing Metrics", lines=7, interactive=False)

    outputs_list = [metrics_output, detailed_list_output]
    submit_button.click(fn=encontrar_entidades, inputs=input_text, outputs=outputs_list)

logging.info("Lanzando la interfaz de Gradio...")
demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)))