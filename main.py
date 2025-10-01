import gradio as gr
from transformers import pipeline
import os
import time
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. Configuración Inicial ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
model_id = os.getenv("MODEL_ID", "mrm8488/bert-spanish-cased-finetuned-ner")

logging.info(f"Cargando el modelo NER: {model_id}...")
ner_pipeline = pipeline("ner", model=model_id, aggregation_strategy="simple")
logging.info(f"¡Modelo '{model_id}' cargado!")


# --- 2. Función de Ayuda para Segmentar Texto ---

def segment_text(text, tokenizer, max_tokens=500):
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
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
    grafico_vacio = go.Figure().add_annotation(text="No data for chart.", showarrow=False)
    metricas_vacias = "--- Processing Metrics ---\n" \
                      "⏱️ Response Time: N/A\n" \
                      "🎟️ Input Tokens: N/A\n" \
                      "🏷️ Output Tokens (Entities): N/A\n" \
                      "🎯 Average Confidence: N/A\n" \
                      "----------------\n\n"
    salidas_vacias = ("No Persons found.", "No Organizations found.", "No Locations found.", "No Miscellaneous found.")

    if not texto:
        logging.warning("Se recibió una entrada de texto vacía.")
        return metricas_vacias, grafico_vacio, *salidas_vacias

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
            return metricas_vacias, grafico_vacio, *salidas_vacias

        # --- LÓGICA MEJORADA DE GRÁFICO Y AGRUPACIÓN ---
        
        # 1. Crear un DataFrame con TODAS las entidades encontradas
        df_todas_entidades = pd.DataFrame(entidades_totales)
        
        # 2. CORRECCIÓN DEL GRÁFICO: Agrupar por categoría y calcular la confianza PROMEDIO
        avg_confidence = df_todas_entidades.groupby('entity_group')['score'].mean().reset_index()
        avg_confidence.columns = ['Category', 'Average Confidence']
        
        # Crear un gráfico de BARRAS, que es mejor para comparar promedios
        grafico_salida = px.bar(
            avg_confidence,
            x='Category',
            y='Average Confidence',
            title='Average Confidence per Entity Category',
            color='Category',
            text=avg_confidence['Average Confidence'].apply(lambda x: f'{x:.2%}'),
            range_y=[0, 1], # El eje Y va de 0 a 1 (0% a 100%)
            color_discrete_map={'PER': '#636EFA', 'ORG': '#00CC96', 'LOC': '#EF553B', 'MISC': '#AB63FA'}
        )
        grafico_salida.update_layout(xaxis_title="Category", yaxis_title="Average Confidence")

        # 3. Para la lista: Agrupamos por texto y categoría, mostrando solo entidades únicas
        entidades_unicas = {}
        for entidad in entidades_totales:
            key = (entidad['word'].strip().lower(), entidad['entity_group'])
            if key not in entidades_unicas:
                entidades_unicas[key] = entidad

        df_unicas = pd.DataFrame(list(entidades_unicas.values()))
        
        entidades_por_categoria = {}
        for categoria in ['PER', 'ORG', 'LOC', 'MISC']:
            sub_df = df_unicas[df_unicas['entity_group'] == categoria]
            if not sub_df.empty:
                # Ordena alfabéticamente para una vista limpia
                lista_entidades = "\n".join(sorted([f"- {row['word']}" for _, row in sub_df.iterrows()]))
                entidades_por_categoria[categoria] = lista_entidades
            else:
                entidades_por_categoria[categoria] = f"No entities of type '{categoria}' found."

        # --- FIN DE LA LÓGICA MEJORADA ---

        tokens_salida = sum(len(tokenizador.tokenize(e['word'])) for e in entidades_totales)
        confianza_promedio = sum(e['score'] for e in entidades_totales) / len(entidades_totales)

        logging.info(f"Análisis completado en {tiempo_respuesta:.2f} segundos. Entidades encontradas: {len(entidades_totales)}")

        resumen_metricas = (
            f"--- Processing Metrics ---\n"
            f"⏱️ Response Time: {tiempo_respuesta:.2f} seconds\n"
            f"🎟️ Input Tokens: {tokens_entrada_total}\n"
            f"🏷️ Output Tokens (Entities): {tokens_salida}\n"
            f"🎯 Average Confidence: {confianza_promedio:.2%}\n"
            f"----------------\n\n"
        )
        
        return resumen_metricas, grafico_salida, entidades_por_categoria["PER"], \
               entidades_por_categoria["ORG"], entidades_por_categoria["LOC"], \
               entidades_por_categoria["MISC"]

    except Exception as e:
        logging.error(f"Ocurrió un error al procesar el texto: {e}", exc_info=True)
        error_msg = "Error: The application encountered an unexpected problem."
        return error_msg, grafico_vacio, *salidas_vacias


# --- 4. Creación y Lanzamiento de la Interfaz con Tema y Layout Mejorados ---

# Define un tema oscuro con una fuente de estilo tecnológico y texto más visible
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace", "sans-serif"],
).set(
    body_background_fill="#111827",
    body_text_color="#e5e7eb", # <-- Color de texto más claro y visible
    button_primary_background_fill="#3b82f6",
    button_primary_text_color="#ffffff",
    background_fill_primary="#1f2937",
    block_background_fill="#374151",
    block_label_background_fill="#1f2937",
    block_title_text_color="*primary_500",
)

# Usamos gr.Blocks para un control total sobre el layout
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# Named Entity Recognition (NER) Extractor")
    gr.Markdown("This model identifies persons (PER), organizations (ORG), locations (LOC), and other miscellaneous entities (MISC) in Spanish text.")

    # Definimos un layout de dos columnas
    with gr.Row():
        # Columna Izquierda: Entradas y Métricas
        with gr.Column(scale=1):
            input_text = gr.Textbox(lines=15, placeholder="Enter or paste the text you want to analyze here...", label="Input Text")
            submit_button = gr.Button("Analyze Text", variant="primary")
            metrics_output = gr.Textbox(label="📊 Processing Metrics", lines=6, interactive=False)

        # Columna Derecha: Salidas Visuales
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="📈 Average Confidence by Category")
            
            # Usamos un Acordeón para organizar las listas de entidades
            with gr.Accordion("Detailed Entity Lists", open=False):
                per_output = gr.Textbox(label="👤 Persons (PER)", interactive=False)
                org_output = gr.Textbox(label="🏢 Organizations (ORG)", interactive=False)
                loc_output = gr.Textbox(label="📍 Locations (LOC)", interactive=False)
                misc_output = gr.Textbox(label="🏷️ Miscellaneous (MISC)", interactive=False)

    # Conectamos el botón a la función
    submit_button.click(
        fn=encontrar_entidades,
        inputs=input_text,
        outputs=[metrics_output, plot_output, per_output, org_output, loc_output, misc_output]
    )

logging.info("Lanzando la interfaz de Gradio...")
demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)))