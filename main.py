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

    # --- CORRECCIÓN 1: Inicializar todas las variables de salida por defecto ---
    grafico_vacio = go.Figure().add_annotation(text="No hay datos para el gráfico.", showarrow=False)
    metricas_vacias = "--- Métricas ---\n" \
                      "⏱️ Tiempo de Respuesta: N/A\n" \
                      "🎟️ Tokens de Entrada: N/A\n" \
                      "🏷️ Tokens de Salida (Entidades): N/A\n" \
                      "🎯 Confianza Promedio: N/A\n" \
                      "----------------\n\n"
    salidas_vacias = (
        "No se encontraron Personas.",
        "No se encontraron Organizaciones.",
        "No se encontraron Ubicaciones.",
        "No se encontraron Misceláneos."
    )

    if not texto:
        logging.warning("Se recibió una entrada de texto vacía.")
        return metricas_vacias, grafico_vacio, *salidas_vacias

    # Filtro inicial para textos excesivamente largos
    LIMITE_PALABRAS = 2000
    conteo_palabras = len(texto.split())
    if conteo_palabras > LIMITE_PALABRAS:
        logging.warning(f"El texto de entrada ({conteo_palabras} palabras) excede el límite de {LIMITE_PALABRAS} palabras.")
        # --- CORRECCIÓN 2: Devolver el número correcto de valores ---
        mensaje_advertencia = f"⚠️ ADVERTENCIA: El texto ingresado es demasiado largo ({conteo_palabras} palabras).\n" \
                              f"Por favor, reduce el texto a menos de {LIMITE_PALABRAS} palabras."
        return mensaje_advertencia, grafico_vacio, *salidas_vacias

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

        # --- LÓGICA DE AGRUPACIÓN Y GRÁFICO ---
        df_todas_entidades = pd.DataFrame(entidades_totales)
        conteo_categorias = df_todas_entidades['entity_group'].value_counts().reset_index()
        conteo_categorias.columns = ['Categoría', 'Ocurrencias']
        
        grafico_salida = px.pie(
            conteo_categorias,
            values='Ocurrencias',
            names='Categoría',
            title='Distribución de Entidades Encontradas',
            hole=0.4,
            color_discrete_map={'PER': '#636EFA', 'ORG': '#00CC96', 'LOC': '#EF553B', 'MISC': '#AB63FA'}
        )
        grafico_salida.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
        grafico_salida.update_layout(showlegend=False)

        entidades_unicas = {}
        for entidad in entidades_totales:
            key = (entidad['word'].lower(), entidad['entity_group'])
            if key not in entidades_unicas:
                entidades_unicas[key] = entidad

        df_unicas = pd.DataFrame(list(entidades_unicas.values()))
        
        entidades_por_categoria = {}
        for categoria in ['PER', 'ORG', 'LOC', 'MISC']:
            sub_df = df_unicas[df_unicas['entity_group'] == categoria]
            if not sub_df.empty:
                lista_entidades = "\n".join(sorted([f"- {row['word']}" for _, row in sub_df.iterrows()]))
                entidades_por_categoria[categoria] = lista_entidades
            else:
                entidades_por_categoria[categoria] = f"No se encontraron entidades de tipo '{categoria}'."

        tokens_salida = sum(len(tokenizador.tokenize(e['word'])) for e in entidades_totales)
        confianza_promedio = sum(e['score'] for e in entidades_totales) / len(entidades_totales)

        logging.info(f"Análisis completado en {tiempo_respuesta:.2f} segundos. Entidades encontradas: {len(entidades_totales)}")

        resumen_metricas = (
            f"--- Métricas ---\n"
            f"⏱️ Tiempo de Respuesta: {tiempo_respuesta:.2f} segundos\n"
            f"🎟️ Tokens de Entrada: {tokens_entrada_total}\n"
            f"🏷️ Tokens de Salida (Entidades): {tokens_salida}\n"
            f"🎯 Confianza Promedio: {confianza_promedio:.2%}\n"
            f"----------------\n\n"
        )
        
        return resumen_metricas, grafico_salida, entidades_por_categoria["PER"], \
               entidades_por_categoria["ORG"], entidades_por_categoria["LOC"], \
               entidades_por_categoria["MISC"]

    except Exception as e:
        logging.error(f"Ocurrió un error al procesar el texto: {e}", exc_info=True)
        error_msg = "Error: La aplicación encontró un problema inesperado al procesar el texto."
        return error_msg, grafico_vacio, *salidas_vacias

# --- 4. Creación y Lanzamiento de la Interfaz con Múltiples Salidas ---
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace", "sans-serif"],
).set(
    body_background_fill="#111827",
    body_background_fill_dark="#111827",
    body_text_color="#d1d5db",
    body_text_color_dark="#d1d5db",
    button_primary_background_fill="#3b82f6",
    button_primary_background_fill_dark="#3b82f6",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    background_fill_primary="#1f2937",
    background_fill_primary_dark="#1f2937",
    block_background_fill="#374151",
    block_background_fill_dark="#374151",
    block_label_background_fill="#1f2937",
    block_label_background_fill_dark="#1f2937",
    block_title_text_color="*primary_500",
    block_title_text_color_dark="*primary_500",
)

output_components = [
    gr.Textbox(label="📊 Processing Metrics", lines=5, interactive=False),
    gr.Plot(label="📈 Entity Distribution by Category"),
    gr.Textbox(label="👤 Persons (PER)", lines=5, interactive=False),
    gr.Textbox(label="🏢 Organizations (ORG)", lines=5, interactive=False),
    gr.Textbox(label="📍 Locations (LOC)", lines=5, interactive=False),
    gr.Textbox(label="🏷️ Miscellaneous (MISC)", lines=5, interactive=False),
]

iface = gr.Interface(
    fn=encontrar_entidades,
    inputs=gr.Textbox(lines=10, placeholder="Enter or paste the text you want to analyze here...", label="Input Text"),
    outputs=output_components,
    title="Named Entity Recognition (NER) Extractor",
    description="This model identifies persons (PER), organizations (ORG), locations (LOC), and other miscellaneous entities (MISC) in Spanish text. It supports long texts and displays detailed metrics and visualizations.",
    theme=theme
)

logging.info("Lanzando la interfaz de Gradio...")
iface.launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)))