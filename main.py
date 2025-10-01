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
    grafico_vacio = go.Figure().add_annotation(text="Sin datos para el gráfico.", showarrow=False)
    grafico_vacio.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    metricas_vacias = "--- Métricas de Procesamiento ---\n" \
                      "⏱️ Tiempo de Respuesta: N/A\n" \
                      "🎟️ Tokens de Entrada: N/A\n" \
                      "🏷️ Tokens de Salida (Entidades): N/A\n" \
                      "🎯 Confianza Promedio: N/A\n" \
                      "----------------\n\n"
    salidas_vacias = ("No se encontraron Personas.", "No se encontraron Organizaciones.", "No se encontraron Ubicaciones.", "No se encontraron Misceláneos.")

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

        df_todas_entidades = pd.DataFrame(entidades_totales)
        
        # --- CORRECCIÓN DEL GRÁFICO: Volver a Gráfico de Pastel con Conteo de Ocurrencias ---
        conteo_categorias = df_todas_entidades['entity_group'].value_counts().reset_index()
        conteo_categorias.columns = ['Categoría', 'Ocurrencias']
        
        grafico_salida = px.pie(
            conteo_categorias,
            values='Ocurrencias',
            names='Categoría',
            title='Distribución de Entidades por Categoría',
            hole=0.4,
            color_discrete_map={'PER': '#636EFA', 'ORG': '#00CC96', 'LOC': '#EF553B', 'MISC': '#AB63FA'}
        )
        grafico_salida.update_traces(textposition='inside', textinfo='percent+label')
        grafico_salida.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#e5e7eb",
            showlegend=True # Mostrar leyenda
        )
        
        # --- MEJORA DE FORMATO EN LISTAS DE ENTIDADES ---
        entidades_unicas = {}
        for entidad in entidades_totales:
            key = (entidad['word'].strip().lower(), entidad['entity_group'])
            if key not in entidades_unicas:
                entidades_unicas[key] = entidad
            elif entidad['score'] > entidades_unicas[key]['score']:
                entidades_unicas[key] = entidad

        df_unicas = pd.DataFrame(list(entidades_unicas.values()))
        
        entidades_por_categoria = {}
        for categoria in ['PER', 'ORG', 'LOC', 'MISC']:
            sub_df = df_unicas[df_unicas['entity_group'] == categoria]
            if not sub_df.empty:
                lista_entidades = "\n".join(
                    sorted([f"**{row['word']}** (Confianza: {row['score']:.1%})" for _, row in sub_df.iterrows()])
                )
                entidades_por_categoria[categoria] = lista_entidades
            else:
                entidades_por_categoria[categoria] = f"_No se encontraron entidades de tipo '{categoria}'._"

        tokens_salida = sum(len(tokenizador.tokenize(e['word'])) for e in entidades_totales)
        confianza_promedio = sum(e['score'] for e in entidades_totales) / len(entidades_totales)
        logging.info(f"Análisis completado en {tiempo_respuesta:.2f} segundos.")

        resumen_metricas = (
            f"--- Métricas de Procesamiento ---\n"
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
        error_msg = "Error: La aplicación encontró un problema inesperado."
        return error_msg, grafico_vacio, *salidas_vacias


# --- 4. Creación y Lanzamiento de la Interfaz con Tema y Layout Mejorados ---

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace", "sans-serif"],
).set(
    body_background_fill="#111827",
    body_text_color="#f3f4f6", # Color de texto principal (más blanco)
    button_primary_background_fill="#4f46e5",
    button_primary_text_color="#ffffff",
    background_fill_primary="#1f2937",
    block_background_fill="#1f2937", # Fondo de bloques igual al fondo principal
    block_border_width="0px", # Sin bordes en los bloques
    block_shadow="*shadow_md",
    block_label_background_fill="#111827", # Fondo de etiquetas igual al fondo del cuerpo
    block_label_text_color="#9ca3af", # Color de etiquetas más sutil (gris)
    input_background_fill="#374151",
)

# CSS para el fondo con gradiente
css = """
body {
    background-image: radial-gradient(circle at top, #1e3a8a 10%, #111827);
    background-attachment: fixed;
}
"""

with gr.Blocks(theme=theme, css=css) as demo:
    # Título en Inglés y centrado, Descripción en Español
    gr.Markdown("<h1 style='text-align: center; color: #e5e7eb;'>Named Entity Recognition (NER) Extractor</h1>")
    gr.Markdown("<p style='text-align: center;'>Este modelo identifica personas (PER), organizaciones (ORG), ubicaciones (LOC), y otras entidades (MISC) en texto en español.</p>")

    # Layout principal de dos columnas
    with gr.Row(variant='panel'):
        # Columna Izquierda
        with gr.Column(scale=2, min_width=400):
            input_text = gr.Textbox(lines=20, placeholder="Pega aquí el texto que quieres analizar...", label="Texto de Entrada")
            with gr.Row():
                clear_button = gr.Button("Limpiar")
                submit_button = gr.Button("Analizar Texto", variant="primary")
        
        # Columna Derecha
        with gr.Column(scale=3, min_width=600):
            metrics_output = gr.Textbox(label="Métricas de Procesamiento", lines=6, interactive=False)
            plot_output = gr.Plot(label="Distribución de Entidades")

    # Layout inferior con 4 columnas para las listas detalladas
    with gr.Accordion("Listas Detalladas de Entidades Únicas", open=True):
        with gr.Row():
            per_output = gr.Markdown(label="👤 Personas (PER)")
            org_output = gr.Markdown(label="🏢 Organizaciones (ORG)")
            loc_output = gr.Markdown(label="📍 Ubicaciones (LOC)")
            misc_output = gr.Markdown(label="🏷️ Misceláneas (MISC)")

    # Lógica de los botones
    outputs_list = [metrics_output, plot_output, per_output, org_output, loc_output, misc_output]
    submit_button.click(fn=encontrar_entidades, inputs=input_text, outputs=outputs_list)
    
    # La función para limpiar devuelve None a cada componente de salida
    clear_button.click(lambda: (None, None, None, None, None, None), outputs=outputs_list)

logging.info("Lanzando la interfaz de Gradio...")
demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)))