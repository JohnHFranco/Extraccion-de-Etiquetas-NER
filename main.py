import gradio as gr
from transformers import pipeline
import os
import time
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

def segment_text(text, tokenizer, max_tokens=500): # Reducido a 500 por el error 514 vs 512
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

    # Inicializar salidas en caso de error o texto vacío
    resumen_metricas = "--- Métricas ---\n" \
                       "⏱️ Tiempo de Respuesta: N/A\n" \
                       "🎟️ Tokens de Entrada: N/A\n" \
                       "🏷️ Tokens de Salida (Entidades): N/A\n" \
                       "🎯 Confianza Promedio: N/A\n" \
                       "----------------\n\n"
    grafico_salida = go.Figure().add_annotation(text="No hay datos para el gráfico.", showarrow=False) # Gráfico vacío
    entidades_por_categoria = {
        "PER": "No se encontraron Personas.",
        "ORG": "No se encontraron Organizaciones.",
        "LOC": "No se encontraron Ubicaciones.",
        "MISC": "No se encontraron Misceláneos."
    }
    resultados_raw = "No se encontraron entidades o hubo un error."


    if not texto:
        logging.warning("Se recibió una entrada de texto vacía.")
        return resumen_metricas, grafico_salida, entidades_por_categoria["PER"], \
               entidades_por_categoria["ORG"], entidades_por_categoria["LOC"], \
               entidades_por_categoria["MISC"], resultados_raw
    
    # Filtro inicial para textos excesivamente largos
    LIMITE_PALABRAS = 2000
    conteo_palabras = len(texto.split())
    if conteo_palabras > LIMITE_PALABRAS:
        logging.warning(f"El texto de entrada ({conteo_palabras} palabras) excede el límite de {LIMITE_PALABRAS} palabras.")
        return "⚠️ ADVERTENCIA: El texto ingresado es demasiado largo " \
               f"({conteo_palabras} palabras).\nPor favor, reduce el texto a " \
               f"menos de {LIMITE_PALABRAS} palabras.", grafico_salida, \
               entidades_por_categoria["PER"], entidades_por_categoria["ORG"], \
               entidades_por_categoria["LOC"], entidades_por_categoria["MISC"], resultados_raw

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

        # --- PREPARACIÓN DE MÉTRICAS Y RESULTADOS ---
        resumen_metricas = (
            f"--- Métricas ---\n"
            f"⏱️ Tiempo de Respuesta: {tiempo_respuesta:.2f} segundos\n"
            f"🎟️ Tokens de Entrada: {tokens_entrada_total}\n"
            f"🏷️ Tokens de Salida (Entidades): {tokens_salida}\n"
            f"🎯 Confianza Promedio: {confianza_promedio:.2%}\n"
            f"----------------\n\n"
        )

        resultados_raw = ""
        if not entidades_totales:
            resultados_raw = "No se encontraron entidades en el texto."
            grafico_salida = go.Figure().add_annotation(text="No hay entidades para el gráfico.", showarrow=False)
        else:
            # Lógica para eliminar duplicados y preservar el orden original del texto
            entidades_vistas = set()
            entidades_unicas_ordenadas = []
            
            for entidad in sorted(entidades_totales, key=lambda x: x['start']):
                # Usamos una tupla con atributos clave para identificar duplicados
                identificador_entidad = (entidad['word'], entidad['entity_group'], entidad['start'], entidad['end'])
                if identificador_entidad not in entidades_vistas:
                    entidades_unicas_ordenadas.append(entidad)
                    entidades_vistas.add(identificador_entidad)
            
            # --- Generar Gráfico y Detalles por Categoría ---
            df_entidades = pd.DataFrame(entidades_unicas_ordenadas)
            
            # Gráfico de pastel de categorías
            conteo_categorias = df_entidades['entity_group'].value_counts().reset_index()
            conteo_categorias.columns = ['Categoría', 'Cantidad']
            grafico_salida = px.pie(
                conteo_categorias,
                values='Cantidad',
                names='Categoría',
                title='Distribución de Entidades por Categoría',
                hole=0.3 # Hace un gráfico de donut
            )
            grafico_salida.update_traces(textposition='inside', textinfo='percent+label')

            # Detalles por categoría
            entidades_por_categoria = {
                "PER": "No se encontraron Personas.",
                "ORG": "No se encontraron Organizaciones.",
                "LOC": "No se encontraron Ubicaciones.",
                "MISC": "No se encontraron Misceláneos."
            }

            for categoria in df_entidades['entity_group'].unique():
                sub_df = df_entidades[df_entidades['entity_group'] == categoria]
                lista_entidades = "\n".join([f"- '{row['word']}' (Confianza: {row['score']:.2%})" for _, row in sub_df.iterrows()])
                entidades_por_categoria[categoria] = f"--- {categoria} ({len(sub_df)}) ---\n" + lista_entidades + "\n"

            # Resultados brutos (todas las entidades listadas como antes)
            for entidad in entidades_unicas_ordenadas:
                resultados_raw += f"Texto: '{entidad['word']}'\n"
                resultados_raw += f"Categoría: {entidad['entity_group']} (Confianza: {entidad['score']:.2%})\n\n"
        
        # Devuelve todas las salidas separadas
        return resumen_metricas, grafico_salida, \
               entidades_por_categoria.get("PER", "No se encontraron Personas."), \
               entidades_por_categoria.get("ORG", "No se encontraron Organizaciones."), \
               entidades_por_categoria.get("LOC", "No se encontraron Ubicaciones."), \
               entidades_por_categoria.get("MISC", "No se encontraron Misceláneos."), \
               resultados_raw

    except Exception as e:
        logging.error(f"Ocurrió un error al procesar el texto: {e}", exc_info=True)
        # En caso de error, devuelve los mensajes de error en los campos correspondientes
        error_msg = "Error: La aplicación encontró un problema inesperado al procesar el texto."
        return error_msg, grafico_salida, entidades_por_categoria["PER"], \
               entidades_por_categoria["ORG"], entidades_por_categoria["LOC"], \
               entidades_por_categoria["MISC"], error_msg


# --- 4. Creación y Lanzamiento de la Interfaz con Múltiples Salidas ---

# Definimos los componentes de salida. ¡Ahora hay 7!
output_components = [
    gr.Textbox(label="📊 Métricas de Procesamiento", lines=5, interactive=False),
    gr.Plot(label="📈 Distribución de Entidades por Categoría"),
    gr.Textbox(label="👤 Entidades: Personas (PER)", lines=5, interactive=False),
    gr.Textbox(label="🏢 Entidades: Organizaciones (ORG)", lines=5, interactive=False),
    gr.Textbox(label="📍 Entidades: Ubicaciones (LOC)", lines=5, interactive=False),
    gr.Textbox(label="🏷️ Entidades: Misceláneas (MISC)", lines=5, interactive=False),
    gr.Textbox(label="📝 Todas las Entidades Encontradas (Raw)", lines=10, interactive=False)
]

iface = gr.Interface(
    fn=encontrar_entidades,
    inputs=gr.Textbox(lines=10, placeholder="Escribe o pega aquí el texto que quieres analizar...", label="Texto de Entrada"),
    outputs=output_components, # Asignamos la lista de componentes de salida
    title="🔎 Reconocimiento de Entidades Nombradas (NER) en Español",
    description="Este modelo identifica personas (PER), organizaciones (ORG), ubicaciones (LOC) y otras entidades misceláneas (MISC) en texto en español. Soporta textos largos y muestra métricas y visualizaciones detalladas."
)

logging.info("Lanzando la interfaz de Gradio...")
iface.launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)))