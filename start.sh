#!/usr/bin/env bash
# Modo seguro: el script fallará si un comando falla o usa una variable no definida.
set -euo pipefail

echo "Iniciando el Extractor de Entidades NER..."

# --- Configuración del Entorno ---
# Define el directorio de trabajo base.
APP_DIR="/app"
# Configura el caché de Hugging Face para que sea persistente si se monta un volumen.
export HF_HOME="${HF_HOME:-$APP_DIR/.cache/huggingface}"
# Crea el directorio de caché si no existe.
mkdir -p "$HF_HOME"

# --- Variables Configurables ---
# Usa las variables de entorno si existen, si no, usa valores por defecto.
export MODEL_ID="${MODEL_ID:-mrm8488/bert-spanish-cased-finetuned-ner}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

echo "=== Configuración de Ejecución ==="
echo "  - ID del Modelo: $MODEL_ID"
echo "  - Puerto de Gradio: $GRADIO_SERVER_PORT"
echo "  - Directorio de Caché de HF: $HF_HOME"
echo "=================================="

# --- Ejecución de la Aplicación ---
# 'exec' reemplaza este script con el proceso de Python.
# Esto es crucial para que Docker maneje las señales de parada correctamente.
echo "Lanzando la aplicación Gradio..."
exec python "$APP_DIR/main.py"