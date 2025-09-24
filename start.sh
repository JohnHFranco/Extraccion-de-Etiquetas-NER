#!/usr/bin/env bash
set -euo pipefail

echo "Iniciando el Extractor de Entidades NER..."

# --- Configuración del Entorno ---
APP_DIR="/app"
export HF_HOME="${HF_HOME:-$APP_DIR/.cache/huggingface}"
mkdir -p "$HF_HOME"

# --- Variables Configurables ---
export MODEL_ID="${MODEL_ID:-mrm8488/bert-spanish-cased-finetuned-ner}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

echo "=== Configuración de Ejecución ==="
echo "  - ID del Modelo: $MODEL_ID"
echo "  - Puerto de Gradio: $GRADIO_SERVER_PORT"
echo "  - Directorio de Caché de HF: $HF_HOME"
echo "=================================="

# --- BLOQUE DE VERIFICACIÓN ---
echo "📥 Verificando caché del modelo..."
python -c "
import os
from pathlib import Path

cache_dir = Path(os.getenv('HF_HOME', '/app/.cache/huggingface'))
model_id = os.getenv('MODEL_ID', '')
# Transforma el nombre del modelo a como se guarda en la caché
model_cache_path = cache_dir / f'models--{model_id.replace(\"/\", \"--\")}'

if model_cache_path.exists():
    print(f'Modelo encontrado en la caché: {model_cache_path}')
else:
    print('Modelo no encontrado en la caché.')
    print('La primera ejecución descargará el modelo. Esto puede tardar unos minutos.')
"
# ---------------------------------------

# --- Ejecución de la Aplicación ---
echo "Lanzando la aplicación Gradio..."
exec python "$APP_DIR/main.py"