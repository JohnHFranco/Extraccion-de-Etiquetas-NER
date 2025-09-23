# 1. Imagen base
FROM python:3.9-slim

# 2. Directorio de trabajo
WORKDIR /app

# 3. Instalar PyTorch solo para CPU
# Usamos el index-url específico para builds de CPU.
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision torchaudio

# 3. Instalar PyTorch para GPU (CUDA 12.8)
# RUN python -m pip install --no-cache-dir \
#     --index-url https://download.pytorch.org/whl/cu128 \
#     torch==2.3.1 torchvision torchaudio

# 3. Instalar dependencias (esto aprovecha el cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el resto del código Y el script de inicio
COPY main.py .
COPY start.sh .

# 5. Exponer el puerto
EXPOSE 7860

# 6. Comando para iniciar la aplicación usando el script
CMD ["./start.sh"]