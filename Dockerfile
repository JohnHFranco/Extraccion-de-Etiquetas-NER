# 1. Imagen base
FROM python:3.9-slim

# 2. Directorio de trabajo
WORKDIR /app        
#workspaces

# 3. Instalar PyTorch solo para CPU
# Usamos el index-url específico para builds de CPU.
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0 torchvision torchaudio

# 3. Instalar PyTorch para GPU (CUDA 12.8)
# RUN python -m pip install --no-cache-dir \
#     --index-url https://download.pytorch.org/whl/cu128 \
#     torch==2.3.1 torchvision torchaudio

# 4. Instalar dependencias (esto aprovecha el cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto del código Y el script de inicio
COPY main.py .
COPY start.sh .

RUN chmod +x ./start.sh

# 6. Exponer el puerto
EXPOSE 7860

# 7. Comando para iniciar la aplicación usando el script
CMD ["./start.sh"]