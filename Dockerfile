# 1. Imagen base
FROM python:3.9-slim

# Vvariable de entorno para mejor manejo de logs en Python
ENV PYTHONUNBUFFERED=1

# Crear un usuario y grupo 'appuser' para no ejecutar como root
RUN addgroup --system appuser && adduser --system --ingroup appuser appuser
# -------------------------

# 2. Directorio de trabajo
WORKDIR /app

# 3. Instalar PyTorch para GPU (CUDA 12.8)
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.3.1 torchvision torchaudio

# 4. Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto del código
COPY main.py . 
COPY start.sh . 

RUN chmod +x ./start.sh

# Cambiar el propietario de los archivos a 'appuser'
RUN chown -R appuser:appuser /app
# Cambiar al nuevo usuario
USER appuser
# -------------------------

# 6. Exponer el puerto
EXPOSE 7860

# 7. Comando para iniciar la aplicación
CMD ["./start.sh"]