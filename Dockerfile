# 1. Imagen base
FROM python:3.9-slim

# 2. Directorio de trabajo
WORKDIR /app

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