# 1. Imagen base
FROM python:3.9-slim

# 2. Directorio de trabajo
WORKDIR /app

# 3. Instalar dependencias (esto aprovecha el cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el resto del código de la aplicación
COPY main.py .

# 5. Exponer el puerto en el que corre Gradio
EXPOSE 7860

# 6. Comando para iniciar la aplicación
CMD ["python", "main.py"]