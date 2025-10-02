# =========================================================================
# ETAPA 1: BUILDER
# Usamos una imagen de PyTorch que incluye herramientas de desarrollo (devel)
# para compilar cualquier dependencia que lo necesite.
# =========================================================================
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel AS builder

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar las dependencias de Python usando pip
# Esto crea una capa con todas las librerías necesarias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# =========================================================================
# ETAPA 2: FINAL
# Usamos una imagen 'runtime' de PyTorch. Es mucho más ligera porque
# no contiene las herramientas de compilación y desarrollo.
# =========================================================================
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime AS final

# Variable de entorno para logs de Python
ENV PYTHONUNBUFFERED=1

# Crear un usuario no-root para mejorar la seguridad
RUN addgroup --system appuser && adduser --system --ingroup appuser appuser

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar SOLO las librerías instaladas desde la etapa 'builder'.
# No traemos ninguna herramienta de compilación ni archivos innecesarios.
COPY --from=builder /opt/conda/lib/python3.11/site-packages /opt/conda/lib/python3.11/site-packages

# Copiar el código de la aplicación
COPY main.py . 
COPY start.sh .
RUN chmod +x ./start.sh

# Crear y dar permisos al directorio de caché donde se guardará el modelo
RUN mkdir -p /app/.cache && \
    chown -R appuser:appuser /app/.cache

# Cambiar el propietario de los archivos de la aplicación al usuario no-root
RUN chown -R appuser:appuser /app

# Cambiar al usuario no-root
USER appuser

# Exponer el puerto de Gradio
EXPOSE 7860

# Comando para iniciar la aplicación
CMD ["./start.sh"]