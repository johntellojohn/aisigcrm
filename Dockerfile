# Usar una imagen base de Python
FROM python:3.12-slim-bookworm

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY . /app

# Instalar las dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --upgrade mysql-connector-python

RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    echo "deb http://mirror.cedia.org.ec/debian bookworm main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://mirror.cedia.org.ec/debian bookworm-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb http://mirror.cedia.org.ec/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Exponer el puerto que usa la aplicación
EXPOSE 5015

# Comando para ejecutar la aplicación
CMD ["python", "aisigcrm.py"]