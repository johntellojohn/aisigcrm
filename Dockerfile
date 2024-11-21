# Usar una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el requirements.txt
COPY requirements.txt requirements.txt

# Instala las dependencias, incluyendo las bibliotecas CUDA necesarias
RUN pip install -r requirements.txt

# Exponer el puerto si tu aplicación lo requiere (ajusta según sea necesario)
EXPOSE 4080

# Comando para ejecutar tu aplicación
CMD ["python", "./aisigcrm.py"]