# Utiliza una base de imagen de Python con soporte para GPU NVIDIA
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Establece el directorio de trabajo
WORKDIR /app

# Copia el requirements.txt
COPY requirements.txt requirements.txt

# Instala las dependencias, incluyendo las bibliotecas CUDA necesarias
RUN apt-get update && apt-get install -y \
    libnvrtc-dev \
    libcusparse-dev \
    libcublas-dev \
    libnvcc-dev \
    && pip install -r requirements.txt

# Copia el código de la aplicación
COPY . .

# Exponer el puerto si tu aplicación lo requiere (ajusta según sea necesario)
EXPOSE 4080

# Instala cualquier dependencia del sistema operativo que necesite tu aplicación
RUN apt-get install -y <lista_de_paquetes>

# Establece variables de entorno
ENV VARIABLE1=valor1
ENV VARIABLE2=valor2

# Comando para ejecutar tu aplicación
CMD ["python", "./aisigcrm.py"]