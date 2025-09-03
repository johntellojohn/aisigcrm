# Guía de Despliegue: App de IA con Docker y GPU en Debian
Esta guía detalla el proceso completo para configurar un servidor Debian nuevo y desplegar una aplicación de IA en Python, utilizando Docker, Docker Compose y habilitando el soporte de GPU NVIDIA.

# Fase 1: Preparación del Servidor Base
Configuraciones iniciales del sistema operativo para asegurar un entorno limpio y actualizado.

## 1.1. Actualización del Sistema
Asegúrate de que todos los paquetes del servidor estén actualizados.

sudo apt update && sudo apt upgrade -y

## 1.2. Instalación de Git
Instala Git para poder clonar tu repositorio de código.

sudo apt install git -y

# Fase 2: Instalación y Configuración de Docker
Instalaremos Docker Engine y Docker Compose para la gestión de contenedores. Los siguientes comandos instalan las versiones Community que coinciden con las que solicitaste.

## 2.1. Instalar Docker Engine y Compose
Sigue los pasos oficiales para instalar la última versión de Docker desde su repositorio.

### 1. Desinstalar versiones antiguas si existen
sudo apt-get remove docker docker-engine docker.io containerd runc

### 2. Instalar paquetes de prerrequisito
sudo apt-get install -y ca-certificates curl gnupg

### 3. Añadir la clave GPG oficial de Docker para verificar los paquetes
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

### 4. Añadir el repositorio de Docker a las fuentes de APT
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

### Instala la Dependencia para HTTPS
  sudo apt-get install apt-transport-https -y

### 5. Instalar Docker Engine, CLI, Containerd y el plugin de Compose
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin


## 2.3. Verificar la Instalación de Docker
Una vez reconectado, verifica que ambos comandos funcionan correctamente sin sudo.

docker --version
docker compose version

# Fase 3: Despliegue del Proyecto Base (Sin GPU)
Clonaremos y ejecutaremos el proyecto en Docker usando solo la CPU para verificar la configuración básica.

## 3.1. Clonar el Repositorio del Proyecto
Navega al directorio deseado (ej. /var/www/html) y clona tu proyecto.

### Navega al directorio (créalo si no existe)
sudo mkdir -p /var/www/html
cd /var/www/html

### Clona tu repositorio
git clone [https://tu-repositorio.com/aisigcrm.git](https://tu-repositorio.com/aisigcrm.git)
cd aisigcrm


## 3.3. Levantar los Contenedores
Este comando leerá tus archivos Dockerfile y docker-compose.yml, construirá la imagen y arrancará el contenedor en segundo plano.

docker compose up --build -d

## 3.4. Verificar el Funcionamiento
Revisa que el contenedor esté corriendo y consulta sus logs para detectar posibles errores.

### Verifica que el contenedor esté en estado 'running' o 'up'
docker compose ps

### Muestra los logs en tiempo real (usa Ctrl+C para salir)
docker compose logs -f aisigcrm_app

En este punto, tu API debe estar funcionando en modo CPU en http://<IP_DE_TU_SERVIDOR>:5015.

# Fase 4: Habilitación de la GPU para Docker (NVIDIA/CUDA)
Ahora, integraremos el soporte de GPU al despliegue existente.

## 4.1. Instalar Drivers de NVIDIA en el Servidor
Este es el paso más crítico. Verifica si los drivers ya están instalados:

nvidia-smi

Si el comando funciona: Muestra una tabla con información de la GPU. Puedes continuar.

Si el comando falla (command not found): Debes instalar los drivers. Sigue estos pasos:

sudo apt update && sudo apt upgrade -y

sudo sed -i 's/ main$/ main contrib non-free non-free-firmware/g' /etc/apt/sources.list

sudo apt update

sudo apt install nvidia-detect -y

nvidia-detect (para ver el driver recomendado por Debian)

sudo apt install nvidia-driver firmware-misc-nonfree -y

sudo reboot

Después de reiniciar, vuelve a ejecutar nvidia-smi para confirmar. No continúes hasta que este comando funcione.

## 4.2. Instalar el NVIDIA Container Toolkit
Este componente permite a Docker comunicarse con los drivers de la GPU.

### Añadir el repositorio y la clave de NVIDIA
curl -fsSL [https://nvidia.github.io/libnvidia-container/gpgkey](https://nvidia.github.io/libnvidia-container/gpgkey) | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L [https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list) | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

### Instalar el toolkit y reiniciar Docker
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

Verifica la instalación con el siguiente comando. Debe ejecutar nvidia-smi dentro de un contenedor.

docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

No continúes hasta que esta prueba crucial funcione correctamente.

### Reconstruir la imagen con la nueva base de CUDA y levantar el contenedor
docker compose up --build -d

## 4.5. Verificación Final
Revisa los logs del nuevo contenedor. Deberías ver mensajes de tus librerías de ML (PyTorch, Whisper, etc.) indicando que han detectado y están utilizando el dispositivo CUDA (la GPU).

docker compose logs -f aisigcrm_app