# Pasos para levantar el proyecto AISIGCRM en un entorno local:
1. Clonar el repositorio mediante- git clone https://github.com/johntellojohn/aisigcrm.git
2. Ingresar a la carpeta y ruta correspondiente de "aisigcrm"
3. Asegurar la validación de versiones de dependencias.
4. Crear una imagen de Docker para correr el dockerfile: 
    docker build -t aisigcrm .
5. Crear un archivo ".env", aquí agregamos los keys de Pinecone y OpenAI (ingresar las llaves sin comillas ni espacios)
6. Levantamos la imagen en el puerto designado para ejecutar un contenedor designado para el proyecto: 
    docker run --env-file .env -p 5010:5010 aisigcrm
7. Tenemos ahí el Flask levantado y corriendo, para comprobarlo podemos entrar a: http://localhost:5010/index donde nos debería mostrar el mensaje de confirmación: "Hello, You Human!!"
8. Para probar las APIs, se puede acceder a sus funciones mediante el nombre del endpoint y su cuerpo requerido por Postman. 

## Librerías Claves:
* Flask: Ideal para manejar las APIs.
* OpenAI: Por nuestro uso de GPT.
* Langchain: Permite orquestrar LLMs.
* Pinecone: Base de datos vectoriales, necesarios para la gestión de datos.
* PyPDF2: Para leer/extraer texto de PDFs