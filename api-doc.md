# Documentación Técnica de las Rutas API
* @app.route('/api/getData', methods=['GET']): Realiza una búsqueda semántica sobre un índice de Pinecone, recupera el más relevante.
*Entrada:* 
    {
        "include_values": [bool], #Si se establece como true, incluye los vectores originales en la respuesta de coincidencias pero va por default como False 
        "top_k": [int], #Número de vectores más similares que se desean recuperar del índice.
        "index": "[string]", # Nombre del índice de Pinecone a consultar.
        "name_space": "[string]" #Espacio de nombres dentro del índice donde se almacenan los vectores, permite segmentar los datos dentro de un mismo índice
    }
*Salida Esperada:*	
    {
        "matches": [
            {
            "id": "example-id",
            "score": 0.9324,
            "values": [/* vector values, if include_values=True */]
            },
            ...
        ],
        "namespace": "real",
        "usage": {
            "read_units": 1
        }
    }
*Dependencias Externas:* pinecone-client (Pinecone, pc.Index(...), index.query(...))
*Dependencias Internas:* PINECONE_API_KEY_PRUEBAS

* @app.route('/api/deleteIndex', methods=['DELETE']): Elimina por completo un índice especificado en Pinecone.
*Entrada*:
    {
    "index": "[string]" #Nombre del índice de Pinecone que se desea eliminar.
    } 
*Salida Esperada*:
    "Todos los registros han sido eliminados"
*Dependencias Externas:* pinecone-client (Pinecone, pc.Index(...))
*Dependencias Internas:* PINECONE_API_KEY_PRUEBAS

* @app.route('/api/createIndex', methods=['POST']): Crea un nuevo índice en Pinecone con una configuración predefinida.
*Entrada*: 
    {
    "index": "[string]" #Nombre del índice a crear
    }
*Salida Esperada*:  
    "Indice Creado con éxito" o "El índice ya existe"
*Dependencias Externas:* pinecone-client (Pinecone, pc.Index(...), ServerlessSpec(...))
*Dependencias Internas:* PINECONE_API_KEY_PRUEBAS


* @app.route('/api/upsertData', methods=['POST']): Inserta un vector en un índice de Pinecone con lógica adicional desde un URL.
*Entrada*: 
    {
    "id_vector": "[string]", #Identificador único para el vector que se insertará en el índice de Pinecone.
    "values_vector": "[string]", #Representación del vector numérico en formato string que será insertado, debe tener la dimensión adecuada para el índice.
    "values_intention": "[string]", #Intención asociado al vector. Se guarda como metadato, útil para clasificación o recuperación semántica.
    "name_space": "[string]", #Namespace dentro del índice de Pinecone donde se insertará el vector. Permite agrupar vectores lógicamente sin crear múltiples índices.
    "index_name": "[string]", #Nombre del índice de Pinecone donde se realizará la operación de inserción.
    "pagina_web_url": "[https://ejemplo.com]" #URL de una página web relacionada con el vector.
    }
*Salida Esperada*: 
    "Información ingresada con éxito"
*Dependencias Externas:* pinecone-client (Pinecone, pc.create_index(...), index.upsert(...), index.fetch(...), index.delete(...)), OpenAIEmbeddings, requests, BeautifulSoup, unidecode
*Dependencias Internas:* PINECONE_API_KEY_PRUEBAS, OPENAI_API_KEY

* @app.route('/api/updateData', methods=['PUT']): Actualiza directamente un vector existente.
*Entrada:*
    {
    "id_vector": "[string]", #Identificador único del vector que se desea actualizar. Debe existir previamente en el índice especificado.
    "values_vector": "[string]", #Nueva representación del vector numérico
    "name_space": "[string]", #Namespace donde se encuentra el vector a actualizar
    "index_name": "[string]" #Nombre del índice de Pinecone donde se realizará la actualización.
    }
*Salida Esperada:* 
    "Información actualizada con éxito"
*Dependencias Externas:* pinecone-client (Pinecone, pc.Index(...), index.upsert(...)), OpenAIEmbeddings, unidecode
*Dependencias Internas:* PINECONE_API_KEY_PRUEBAS, OPENAI_API_KEY

* @app.route('/api/deleteData', methods=['DELETE']): Elimina un vector específico dentro de un índice de Pinecone, utilizando su identificador (id_vector) y el namespace al que pertenece.
*Entrada:* 
    { 
    "index_name": "[string]", #Nombre del índice de Pinecone donde se encuentra almacenado el vector que se desea eliminar.
    "name_space": "[string]", #Namespace específico dentro del índice que organiza los vectores
    "id_vector": "[string]" #Identificador único del vector que se desea eliminar del índice
    }

*Salida Esperada:* 
    "Vector Prompt fue eliminado con éxito"
*Dependencias Externas:* pinecone-client (Pinecone, pc.Index(...), index.delete(...))
*Dependencias Internas:* PINECONE_API_KEY_PRUEBAS

* @app.route('/api/deleteNamespace', methods=['DELETE']): Elimina todos los vectores contenidos en un namespace específico dentro de un índice de Pinecone.
*Entrada:*
    {
    "index_name": "[string]", #Nombre del índice de Pinecone que contiene el namespace a eliminar
    "name_space": "[string]" #Nombre del namespace dentro del índice que se desea eliminar
    }
*Salida Esperada:* 
    "El namespace real fue eliminado con éxito"
*Dependencias Externas:* pinecone-client (Pinecone, pc.Index(...), index.delete(delete_all=True, namespace=...))
*Dependencias Internas:* PINECONE_API_KEY_PRUEBAS

* @app.route('/api/deleteHistory', methods=['DELETE']): Elimina por completo el contenido de historiales de usuario almacena del namespace dentro de un índice de Pinecone.
*Entrada:*
    {
    "index": "[string]" #Nombre del índice de Pinecone del cual se eliminará completamente el contenido de historiales de usuario, específicamente del namespace asociado a dicho índice
    }
*Salida Esperada:* 
    {
    "response": "Historial eliminado correctamente."
    }
*Dependencias Externas:* pinecone-client (Pinecone, pc.Index(...), index.delete(delete_all=True, namespace='user_history')), openai
*Dependencias Internas:* PINECONE_API_KEY_PRUEBAS

* @app.route('/api/upsertFile', methods=['POST']): Extrae texto de un archivo (PDF, Word, Excel o TXT) a partir de su URL, lo convierte en un vector y lo almacena (o actualiza) en Pinecone con metadatos.
*Entrada:*
    {
    "id_vector": "[string]", #Identificador único para el vector que se generará o actualizará en el índice de Pinecone.
    "link_file_id": "[string]", #Identificador relacionado con el archivo original
    "link_file": "[https://example.com/document.pdf]", #URL pública del archivo a procesar
    "type_file": "[string]", #Tipo de archivo que se está procesando por ejemplo: "pdf", "docx", "xlsx", "txt".
    "name_space": "[string]", #Espacio de nombres en Pinecone donde se almacenará el vector
    "index": "[string]" #Nombre del índice en Pinecone donde se insertará o actualizará el vector
    }
*Salida Esperada:* 
    "Información ingresada con éxito"
*Dependencias Externas:* pinecone-client, OpenAIEmbeddings, requests, PyPDF2, mammoth, pandas, openpyxl, xlrd
*Dependencias Internas:* PINECONE_API_KEY_PRUEBAS, OPENAI_API_KEY

* @app.route('/api/upsertFileGeneral', methods=['POST']): Extraer texto desde un archivo (PDF, DOCX, DOC, TXT, XLS, XLSX), genera su representación vectorial con OpenAI, y lo almacena o actualiza en un índice de Pinecone junto con metadatos.
*Entrada:* 
    {
    "id_vector": "[string]", #Identificador único del vector que será generado a partir del contenido del archivo
    "link_file_id": "[string]", #Identificador de referencia asociado al archivo original
    "link_file": "https://example.com/document.pdf", #URL directa al archivo que se desea procesar
    "type_file": "[string]", #Tipo del archivo que se está enviando
    "name_space": "[string]", #Espacio de nombres de Pinecone en el que se almacenará el vector
    "index": "[string]" #Nombre del índice de Pinecone donde se insertará o actualizará el vector con su información
    }
*Salida Esperada:* 
    "Información ingresada con éxito"
*Dependencias Externas:* pinecone-client, OpenAIEmbeddings, requests, PyPDF2, mammoth, pandas, openpyxl, xlrd
*Dependencias Internas:* PINECONE_API_KEY_PRUEBAS, OPENAI_API_KEY

* @app.route('/api/chatbot', methods=['POST']): Este API funciona como interfaz conversacional con un chatbot potenciado por un modelo de GPT. La lógica del chatbot es:

    1. Recibir una pregunta del usuario mediante la validación de las entradas.

    2. Recuperar contexto relevante desde Pinecone (documentos relacionados y/o historial de conversación).

    3. Usar un modelo de lenguaje (gpt-4o-mini) para generar una respuesta.

    4. Si hay un JSON estructurado (json_gpt), lo modifica o completa según lo que el usuario solicita.

    5. Devuelve una respuesta conversacional, un JSON (si aplica) y un estado de la conversación.
*Entradas:*
    {
    "question": "[string]", #La pregunta que quieres enviar al chatbot.
    "user_id": "[string]", #Identificador del usuario para manejar historial.
    "max_histories": [int], #Máximo de interacciones a recordar (opcional)
    "name_space": "[string]", #Namespace de Pinecone (opcional, default='real')
    "json_gpt": [object], #JSON dinámico para completar/modificar (opcional)
    "index": "[string]" # Nombre del índice en Pinecone que se está utilizando.
    }
*Salida Esperada:*
    {
    "respuesta": "Texto de respuesta del chatbot",
    "json_gpt": {
        "[JSON modificado (si se proporcionó uno)]"
    },
    "estado_conversacion": "[estado actual:('pendiente', 'esperando confirmación', 'finalizado')]"
    }
*Dependencias Externas:* Pinecone, OpenAI (GPT-4o-mini), OpenAIEmbeddings, LangChain (load_qa_chain), flask, json, re, datetime, sys
*Dependencias Internas:* PINECONE_API_KEY_PRUEBAS, OPENAI_API_KEY
* Se usa ChatOpenAI con el modelo gpt-4o-mini a través de LangChain, se utiliza para: generar respuestas al usuario, interpretar y completar estructuras JSON dinámicas, evaluar contexto proveniente de Pinecone (los documentos e historial).
* Se usa Pinecone como su base vectorial de las siguientes formas principales: recupera documentos relevantes relacionados con la pregunta, guarda el historial de interacción del usuario como texto vectorizado, ese historial se recupera y se reinyecta al modelo para conservar contexto conversacional.
* Toda la lógica está construida sobre Pinecone y embeddings de OpenAI.

* @app.route('/api/uploadPdfChat', methods=['POST']): Endpoint principal para procesar un PDF (texto e imágenes), y almacenar su contenido y metadata vectorizado en Pinecone para futuras consultas con un chatbot.
*Ejemplo de Entrada:* 
    {
        "file_url": "https://openknowledge.fao.org/server/api/core/bitstreams/254de3d2-6e96-4344-9a4d-cab35a8732dc/content", # URL del PDF a procesar
        "index_name": "chatbotprueba", # Nombre del índice en Pinecone donde se desea guardar los datos.
        "pdf_id": "manual_pdf_003" # Un ID único para este PDF, se usará como namespace en Pinecone
    }
*Sugerencias para Estructurar Pregunta al Chatbot sobre una Imagen:* Si hay que hablar de cierta imagen, la pregunta debe incluir:
    * Número de página y número/referencia de la imagen.
    * Y 1-3 palabras clave únicas o frases cortas extraídas directamente del contenido visible de esa imagen o de su descripción AI. 
    * Ejemplo: 
    {    "user_id": "12345",
    "index": "chatbotprueba",
    "name_space": "manual_pdf_002",
    "question": "En el PDF manual_pdf_002, explícame la importancia y el contexto de la imagen 2 que está en la página 18. Esta imagen muestra una interfaz de chat con el usuario 'Ronnie' preguntando 'Where can I find the nearest donation bin?' y tiene el número de teléfono 1303-555-1212."    }
*Salida Esperada:*
    {
    "estado_conversacion": false,
    "json_gpt": false,
    "respuesta": "La imagen 2 en la página 18 del PDF 'manual_pdf_002' es importante porque ilustra un ejemplo de interacción en una interfaz de chat, que es una herramienta clave para la comunicación con los clientes. En este caso, el usuario 'Ronnie' está haciendo una consulta sobre la ubicación del contenedor de donaciones, lo que refleja cómo los clientes pueden utilizar el chat para obtener información rápida y directa.\n\nEl contexto de esta imagen se relaciona con la funcionalidad de chat que se menciona en el documento, donde se describe cómo los usuarios pueden comunicarse con el soporte al cliente a través de un sistema de chat. La interacción muestra la dinámica de un chat en tiempo real, donde el cliente plantea una pregunta y el agente (en este caso, Erik) responde solicitando información adicional (el código postal) para poder ayudar mejor.\n\nEl texto OCR extraído de la imagen incluye el nombre y número de teléfono de Ronnie, así como su consulta y la respuesta del agente. Esto resalta la importancia de tener un sistema de chat accesible y eficiente para mejorar la experiencia del cliente y facilitar la resolución de dudas o problemas."
    }
*Dependencias Externas:* PyMuPDF, Pillow, pytesseract, langchain-openai, pinecone-client
*Dependencias Internas:* base64, PINECONE_API_KEY_PRUEBAS, OPENAI_API_KEY

* @app.route('/api/deleteFile', methods=['DELETE']): Elimina un vector previamente almacenado en Pinecone, asociado a un archivo específico, dentro de un espacio de nombres determinado.

*Ejemplo de Entrada:* 
    {
    "file_id": "[string]",        // Identificador único del vector/archivo que se desea eliminar del índice.
    "namespace": "[string]",      // Espacio de nombres en Pinecone donde se encuentra el vector.
    "index": "[string]"           // Nombre del índice en Pinecone desde el cual se eliminará el vector.
    }
*Salida Esperada:*
    {
    "response": "Archivo [file_id] eliminado correctamente en namespace [namespace].",
    "status_code": 200
    }
*Dependencias Externas:* pinecone-client, openai, flask
*Dependencias Internas:* PINECONE_API_KEY_PRUEBAS, OPENAI_API_KEY

* @app.route('/api/consultaSql', methods=['POST']): Recibe una pregunta en lenguaje natural del usuario, la transforma en una consulta SQL, la ejecuta sobre una base de datos MySQL y genera una respuesta en lenguaje natural basada en los resultados obtenidos.
*Ejemplo de Entrada:* 
    {
    "question": "[string]",     // Pregunta en lenguaje natural formulada por el usuario.
    "user_id": "[string]"       // (Opcional) Identificador del usuario que realiza la consulta.
    }
*Salida Esperada:*
    {
    "user_question": "[string]",        // Pregunta original realizada por el usuario.
    "chatbot_answer": "[string]",       // Respuesta en lenguaje natural generada a partir de los resultados de la consulta.
    "generated_sql": "[string]",        // Consulta SQL generada a partir de la pregunta.
    "status": "success"
    }
*Dependencias Externas:* flask, mysql-connector-python, openai (u otra librería LLM si se usa para la generación SQL o respuesta), traceback
*Dependencias Internas:* generate_sql_from_question, execute_mysql_query, generate_natural_response


## Notas: Dependencias Críticas y Consideraciones para Refactorización
* Dependencias Críticas: 
    - Gestión de APIs de OpenAI y Pinecone.
    - Librerías principales (Flask, LangChain, Pandas/NumPy, PyPDF2/PdfReader, Mammoth,BeautifulSoup)
* Ausencia de manejo de errores en caso de fallo en llamadas como en el guardado o carga de archivos; se puede implementar bloques de try/except en puntos críticos y registrar los posibles errores.
* Evitar código duplicado para procesar diferentes tipos de archivos, sin clara diferencia entre: /api/upsertFileGeneral y /api/upsertFile.
* El endpoint /api/chatbot está sobrecargado y maneja múltiples responsabilidades; se recomienda dividir en múltiples funciones con responsabilidades únicas para establecer un flujo de trabajo más claro. 
* Implementar algúna medida de seguridad ya que no hay autenticación ni validación de entrada y sería bueno de aplicarlo.
* Si existe en el futuro errores por el tamaño del archivo pdf que se desea subir procesar en /api/uploadPdfChat, se sugiere aplicar el siguiente modelo:  https://ai.google.dev/gemini-api/docs/document-processing?hl=en&authuser=2&lang=python 
* Para el /api/uploadPdfChat, se optó por utilizar un método híbrido, extraer la descripción en lenguaje natural de cada imagen y vectorizarla como texto, y usar el modelo CLIP para generar embeddings visuales de las imagenes y almacenarlos en Pinecone (soporta embeddings multimodales).

* @app.route('/api/analizardoc', methods=['POST']): Analiza un documento (PDF o imagen) desde una URL a partir de una pregunta o instrucción. Extrae la información solicitada en un formato JSON estructurado y, además, genera una respuesta conversacional que resume los hallazgos.
*Entrada:* {
        "file_url": "[string]",     // URL pública del archivo PDF o imagen (png, jpg, etc.).
        "prompt": "[string]"        // La pregunta o instrucción sobre qué información extraer.
    }
*Salida Esperada:*
    {
        "structured_data": {        // El JSON con la información extraída según el prompt.
            "campo_extraido_1": "valor1",
            "campo_extraido_2": "valor2"
        },
        "conversational_response": "[string]" // Una respuesta en lenguaje natural resumiendo los datos.
    }

---
* @app.route('/api/orquestador_gpt', methods=['POST']): Gestiona un flujo de conversación multi-paso (ej. agendamiento de citas). Mantiene el estado de la conversación, obtiene datos de APIs externas si es necesario, y usa un LLM para formular la siguiente pregunta al usuario de manera inteligente.
*Entrada:* {
        "flujo_id": [int],          // ID del flujo de conversación a ejecutar (desde la base de datos).
        "chat_id": "[string]",      // Identificador único para la sesión de chat.
        "mensaje_usuario": "[string]", // La respuesta más reciente del usuario.
        "estado_actual": {},        // Un objeto JSON con los datos recolectados hasta el momento.
        "db_name": "[string]"       // Nombre de la base de datos a la que debe conectarse el orquestador.
    }
*Salida Esperada:*
    {
        "mensaje_bot": "[string]",      // El siguiente mensaje o pregunta para el usuario.
        "nuevo_estado": {               // El estado de la conversación actualizado con la última respuesta.
            "variable1": "valor1",
            "variable2": "valor2"
        },
        "accion": "[string]"            // Una acción para el frontend (ej: "finalizado", "reversar_paso").
    }

---
* @app.route('/api/transcribir_voz_a_texto', methods=['POST']): Recibe un archivo de audio y utiliza el modelo Whisper de OpenAI para convertir el habla en texto. Devuelve únicamente la transcripción.
*Entrada (form-data):* - **KEY:** `archivo_audio`
    - **VALUE:** (Archivo de audio, ej: `audio.ogg`, `audio.mp3`)
*Salida Esperada:*
    {
        "transcripcion": "[string]"     // El texto extraído del archivo de audio.
    }

---
* @app.route('/api/inscribir_voz', methods=['POST']): Registra la voz de un usuario creando un perfil de voz robusto. Requiere múltiples archivos de audio para generar una "huella de voz" promediada, lo que aumenta significativamente la precisión.
*Entrada (form-data):* - **KEY:** `db_name`, **VALUE:** `[string]`
    - **KEY:** `user_id`, **VALUE:** `[string]`
    - **KEY:** `nombre_usuario`, **VALUE:** `[string]`
    - **KEY:** `archivos_audio`, **VALUE:** (Primer archivo de audio)
    - **KEY:** `archivos_audio`, **VALUE:** (Segundo archivo de audio)
    - **KEY:** `archivos_audio`, **VALUE:** (Tercer archivo de audio, y así sucesivamente...)
*Salida Esperada:*
    {
        "status": "exito",
        "mensaje": "La voz del usuario '[nombre_usuario]' ha sido registrada con un perfil de voz mejorado."
    }

---
* @app.route('/api/identificar_hablante', methods=['POST']): Compara un archivo de audio con las huellas de voz registradas en la base de datos para identificar a quién pertenece la voz.
*Entrada (form-data):* - **KEY:** `db_name`, **VALUE:** `[string]`
    - **KEY:** `archivo_audio`, **VALUE:** (Archivo de audio a identificar)
*Salida Esperada:*
    {
        "hablante_identificado": "[string]", // Nombre del usuario reconocido o "desconocido".
        "confianza": [float]                 // Puntuación de similitud (entre -1.0 y 1.0).
    }

* @app.route('/api/analizar_emocion', methods=['POST']): Analiza un archivo de audio para detectar la emoción predominante en la voz. Devuelve una alerta binaria (0 para estable, 1 para inestable) si la emoción detectada corresponde a miedo, ira o disgusto.
*Entrada (form-data):*
- **KEY:** `archivo_audio`
- **VALUE:** (Archivo de audio, ej: `voz_cliente.wav`)
*Salida Esperada:*
    {
        "alerta": 1,
        "mensaje": "Emocion de voz inestable. PRECAUCION",
        "emocion_detectada": "angry",
        "confianza": 0.987
    }
**Nota:** El campo `alerta` será `0` para emociones estables (como 'happy', 'sad', 'neutral') y `1` para emociones de precaución ('fear', 'angry', 'disgust').