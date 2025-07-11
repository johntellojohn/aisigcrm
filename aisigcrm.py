from flask import Flask, jsonify, request
from openai import OpenAI
from collections import deque
import pandas as pd
import numpy as np
from unidecode import unidecode
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import os
import ast
import re
from datetime import datetime
from bs4 import BeautifulSoup
from google.cloud import vision
import openai
import traceback
import sys
import re
import mammoth
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI  
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from dotenv import load_dotenv
# Importaciones necesarias para el manejo de imágenes y OCR.
from PyPDF2 import PdfReader
from io import BytesIO # permite tratar bytes como si fueran un archivo.
import pytesseract # librería que interactúa con el motor Tesseract OCR.
from PIL import Image # para abrir y manipular datos de imágenes
import fitz # Alias para PyMuPDF, librería para trabajar con PDFs
import base64 # Para codificar imágenes y enviarlas a la API de OpenAI
from langchain.text_splitter import CharacterTextSplitter 
from dotenv import load_dotenv
import mysql.connector

print("Este es un mensaje de prueba", flush=True)  # M  todo 1
sys.stdout.flush()  # M  todo 2

load_dotenv()

# Obtener credenciales desde .env
os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Inicializar Pinecone
pc = Pinecone(os.getenv('PINECONE_API_KEY'))


app = Flask(__name__)


############
# Chat Bot #
############


PINECONE_API_KEY_PRUEBAS = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@app.route('/index')
def index():
    return "Hello, You Human!!"

@app.route('/api/getData', methods=['GET'])
def get_data():

    include_values = request.args.get('include_values', False)
    top_k = request.args.get('top_k', 3)
    index = request.args.get('index', "chatbotprueba")
    name_space = request.args.get('name_space', "real")

    '''
        /api/getData?include_values=False&top_k=3&index=chatbot&name_space=real
    '''

    try:
        
        pc = Pinecone(api_key=PINECONE_API_KEY_PRUEBAS)
        index = pc.Index(index)

        query_response = index.query(
            namespace=name_space,
            vector=[0.3] * 1536,
            top_k=top_k,
            include_values=include_values
        )
        
        data = {
            'matches': [
                {
                    'id': match.id,
                    'score': match.score,
                    'values': match.values
                } for match in query_response.matches
            ],
            'namespace': query_response.namespace,
            'usage': {
                'read_units': query_response.usage.read_units
            }
        }

        return jsonify(data), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/api/deleteIndex', methods=['DELETE'])
def delete_index():

    # Cuerpo
    data = request.get_json()
    index = data.get('index')

    """
        {
            "index": "chatbot"
        }
    """

    if not index:
        return jsonify(response="El index es requerido."), 400

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY_PRUEBAS)
        pc.delete_index(index)

        return "Todos los registros han sido eliminados", 200

    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/api/createIndex', methods=['POST'])
def create_index():

    # Cuerpo
    data = request.get_json()
    index = data.get('index')

    """
        {
            "index": "chatbot"
        }
    """

    if not index:
        return jsonify(response="El index es requerido."), 400

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY_PRUEBAS)

        if index not in pc.list_indexes().names():
            pc.create_index(
                name=index,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            return "Indice Creado con éxito", 200
        else:
            return "El índice ya existe", 200
        
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/api/upsertData', methods=['POST'])
def upsert_data():
    data = request.get_json()
    id_vector = data.get('id_vector')
    values_vector = data.get('values_vector')
    values_intention = data.get('values_intention')
    name_space = data.get('name_space')
    index_name = data.get('index_name')
    pagina_web_url = data.get('pagina_web_url', '')

    if not index_name or not id_vector or not values_vector or not values_intention or not name_space:
        return jsonify(response="Se requiere de la siguiente información (id_vector, values_vector, name_space, index_name)."), 400

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY_PRUEBAS)

        # Verificar si el índice existe, si no, crearlo
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        index = pc.Index(index_name)

        # Agregar URL
        if pagina_web_url:
            response = requests.get(pagina_web_url)
            response.raise_for_status()

            # Analizar el contenido HTML de la página
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extraer información básica
            title = soup.title.string if soup.title else 'No title found'
            h1_tags = [h1.get_text() for h1 in soup.find_all('h1')]
            p_tags = [p.get_text() for p in soup.find_all('p')]

            # Preparar la respuesta
            url_info = f"Informacion de URL:\nTitle: {title}\nH1 Tags: {h1_tags}\nP Tags: {p_tags}\n"

            # Añadir la información de la URL a formatted_text y vector
            values_vector += f"\n\n{url_info}"

        # Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Normalizar y formatear el texto
        normalized_text = unidecode(values_vector)
        lines = normalized_text.split('. ')
        formatted_text = '\n'.join(lines)

        # Generar vector de embedding
        vector = embeddings.embed_query(values_vector)

        # Buscar el vector con el ID "InstruccionesDelBot"
        instructions_id = "IntencionesDelBot"
        existing_vector = index.fetch(ids=[instructions_id], namespace=name_space)
        values_intructions = """
            Lista de intenciones:
        """ + values_intention  

        if instructions_id not in existing_vector['vectors']:
            instructions_values = embeddings.embed_query(values_intructions)
            index.upsert(
                vectors=[
                    {
                        "id": instructions_id,
                        "values": instructions_values,
                        "metadata": {
                            "text": values_intructions
                        }
                    }
                ],
                namespace=name_space
            )
        else:
            index.delete(ids=instructions_id, namespace=name_space)
            instructions_values = embeddings.embed_query(values_intructions)
            index.upsert(
                vectors=[
                    {
                        "id": instructions_id,
                        "values": instructions_values,
                        "metadata": {
                            "text": values_intructions
                        }
                    }
                ],
                namespace=name_space
            )

        # Buscar el vector con el ID "Prompt"
        existing_new_vector = index.fetch(ids=[id_vector], namespace=name_space)

        if id_vector not in existing_new_vector['vectors']:
            index.upsert(
                vectors=[
                    {
                        "id": id_vector,
                        "values": vector,
                        "metadata": {
                            "text": formatted_text
                        }
                    }
                ],
                namespace=name_space
            )

        else:
            index.delete(ids=id_vector, namespace=name_space)
            index.upsert(
                vectors=[
                    {
                        "id": id_vector,
                        "values": vector,
                        "metadata": {
                            "text": formatted_text
                        }
                    }
                ],
                namespace=name_space
            )

        index.describe_index_stats()
        return "Información ingresada con éxito", 200

    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/api/updateData', methods=['PUT'])
def update_data():
    # Obtener datos del request
    data = request.get_json()
    id_vector = data.get('id_vector')
    values_vector = data.get('values_vector')
    name_space = data.get('name_space')
    index_name = data.get('index_name')

    if not all([id_vector, values_vector, name_space, index_name]):
        return jsonify(response="Se requiere de la siguiente información (id_vector, values_vector, name_space, index_name)."), 400

    try:
        # Inicializar Pinecone y obtener el índice
        pc = Pinecone(api_key=PINECONE_API_KEY_PRUEBAS)
        index = pc.Index(index_name)

        # Normalizar y formatear el texto
        normalized_text = unidecode(values_vector)
        lines = normalized_text.split('. ')
        formatted_text = '\n'.join(lines)

        # Crear una cadena unificada de valores si es necesario
        values_str = ' '.join(map(str, values_vector))

        # Generar vector de embedding
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector = embeddings.embed_query(values_str)

        # Actualizar el vector en Pinecone
        index.upsert(
            vectors=[
                {
                    "id": id_vector,
                    "values": vector,
                    "metadata": {
                        "text": formatted_text
                    }
                }
            ],
            namespace=name_space
        )

        return "Información actualizada con éxito", 200
    
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/api/deleteData', methods=['DELETE'])
def delete_data():
    # Cuerpo
    data = request.get_json()
    names_space = data.get('name_space')
    index_name = data.get('index_name')
    id_vector = data.get('id_vector')

    if not index_name or not names_space:
            return jsonify(response="Se requiere de la siguiente información (index_name, name_space)."), 400

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY_PRUEBAS)
        index = pc.Index(index_name)

        index.delete(ids=[id_vector], namespace=names_space)

        return (f"Vector {id_vector} fué eliminado con éxito"), 200
    
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/api/deleteNamespace', methods=['DELETE'])
def delete_namespace():
    # Cuerpo
    data = request.get_json()
    names_space = data.get('name_space')
    index_name = data.get('index_name')

    if not index_name or not names_space:
            return jsonify(response="Se requiere de la siguiente información (index_name, name_space)."), 400

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY_PRUEBAS)
        index = pc.Index(index_name)

        index.delete(delete_all=True, namespace=names_space)

        return (f"El namespace {names_space} fué eliminado con éxito"), 200
    
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    """
    Endpoint para manejar las interacciones con el chatbot.
    Procesa preguntas, mantiene historial de conversación y maneja JSON dinámicos.
    
    Parámetros esperados en el JSON de la solicitud:
    - question: La pregunta del usuario (requerido)
    - user_id: ID del usuario (requerido)
    - max_histories: Máximo de interacciones a recordar (opcional, default=10)
    - name_space: Namespace de Pinecone (opcional, default='real')
    - json_gpt: JSON dinámico para completar/modificar (opcional)
    - index: Nombre del índice Pinecone a usar (requerido)
    
    Retorna:
    - respuesta: Texto de respuesta del chatbot
    - json_gpt: JSON modificado (si se proporcionó uno)
    - estado_conversacion: Estado actual ('pendiente', 'finalizado')
    """
    # - estado_conversacion: Estado actual ('pendiente', 'esperando confirmación', 'finalizado')
    
    # 1. OBTENER Y VALIDAR DATOS DE LA SOLICITUD
    try:
        data = request.get_json()
        pregunta = data.get('question')
        user_id = data.get('user_id') 
        max_histories = data.get('max_histories', 10)
        name_space = data.get('name_space', 'real') 
        json_gpt = data.get('json_gpt')
        index_name = data.get('index')

        # Validar campos requeridos
        if not pregunta or not user_id or not index_name:
            return jsonify(response="La pregunta, el ID de usuario y el nombre del índice son requeridos."), 400
            
        user_id_int = int(user_id)  # Convertir a entero para consistencia
    
    except Exception as e:
        return jsonify(response=f"Error procesando datos de entrada: {str(e)}"), 400
    
    # 2. CONFIGURACIÓN INICIAL Y CONEXIONES
    try:
        # Inicializar conexión con Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY_PRUEBAS)
        index = pc.Index(data.get('index'))  # Usar índice proporcionado
        
        
        # Inicializar embeddings de OpenAI
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # 3. BUSCAR CONTEXTO RELEVANTE EN PINECONE
        # 3.1. Buscar documentos relevantes al namespace principal
        # Se aumentó top_k de 1 a 5 en la consulta a Pinecone para mejorar la robustez del sistema, 
        # permitiendo recuperar múltiples resultados semánticamente relevantes y aumentar la probabilidad de respuestas precisas.
        query_vector = embeddings.embed_query(pregunta)
        result = index.query(
            namespace=name_space,
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )

        docs = [match['metadata']['text'] for match in result.get('matches', []) if 'metadata' in match and 'text' in match['metadata']]
        #Uso de .get() para acceder a 'matches' de forma segura, evita un KeyError si 'matches' no está presente en la respuesta de Pinecone.

        # 3.2. Buscar historial previo del usuario
        prompt_history = index.query(
            namespace="user_history",
            id=str(user_id),
            top_k=5,
            include_metadata=True
        )
        user_history = [match['metadata']['text'] for match in prompt_history['matches'] if 'metadata' in match]
        user_history_string = ''.join(user_history)
        
        # Verificar si es el primer mensaje del usuario
        primeraRespuesta = not bool(user_history_string)
        
        # 3.3. Buscar base de conocimiento
        files_upload = index.query(
            namespace="file",
            vector=query_vector,
            top_k=10,
            include_metadata=True
        )
        
        # Estructura la base de conocimientos con todos los archivos relevantes
        base_conocimientos = [
            match['metadata']['text']
            for match in files_upload.get('matches', [])
            if 'metadata' in match
        ]
        
        # 4. PREPARAR LOS DOCUMENTOS PARA EL MODELO
        input_documents = (
            [Document(page_content=text) for text in docs] +    # Documentos relevantes
            [Document(page_content=text) for text in user_history] +  # Historial de usuario
            [Document(page_content=text) for text in base_conocimientos]  # Base de conocimiento general basada en archivos cargados
        )
        
        # 5. CONSTRUIR EL PROMPT COMPLETO
        if isinstance(json_gpt, dict) and json_gpt:
            # Si hay un JSON, construir prompt estructurado
            full_prompt = f"""
                ### IMPORTANTE:
                ***Si hay campos vacíos***, pregunta uno a la vez, en orden, y no pases al siguiente hasta que el actual esté respondido.

                JSON actual:
                {json.dumps(json_gpt, indent=2)}

                Mensaje del usuario:
                \"\"\"{pregunta}\"\"\"

                Devuelve tu respuesta en el siguiente formato:

                JSON:
                <JSON actualizado aquí>

                RESPUESTA:
                <Respuesta conversacional aquí>

                RESUMEN:
                <Resumen amigable del JSON en formato lista>

                ESTADO:
                <palabra clave aquí: "pendiente" o "finalizado">

                # Estados posibles:
                # "pendiente" → si aún falta información.
                # "finalizado" → si el usuario confirma que ya no desea cambiar nada más.
            """
            # "esperando confirmación" → si ya se completó todo y estás preguntando si desea modificar algo.
            # <palabra clave aquí: "pendiente", "esperando confirmación" o "finalizado">
        else:
            full_prompt = pregunta  # Si no hay JSON, usar la pregunta directamente
            
        # 6. EJECUTAR EL MODELO DE LENGUAJE
        llm = ChatOpenAI(model_name='gpt-4o-mini', openai_api_key=OPENAI_API_KEY, temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=input_documents, question=full_prompt)
        
        # FULL PROMPT: Mostrar respuesta cruda
        print('*****************************************')
        print(full_prompt)
        print('*****************************************')
        
        # Debug: Mostrar respuesta cruda
        print('*****************************************')
        print(respuesta)
        print('*****************************************')
        
        # 7. PROCESAR LA RESPUESTA DEL MODELO
        if isinstance(json_gpt, dict) and json_gpt:
            # Extraer componentes de la respuesta estructurada
            # 7.1. Extraer texto inicial (si existe)
            match_texto = re.search(r'(.*?)(?=JSON:\s*\{.*?\}\s*RESPUESTA:)', respuesta, re.DOTALL)
            respuesta_texto = match_texto.group(1).strip() if match_texto else ""
            
            # 7.2. Extraer JSON, respuesta conversacional y estado
            match = re.search(
                r'JSON:\s*(\{.*?\})\s*RESPUESTA:\s*(.*?)\s*RESUMEN:\s*(.*?)(?=\n+ESTADO:)',
                respuesta,
                re.DOTALL
            )
            
            if match:
                json_str = match.group(1)
                json_completado_raw = json.loads(json_str)
                
                # Si no había texto inicial, usar el conversacional
                if not respuesta_texto:
                    respuesta_texto = match.group(2).strip()
                else:
                    respuesta_texto = f"{respuesta_texto}\n\n{match.group(2).strip()}"
            
                # Si es primera vez agregar resumen
                if primeraRespuesta:
                    respuesta_texto = f"**Resumen:**\n\n {match.group(3).strip()}\n\n{respuesta_texto}"
                
                # Extraer estado de la conversación
                match_estado = re.search(r'ESTADO:\s*"?([^"\n]+)"?', respuesta)
                estado_conversacion = match_estado.group(1).strip() if match_estado else "desconocido"
        else:
            # Cuando no hay JSON involucrado
            json_completado_raw = False
            respuesta_texto = respuesta
            estado_conversacion = False
            
        # 8. ACTUALIZAR HISTORIAL DE CONVERSACIÓN
        # 8.1. Construir nuevo historial
        userHistory = (f"{user_history_string} - Pregunta: {pregunta} - Respuesta:{respuesta}") # AQUI SE ESTA CONCATENANDO MUCHA HISTORIA Y ESTO SOBREPASA EL LIMITE DE 40960 BYTES POR CADA VECTOR EN PINECONE, Y A MEDIDA QUE AVANZA LA CONVERSACION SE HACE MAS EXTENSA
        count = userHistory.count("- Respuesta:")
        
        # 8.2. Limitar historial según max_histories
        if count > max_histories:
            patron = re.compile(r"Historial de conversacion:(.*?- Respuesta:.*? - Pregunta:.*?)- Respuesta:", re.DOTALL)
            userHistory = re.sub(patron, "Historial de conversacion:\n-Respuesta:", userHistory, 1)
            print(f"Historial acortado por max_histories. Nuevo tamaño (caracteres): {len(userHistory)}", flush=True)

        # 8.3. Actualizar vector en Pinecone
        #instructions_values = embeddings.embed_query(userHistory)
        #existing_new_vector = index.fetch(ids=[user_id], namespace="user_history")
        #current_datetime = datetime.now().isoformat()

        # 8.3. Preparar historial para metadatos (userHistory_for_metadata) y truncar si excede el límite de bytes.
        userHistory_for_metadata = userHistory # Este es el que se guardará en metadata['text']

        MAX_METADATA_BYTES = 40000  # Límite de Pinecone es 40960 bytes, damos un margen de seguridad.
        userHistory_bytes_check = userHistory_for_metadata.encode('utf-8', errors='ignore')

        if len(userHistory_bytes_check) > MAX_METADATA_BYTES:
            # Si excede el límite, truncar desde el principio para mantener lo más reciente.
            # Tomamos los últimos MAX_METADATA_BYTES bytes del historial codificado.
            bytes_to_keep_for_metadata = userHistory_bytes_check[-MAX_METADATA_BYTES:]
            # Decodificar de nuevo a string. 'errors='replace'' maneja errores si se corta un caracter multibyte,
            # reemplazándolo con el carácter �.
            userHistory_for_metadata = bytes_to_keep_for_metadata.decode('utf-8', errors='replace') 
            
            print(f"ADVERTENCIA: Historial para metadatos fue truncado debido al límite de tamaño de Pinecone.", flush=True)
            print(f"           Tamaño original (bytes): {len(userHistory_bytes_check)}, Tamaño truncado para metadata (bytes): {len(userHistory_for_metadata.encode('utf-8', errors='ignore'))}", flush=True)
        
        # if user_id not in existing_new_vector['vectors']:
        #     # Crear nuevo vector de historial
        #     index.upsert(
        #         vectors=[{
        #             "id": user_id,
        #             "values": instructions_values,
        #             "metadata": {
        #                 "text": "Historial de conversacion:\n" + userHistory,
        #                 "date": current_datetime
        #             }
        #         }],
        #         namespace="user_history"
        #     )
        # else:
        #     # Actualizar vector existente
        #     index.delete(ids=user_id, namespace="user_history")
        #     index.upsert( #CUANDO SE INTENTA SUBIR VECTOR CON METADATOS GRANDES, PINECONE RECHAZA LA SOLICITUD 
        #         vectors=[{
        #             "id": user_id,
        #             "values": instructions_values,
        #             "metadata": {
        #                 "text": userHistory, #AQUI DA EL ERROR
        #                 "date": current_datetime
        #             }
        #         }],
        #         namespace="user_history"
        #     )
        # 8.4. Actualizar vector en Pinecone
        # El embedding se genera a partir de 'userHistory' (que es el historial completo o limitado por max_histories).
        instructions_values = embeddings.embed_query(userHistory) 
        
        # 'user_id' ya es un entero debido a user_id_int. Para Pinecone, los IDs de vector deben ser strings.
        user_id_pinecone_str = str(user_id) 
        
        # No es necesario hacer fetch ni delete antes del upsert.
        # Pinecone 'upsert' actualiza un vector si el ID existe, o lo crea si no existe.
        current_datetime = datetime.now().isoformat()
        
        metadata_payload_for_history = {
            "text": userHistory_for_metadata,
            "date": current_datetime
        }

        # Upsert directamente.
        index.upsert(
            vectors=[{
                "id": user_id_pinecone_str, # ID del vector es el user_id (convertido a string).
                "values": instructions_values,
                "metadata": metadata_payload_for_history
            }],
            namespace="user_history"
        )
        # 9. CONSTRUIR RESPUESTA FINAL
        respuestaIA = {
            "respuesta": respuesta_texto,
            "json_gpt": json_completado_raw,
            "estado_conversacion": estado_conversacion,
        }
        
        return jsonify(respuestaIA), 200
    
    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
        print("!!!!! ERROR 500 DETECTADO EN /api/chatbot !!!!!", flush=True)
        print(f"!!!!! Tipo de Excepción: {type(e).__name__}", flush=True)
        print(f"!!!!! Mensaje de Error: {str(e)}", flush=True)
        print("!!!!! TRAZA DE ERROR COMPLETA:", flush=True)
        
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush() 
        
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)


        exc_type, exc_obj, tb = sys.exc_info()
        line_number = tb.tb_lineno if tb else "N/A"
        filename = tb.tb_frame.f_code.co_filename if tb and tb.tb_frame else "N/A"
        
        error_response_data = {
            "response": f"Ocurrió un error INTERNO en el servidor: {str(e)}",
            "archivo_error": filename,
            "linea_error": line_number,
            "tipo_error": str(exc_type.__name__ if exc_type else "N/A"),
            "mensaje_debug": "Revisar los logs del servidor Docker para la traza completa."
        }
        return jsonify(error_response_data), 500
    #except Exception as e:
        # Manejo detallado de errores
        #exc_type, exc_obj, tb = sys.exc_info()
        #line_number = tb.tb_lineno
        #filename = tb.tb_frame.f_code.co_filename
        #return jsonify({
            #"response": f"Ocurrió un error: {str(e)}",
            #"archivo": filename,
            #"linea": line_number,
            #"tipo": str(exc_type.__name__)
        #}), 500
    

@app.route('/api/chatbot/v2', methods=['POST'])
def chatbot_v2():
    """   
    Parámetros esperados en el JSON de la solicitud:
    - question: La pregunta del usuario (requerido)
    - user_id: ID del usuario (requerido)
    - name_space: Namespace de Pinecone (requerido, e.g., el pdf_id)
    - index: Nombre del índice Pinecone a usar (requerido)
    """
    
    # 1. OBTENER Y VALIDAR DATOS DE LA SOLICITUD
    try:
        data = request.get_json()
        pregunta_original = data.get('question')
        user_id = data.get('user_id') 
        name_space = data.get('name_space') 
        index_name = data.get('index')      
        
        top_k = 5 

        if not all([pregunta_original, user_id, name_space, index_name]):
            return jsonify(response="Los campos 'question', 'user_id', 'name_space', y 'index' son requeridos."), 400
            
    except Exception as e:
        print(f"Error procesando datos de entrada para chatbot_v2: {traceback.format_exc()}", flush=True)
        return jsonify(response=f"Error procesando datos de entrada: {str(e)}"), 400
    
    # 2. CONFIGURACIÓN INICIAL Y CONEXIONES
    try:
        index_list_response = pc.list_indexes()
        current_index_names = [idx_model.name for idx_model in index_list_response.indexes]
        if index_name not in current_index_names:
             return jsonify(error=f"El índice '{index_name}' no existe en Pinecone."), 404
        index_instance = pc.Index(index_name)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # 3. BUSCAR CONTEXTO RELEVANTE EN PINECONE
        print(f"Buscando en Pinecone - Index: {index_name}, Namespace: {name_space}, TopK: {top_k}", flush=True)
        print(f"Pregunta Original para embedding: {pregunta_original}", flush=True)
        
        query_vector = embeddings.embed_query(pregunta_original)
        
        search_results = index_instance.query(
            namespace=name_space,
            vector=query_vector,
            top_k=top_k, 
            include_metadata=True
        )
        
        retrieved_docs_text = [
            match['metadata']['text'] 
            for match in search_results.get('matches', []) 
            if 'metadata' in match and 'text' in match['metadata']
        ]

        if not retrieved_docs_text:
            print(f"No se encontraron documentos relevantes en Pinecone.", flush=True)
        else:
            print(f"Documentos recuperados de Pinecone: {len(retrieved_docs_text)}", flush=True)

        # 4. PREPARAR CONTEXTO Y PROMPT PARA LLM
        context_string = "\n\n---\n\n".join(retrieved_docs_text)
        
        if not context_string.strip():
            context_string = "No se encontró información contextual relevante en la base de datos de documentos para esta pregunta."

        optimized_prompt = f"""Eres un asistente de IA altamente competente especializado en responder preguntas basándote ÚNICAMENTE en el CONTEXTO DEL DOCUMENTO proporcionado.
        Tu tarea es analizar la PREGUNTA DEL USUARIO y el CONTEXTO DEL DOCUMENTO. El contexto puede contener varios fragmentos de información recuperados.

        Instrucciones Importantes:
        1.  Examina cuidadosamente TODO el CONTEXTO DEL DOCUMENTO para encontrar la información más relevante para responder la PREGUNTA DEL USUARIO.
        2.  Sintetiza una respuesta coherente y concisa utilizando solo la información del CONTEXTO DEL DOCUMENTO.
        3.  Si la información necesaria para responder completamente la pregunta no se encuentra en el CONTEXTO DEL DOCUMENTO, debes declararlo explícitamente. Por ejemplo, di: "Basado en la información proporcionada en el documento, no puedo responder [parte específica de la pregunta] porque no se encontró información al respecto."."
        4.  NO inventes información, no hagas suposiciones fuera del texto proporcionado y NO utilices conocimiento externo. Tu conocimiento se limita estrictamente al CONTEXTO DEL DOCUMENTO.
        5.  Si el contexto parece vacío o dice que no se encontró información, refleja eso en tu respuesta.

        CONTEXTO DEL DOCUMENTO (puede contener varios fragmentos recuperados, usa los más relevantes):
        \"\"\"
        {context_string}
        \"\"\"

        PREGUNTA DEL USUARIO:
        \"\"\"
        {pregunta_original}
        \"\"\"

        Respuesta Detallada y Concisa (basada únicamente en el CONTEXTO DEL DOCUMENTO anterior):
        """
        
        print("----- CHATBOT_V2: PROMPT PARA LLM -----", flush=True)
        print(f"Prompt enviado al LLM (sin el contexto detallado para brevedad del log): PREGUNTA='{pregunta_original}', Num_Context_Chunks={len(retrieved_docs_text)}", flush=True)
        print("----- FIN PROMPT PARA LLM -----", flush=True)
        
        # 5. EJECUTAR EL MODELO DE LENGUAJE
        llm = ChatOpenAI(model_name='gpt-4o', openai_api_key=OPENAI_API_KEY, temperature=0.0) 
        
        llm_response = llm.invoke(optimized_prompt)
        respuesta_llm = llm_response.content

        print(f"CHATBOT_V2: Respuesta cruda del LLM: {respuesta_llm[:300]}...", flush=True)
        
        # 6. CONSTRUIR RESPUESTA FINAL
        api_response = {
            "respuesta": respuesta_llm.strip(),
            "documentos_recuperados": len(retrieved_docs_text),
            "top_k_configurado": top_k
        }
        
        return jsonify(api_response), 200
    
    except openai.APIError as e_openai:
        print(f"OpenAI API Error en chatbot_v2: {traceback.format_exc()}", flush=True)
        return jsonify(response=f"Error de API OpenAI: {str(e_openai)}"), 500
    except Exception as e:
        print(f"Error general en chatbot_v2: {traceback.format_exc()}", flush=True)
        exc_type, exc_obj, tb = sys.exc_info()
        line_number = tb.tb_lineno if tb else "N/A"
        filename = tb.tb_frame.f_code.co_filename if tb and tb.tb_frame else "N/A"
        return jsonify({
            "response": f"Ocurrió un error en chatbot_v2: {str(e)}",
            "archivo": filename,
            "linea": line_number,
            "tipo": str(exc_type.__name__ if exc_type else "N/A")
        }), 500

@app.route('/api/deleteHistory', methods=['DELETE'])
def delete_history():
    try:
        data = request.get_json()
        if not data or 'index' not in data:
            return jsonify(response="Datos de entrada inválidos."), 400

        pc = Pinecone(api_key=PINECONE_API_KEY_PRUEBAS)
        index = pc.Index(data.get('index'))
        index.delete(delete_all=True, namespace='user_history')
        return jsonify(response="Historial eliminado correctamente."), 200

    except openai.error.AuthenticationError:
        return jsonify(response="La API key no es válida."), 401
    except Exception as e:
        return jsonify(response=f"Ocurrió un error: {str(e)}"), 500

@app.route('/api/upsertFile', methods=['POST'])
def upsert_file():
    data = request.get_json()
    id_vector = data.get('id_vector')
    file_url_id = data.get('link_file_id') 
    file_url = data.get('link_file')
    type_file = data.get('type_file') 
    name_space = data.get('name_space')  
    index_name = data.get('index')

    if not index_name or not file_url or not name_space or not type_file:
        return jsonify(response="Se requiere de la siguiente información (link_file, type_file, name_space, index)."), 400
    if not file_url_id:
         return jsonify(response="El parámetro 'link_file_id' también es requerido."), 400

    try:
        print(f"Descargando archivo desde: {file_url}", flush=True)
        
        response = requests.get(file_url, verify=False)
        response.raise_for_status()
        file_bytes = response.content 
        
        text = "" 

        if type_file == "pdf":
            print("Procesando archivo PDF.", flush=True)
            pdf_doc_for_check = fitz.open(stream=file_bytes, filetype="pdf")
            has_images = False
            if len(pdf_doc_for_check) > 0:
                for page_num_check in range(min(len(pdf_doc_for_check), 5)):
                    if pdf_doc_for_check.load_page(page_num_check).get_images(full=True):
                        has_images = True
                        break
            
            if not has_images:
                print("PDF sin imágenes detectadas", flush=True)
                pdf_file_obj_for_reader = BytesIO(file_bytes)
                reader = PdfReader(pdf_file_obj_for_reader)
                temp_text_list_pypdf = []
                for page_obj in reader.pages:
                    extracted_page_content = page_obj.extract_text()
                    if extracted_page_content:
                        temp_text_list_pypdf.append(extracted_page_content)
                text = "\n".join(temp_text_list_pypdf)
                pdf_doc_for_check.close()
            else:
                print("PDF con imágenes detectadas", flush=True)
                all_extracted_content_parts_pdf = []
                for page_num in range(len(pdf_doc_for_check)):
                    current_fitz_page_obj = pdf_doc_for_check.load_page(page_num)
                    display_page_number = page_num + 1
                    text_from_fitz_page_content = current_fitz_page_obj.get_text("text")
                    if text_from_fitz_page_content.strip():
                        all_extracted_content_parts_pdf.append(f"Texto de Página {display_page_number}:\n{text_from_fitz_page_content.strip()}")

                    images_on_current_page = current_fitz_page_obj.get_images(full=True)
                    for img_idx, img_info_data_fitz in enumerate(images_on_current_page):
                        display_img_index_num = img_idx + 1
                        img_xref_from_data = img_info_data_fitz[0]
                        try:
                            base_image_from_pdf = pdf_doc_for_check.extract_image(img_xref_from_data)
                            image_bytes_for_processing_now = base_image_from_pdf["image"]
                            print(f"Procesando Imagen {display_img_index_num} de la Página {display_page_number}...", flush=True)
                            ai_description_from_openai = get_image_description_openai(image_bytes_for_processing_now, max_tokens=70)
                            ocr_text_from_image_tess = extract_text_from_image_tesseract(image_bytes_for_processing_now)
                            image_summary_text_block = (
                                f"Resumen de Contenido de Imagen {display_img_index_num} (ubicada en la Página {display_page_number}):\n"
                                f"Descripción Visual: {ai_description_from_openai}\n"
                                f"Texto Extraído de Imagen: {ocr_text_from_image_tess if ocr_text_from_image_tess and 'Error en OCR' not in ocr_text_from_image_tess else 'No se extrajo texto o hubo un error.'}"
                            )
                            all_extracted_content_parts_pdf.append(image_summary_text_block)
                        except Exception as e_img_processing_detail_err:
                            print(f"Error procesando imagen XREF {img_xref_from_data} en la página {display_page_number}: {e_img_processing_detail_err}", flush=True)
                            all_extracted_content_parts_pdf.append(f"Error al procesar información de Imagen {display_img_index_num} en la Página {display_page_number}.")
                pdf_doc_for_check.close()
                text = "\n\n---\n\n".join(all_extracted_content_parts_pdf)
        
        elif type_file == "docx":
            file_stream = BytesIO(file_bytes) 
            result = mammoth.extract_raw_text(file_stream)
            if result.value is not None:
                text += result.value
            else:
                text += "No se pudo extraer texto del documento DOCX"
        elif type_file == "doc":
            file_stream = BytesIO(file_bytes)
            result = mammoth.extract_raw_text(file_stream)
            if result.value is not None:
                text += result.value
            else:
                text += "No se pudo extraer texto del documento DOC"
        elif type_file == "txt":
            text = file_bytes.decode('utf-8', errors='ignore') # Decodificar de txt a bytes directamente
        elif type_file in ["xls", "xlsx"]:
            excel_file = BytesIO(file_bytes)
            if type_file == "xls":
                df = pd.read_excel(excel_file, engine='xlrd')
            else:
                df = pd.read_excel(excel_file, engine='openpyxl')
            text = df.to_string(index=False)
        else:
            return jsonify(response="Tipo de archivo no soportado. Use pdf, docx, txt, xls o xlsx."), 400

        if not text or not text.strip():
            return jsonify(response="No se pudo extraer texto del archivo."), 400
        
        # --- Lógica de Pinecone con adaptación para chunking ---
        pc = Pinecone(api_key=PINECONE_API_KEY_PRUEBAS)
        index_list_resp_pinecone = pc.list_indexes()
        current_pinecone_indices_names = [idx.name for idx in index_list_resp_pinecone.indexes]
        if index_name not in current_pinecone_indices_names:
            pc.create_index(name=index_name, dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

        # 2. OBTENCIÓN DE INSTANCIA DEL ÍNDICE
        index_pinecone_instance = pc.Index(index_name) 
        embeddings_openai_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) 

        # 3. PREPARACIÓN DEL CONTENIDO COMPLETO
        #    'values_intructions_full_content' es tu 'values_intructions' original.
        values_intructions_full_content = f"""
            Contenido del archivo ({type_file}):
            {text} 
        """
        #    'text' aquí es el 'processed_text_for_pinecone' que ya incluye texto y resúmenes de imágenes si el archivo era un PDF con imágenes.

        # 4. DEFINICIONES PARA CHUNKING Y METADATOS
        MAX_METADATA_BYTES_FOR_PINECONE = 39000 # Límite práctico para el campo 'text' en metadatos de Pinecone.
        vectors_for_pinecone_upsert = [] # Lista para acumular los vectores a subir.
        base_pinecone_vector_id = "file" + str(file_url_id) # es 'instructions_id' original. Se usará como base para IDs de chunks.

        # 5. ESTRATEGIA DE ACTUALIZACIÓN 
        #    OBJETIVO: Cuando se actualiza un archivo (mismo 'file_url_id'), queremos reemplazar su contenido anterior en Pinecone.
        #    PROBLEMA:
        #        - Si antes el archivo se guardó como 1 solo vector y ahora se va a guardar como chunks.
        #        - Si antes se guardó como N chunks y ahora se va a guardar como M chunks (o como 1 solo vector).
        #    SOLUCIÓN INTRODUCIDA:
        #        - Se intenta borrar el vector que tendría el ID base ('file' + file_url_id). Esto cubre el caso
        #          donde el archivo se guardó previamente como un solo vector y ahora se quiere actualizar (ya sea
        #          como un solo vector de nuevo o como chunks).
        #        - SI EL ARCHIVO SE GUARDÓ ANTES COMO CHUNKS: Esta simple lógica de borrado NO eliminará
        #          esos chunks antiguos (que tendrían IDs como 'file..._chunk_0', 'file..._chunk_1', etc.).
        #          Eliminar chunks antiguos de forma selectiva es más complejo y requeriría:
        #            a) Almacenar los IDs de todos los chunks asociados a un 'file_url_id' en algún lado, o
        #            b) Usar la funcionalidad de 'delete by metadata filter' de Pinecone si los chunks tienen un metadato común
        #               como 'file_url_id_param' (que se añadió en el nuevo código).
        #               Ejemplo: index_pinecone_instance.delete(filter={"file_url_id_param": str(file_url_id)}, namespace=name_space)
        #            c) O borrar por prefijo de ID si la API de Pinecone lo permitiera directamente (no es común).
        try:
            print(f"Intentando borrar datos antiguos para file_url_id '{file_url_id}' (ID base: {base_pinecone_vector_id}) en namespace '{name_space}'.", flush=True)
            index_pinecone_instance.delete(ids=[base_pinecone_vector_id], namespace=name_space)
            index_pinecone_instance.delete(filter={"file_url_id_param": str(file_url_id)}, namespace=name_space)
            # Para una limpieza más completa de chunks si 'file_url_id_param' se usa consistentemente en metadatos:
            # index_pinecone_instance.delete(filter={"file_url_id_param": str(file_url_id)}, namespace=name_space)
            # Esto borraría TODOS los vectores (chunks o no) que tengan ese file_url_id_param.
            print(f"Intento de borrado de datos antiguos para '{file_url_id}' completado en namespace '{name_space}'.", flush=True)
        except Exception as e_delete_old:
            error_message_lower = str(e_delete_old).lower()
            if "namespace not found" in error_message_lower or "namespace does not exist" in error_message_lower:
                print(f"Advertencia al intentar borrar datos antiguos: Namespace '{name_space}' probablemente aún no existe o no contenía datos para '{file_url_id}'. Error: {e_delete_old}", flush=True)
            else:
                print(f"Error inesperado durante el borrado preventivo de datos antiguos: {e_delete_old}", flush=True)


        # 6. LÓGICA DE CHUNKING
        #    OBJETIVO: Pinecone tiene un límite en el tamaño de los metadatos (especialmente el campo 'text' y por exceder límites de metadatos.).
        #              Si el contenido extraído del archivo es muy grande, no cabrá en un solo vector.
        #        - El código original asumía que 'values_intructions' siempre cabría. No había chunking.
        #        - El nuevo código verifica el tamaño en bytes del contenido.
        if len(values_intructions_full_content.encode('utf-8')) <= MAX_METADATA_BYTES_FOR_PINECONE:
            # CASO 1: EL CONTENIDO ES PEQUEÑO Y CABE EN UN SOLO VECTOR 
            #    - Se crea un solo vector con 'base_pinecone_vector_id'.
            #    - Los metadatos incluyen información para indicar que no es un chunk.
            print(f"Contenido cabe en un solo vector. ID: {base_pinecone_vector_id}", flush=True)
            embedding_for_single_vector = embeddings_openai_model.embed_query(values_intructions_full_content)
            vectors_for_pinecone_upsert.append({
                "id": base_pinecone_vector_id,
                "values": embedding_for_single_vector,
                "metadata": { 
                    "text": values_intructions_full_content,
                    "original_file_type": type_file,
                    "source_url": file_url,
                    "file_url_id_param": str(file_url_id), # Para filtrado/agrupación
                    "is_chunked": False, # Indica que este no es parte de un conjunto de chunks
                    "chunk_index": 0,    # Índice del chunk (0 para el único chunk)
                    "total_chunks": 1    # Número total de chunks (1 en este caso)
                }
            })
        else:
            # CASO 2: EL CONTENIDO ES DEMASIADO GRANDE, NECESITA CHUNKING
            #    - Se usa CharacterTextSplitter de Langchain para dividir el texto.
            #    - CADA CHUNK SE CONVIERTE EN UN VECTOR SEPARADO EN PINECONE.
            #    - Cada chunk tendrá un ID único: base_id + "_chunk_" + índice_del_chunk.
            #    - Los metadatos de cada chunk incluyen información sobre su origen y su posición en la secuencia.
            print(f"Contenido excede límite ({MAX_METADATA_BYTES_FOR_PINECONE} bytes). Dividiendo en chunks para ID base: {base_pinecone_vector_id}", flush=True)
            text_splitter_for_chunks = CharacterTextSplitter(
                separator="\n\n---\n\n", # Un separador que podrías usar en tu `processed_text_for_pinecone`
                chunk_size=15000,       # Tamaño del chunk en CARACTERES. Es una heurística para no exceder los bytes.
                                        # Ajusta este valor según tus pruebas.
                chunk_overlap=200,      # Superposición entre chunks para mantener contexto.
                length_function=len
            )
            
            text_chunks_from_splitter = text_splitter_for_chunks.split_text(values_intructions_full_content)
            print(f"Generados {len(text_chunks_from_splitter)} chunks.", flush=True)

            for i, single_chunk_text in enumerate(text_chunks_from_splitter):
                chunk_pinecone_id = f"{base_pinecone_vector_id}_chunk_{i}" # ID único para el chunk
                
                # Salvaguarda: truncar el texto del chunk si excede el límite de metadatos de Pinecone
                # Esto es para el campo 'text' de los metadatos, no para el texto que se embedde.
                # El texto original del chunk se usa para el embedding.
                final_text_for_chunk_metadata = single_chunk_text
                if len(single_chunk_text.encode('utf-8')) > MAX_METADATA_BYTES_FOR_PINECONE:
                    print(f"Chunk {i} todavía es demasiado grande ({len(single_chunk_text.encode('utf-8'))} bytes) para metadatos después de dividir. Será truncado para metadatos.", flush=True)
                    # Lógica de truncado cuidadoso para no cortar en medio de un carácter multibyte UTF-8
                    encoded_single_chunk = single_chunk_text.encode('utf-8')
                    truncated_bytes_of_chunk = encoded_single_chunk[:MAX_METADATA_BYTES_FOR_PINECONE]
                    temp_truncated_chunk_text = ""
                    # Intenta decodificar desde el final para encontrar un punto válido
                    for k_idx_chunk in range(len(truncated_bytes_of_chunk), 0, -1):
                        try:
                            temp_truncated_chunk_text = truncated_bytes_of_chunk[:k_idx_chunk].decode('utf-8')
                            break
                        except UnicodeDecodeError: 
                            continue # Sigue intentando con un byte menos
                    final_text_for_chunk_metadata = (temp_truncated_chunk_text + "\n... (Chunk truncado para metadatos)") if temp_truncated_chunk_text else "Chunk truncado por tamaño para metadatos."

                embedding_for_chunk = embeddings_openai_model.embed_query(single_chunk_text) # Embeber el chunk original completo.
                vectors_for_pinecone_upsert.append({
                    "id": chunk_pinecone_id,
                    "values": embedding_for_chunk,
                    "metadata": {
                        "text": final_text_for_chunk_metadata, # Texto del chunk (potencialmente truncado para metadatos)
                        "original_file_type": type_file,
                        "source_url": file_url,
                        "file_url_id_param": str(file_url_id), # ID base para agrupar chunks
                        "is_chunked": True,                    # Indica que es parte de un conjunto de chunks
                        "chunk_index": i,                      # Índice de este chunk
                        "total_chunks": len(text_chunks_from_splitter) # Número total de chunks para este archivo
                    }
                })

        # 7. UPSERT DE VECTORES A PINECONE (MODIFICADO PARA MANEJAR UNO O VARIOS VECTORES)
        #    - El código original siempre subía un solo vector.
        #    - El nuevo código sube la lista 'vectors_for_pinecone_upsert', que puede contener
        #      un solo vector (si el contenido era pequeño) o múltiples vectores (si se hizo chunking).
        if vectors_for_pinecone_upsert:
            if len(vectors_for_pinecone_upsert) > 100:
                    print(f"ADVERTENCIA - Intentando subir {len(vectors_for_pinecone_upsert)} vectores. Pinecone recomienda lotes de <=100. Implementar batching si esto es común.", flush=True)
            
            index_pinecone_instance.upsert(vectors=vectors_for_pinecone_upsert, namespace=name_space)
            print(f"{len(vectors_for_pinecone_upsert)} vector(es) subido(s) a Pinecone.", flush=True)
        else:
            print("No se generaron vectores para subir (esto no debería ocurrir si 'text' tiene contenido).", flush=True)
        return jsonify(response=f"Información ingresada con éxito. Se crearon {len(vectors_for_pinecone_upsert)} vector(es)."), 200

    except Exception as e:
        print(f"---- DETAILED ERROR  ----", flush=True)
        print(f"Exception Type: {type(e).__name__}", flush=True)
        print(f"Exception Args: {e.args}", flush=True)
        print(f"Full Traceback within upsertFile:\n{traceback.format_exc()}", flush=True)
        return jsonify(error=f"Error: {str(e)}", traceback=traceback.format_exc()), 500

@app.route('/api/deleteFile', methods=['DELETE'])
def delete_file():
    try:
        data = request.get_json()
        
        # Validar los datos de entrada
        if not data or 'index' not in data or 'file_id' not in data or 'namespace' not in data:
            return jsonify(response="Datos de entrada inválidos. Se requiere index, file_id y namespace."), 400

        # Configurar Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY_PRUEBAS)
        index = pc.Index(data.get('index'))

        # Eliminar el archivo específico dentro del namespace
        index.delete(ids=[data.get('file_id')], namespace=data.get('namespace'))

        return jsonify(response=f"Archivo {data.get('file_id')} eliminado correctamente en namespace {data.get('namespace')}.", status_code=200), 200

    except openai.error.AuthenticationError:
        return jsonify(response="La API key no es válida.", status_code=401), 401

    except Exception as e:
        return jsonify(response=f"Ocurrió un error: {str(e)}", status_code=500), 500

@app.route('/api/verifyFileDeletion', methods=['POST'])
def verify_file_deletion():
    """
    Verifica si un archivo ha sido eliminado de la base de datos vectorial (Pinecone).
    Se asume que un archivo eliminado no debería existir al ser consultado por su ID.

    Parámetros esperados en el JSON de la solicitud:
    - file_id: El ID del archivo a verificar (puede ser el ID base o un ID de chunk) (requerido)
    - namespace: El namespace de Pinecone donde se encuentra el archivo (requerido)
    - index: El nombre del índice de Pinecone a usar (requerido)

    Retorna:
    - JSON con `success: true` si el archivo no se encuentra, `false` si se encuentra o hay un error.
    """
    try:
        data = request.get_json()

        if not data or 'index' not in data or 'file_id' not in data or 'namespace' not in data:
            return jsonify(response="Datos de entrada inválidos. Se requiere 'index', 'file_id' y 'namespace'."), 400

        index_name = data.get('index')
        file_id_to_check = data.get('file_id')
        namespace = data.get('namespace')

        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY')) 

        index_list_response = pc.list_indexes()
        current_index_names = [idx_model.name for idx_model in index_list_response.indexes]
        if index_name not in current_index_names:
            return jsonify(success=False, message=f"El índice '{index_name}' no existe en Pinecone."), 404

        index_instance = pc.Index(index_name)

        # Intentar obtener el vector por su ID. Si no se encuentra, significa que fue eliminado.
        fetch_response = index_instance.fetch(ids=[file_id_to_check], namespace=namespace)

        if not fetch_response['vectors']:
            return jsonify(success=True, message=f"El archivo con ID '{file_id_to_check}' no se encontró en el namespace '{namespace}', lo que indica que fue eliminado o no existía."), 200
        else:
            return jsonify(success=False, message=f"El archivo con ID '{file_id_to_check}' todavía existe en el namespace '{namespace}'. La eliminación no fue verificada."), 409 # Conflict

    except openai.APIError as e_openai:
        print(f"OpenAI API Error en verifyFileDeletion (inesperado): {traceback.format_exc()}", flush=True)
        return jsonify(success=False, message=f"Error de API OpenAI inesperado: {str(e_openai)}"), 500
    except Exception as e:
        print(f"Error general en verifyFileDeletion: {traceback.format_exc()}", flush=True)
        exc_type, exc_obj, tb = sys.exc_info()
        line_number = tb.tb_lineno if tb else "N/A"
        filename = tb.tb_frame.f_code.co_filename if tb and tb.tb_frame else "N/A"
        return jsonify({
            "success": False,
            "message": f"Ocurrió un error inesperado al verificar la eliminación: {str(e)}",
            "archivo": filename,
            "linea": line_number,
            "tipo": str(exc_type.__name__ if exc_type else "N/A")
        }), 500

def extract_text(content, file_type):
    """ Extrae el texto del archivo según su tipo. """
    text = ""

    if file_type == "pdf":
        pdf_file = BytesIO(content)
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()

    elif file_type in ["docx", "doc"]:
        file_stream = BytesIO(content)
        result = mammoth.extract_raw_text(file_stream)
        text = result.value if result.value else f"No se pudo extraer texto del documento {file_type.upper()}"

    elif file_type == "txt":
        text = content.decode("utf-8")

    elif file_type in ["xls", "xlsx"]:
        excel_file = BytesIO(content)
        df = pd.read_excel(excel_file, engine="openpyxl" if file_type == "xlsx" else "xlrd")
        text = df.to_string(index=False)

    return text

def extract_text_from_image_tesseract(image_bytes):
    """Función  para extraer texto (en español e inglés) de una imagen usando Tesseract OCR. Toma los bytes de una imagen, intenta leer el texto y lo devuelve."""
    try:
        img = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, lang='spa+eng')
        if not text.strip():
            return "No se detectó texto en la imagen con Tesseract."
        return text
    except pytesseract.TesseractNotFoundError: # Error específico si Tesseract no se encuentra.
        print("Error: Tesseract no está instalado o no se encuentra en el PATH.", flush=True)
        print("Por favor, instale Tesseract OCR y asegúrese de que esté en el PATH del sistema.", flush=True)
        print("O configure pytesseract.pytesseract.tesseract_cmd con la ruta al ejecutable.", flush=True)
        return "Error en OCR: Tesseract no encontrado. Verifique la instalación."
    except Exception as e: # Captura cualquier otro error durante el OCR.
        print(f"Error durante OCR con Tesseract: {e}", flush=True)
        return f"Error en OCR (Tesseract): {str(e)}"
    
def get_image_description_openai(image_bytes, max_tokens=150):
    """Función para obtener una descripción de una imagen usando la API GPT-4o (Vision) de OpenAI.
    Toma los bytes de una imagen, los envía a OpenAI y devuelve la descripción generada."""
    global client # Usa el cliente de OpenAI inicializado globalmente
    if not client:
        print("OpenAI client not initialized. Skipping image description.", flush=True)
        return "Descripción de imagen no disponible (cliente OpenAI no inicializado)."

    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }
        
        payload = {
            "model": "gpt-4o", 
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe esta imagen en detalle EN ESPAÑOL. ¿Cuáles son los elementos clave? Si es un diagrama o gráfico, explica su propósito. Si es un objeto, describe su apariencia y función, si es evidente."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}" 
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens
        }
        

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=payload["messages"],
            max_tokens=max_tokens
        )
        description = response.choices[0].message.content.strip()
        return description

    except openai.APIError as e: 
        print(f"OpenAI API error while getting image description: {e}", flush=True)
        return f"Error de API OpenAI al describir imagen: {str(e)}"
    except Exception as e:
        print(f"Unexpected error getting image description: {e}", flush=True)
        return f"Error inesperado al describir imagen: {str(e)}"
    
@app.route('/api/uploadPdfChat', methods=['POST'])
def upload_pdf_chat():
    data = request.get_json()
    file_url = data.get('file_url')
    index_name = data.get('index_name')
    pdf_id = data.get('pdf_id')

    if not all([file_url, index_name, pdf_id]):
        return jsonify(response="Información requerida: file_url, index_name, pdf_id."), 400

    try:
        # 1: Descarga el PDF desde la URL proporcionada
        print(f"Descargando PDF desde: {file_url}", flush=True)
        response = requests.get(file_url)
        response.raise_for_status()
        pdf_bytes = response.content
        print("PDF descargado.", flush=True)

        # 2: Verifica que el índice de Pinecone especificado exista
        index_list_response = pc.list_indexes()
        current_index_names = [index_model.name for index_model in index_list_response.indexes]

        if index_name not in current_index_names:
            print(f"Error: El índice de Pinecone '{index_name}' no existe.", flush=True)
            return jsonify(error=f"El índice de Pinecone '{index_name}' no existe."), 404

        index_instance = pc.Index(index_name)
        print(f"Usando índice de Pinecone existente: {index_name}", flush=True)

        embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        vectors_to_upsert = []
        
        # 3: Procesar el PDF página por página
        print("Procesando PDF con PyMuPDF...", flush=True)
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            page_content_parts_for_pinecone = [] 
            page_number_for_display = page_num + 1

            # 3.1: Extraer y procesar el texto de la página
            page_text_content = page.get_text("text")
            if page_text_content.strip():
                page_text_header = f"Contenido de texto de la Página {page_number_for_display} del PDF '{pdf_id}':\n"
                full_page_text_for_chunking = page_text_header + page_text_content.strip()
                
                text_chunks = text_splitter.split_text(full_page_text_for_chunking)
                for chunk_idx, chunk in enumerate(text_chunks):
                    vector_id = f"{pdf_id}_page_{page_number_for_display}_textchunk_{chunk_idx}"
                    vector_embedding = embeddings_model.embed_query(chunk)
                    metadata = {
                        "text": chunk, 
                        "pdf_id": pdf_id,
                        "page_number": page_number_for_display,
                        "source_type": "pdf_page_text"
                    }
                    vectors_to_upsert.append({
                        "id": vector_id,
                        "values": vector_embedding,
                        "metadata": metadata
                    })

            # 3.2: Extraer y procesar las imágenes de la página
            image_list = page.get_images(full=True)
            if image_list:
                print(f"Página {page_number_for_display}: Encontradas {len(image_list)} imágenes.", flush=True)
            
            for img_idx, img_info in enumerate(image_list):
                image_index_on_page_for_display = img_idx + 1
                xref = img_info[0]
                try:
                    base_image = pdf_document.extract_image(xref)
                    image_bytes_data = base_image["image"]

                    print(f"  Procesando Imagen {image_index_on_page_for_display} en Página {page_number_for_display} (XREF: {xref})...", flush=True)

                    #  Obtener descripción de la imagen generada por IA
                    print(f"    Obteniendo descripción AI para Imagen {image_index_on_page_for_display}...", flush=True)
                    ai_description = get_image_description_openai(image_bytes_data)
                    print(f"    Descripción AI: {ai_description[:100]}...", flush=True)

                    # Obtener texto de la imagen usando OCR
                    print(f"    Obteniendo texto OCR para Imagen {image_index_on_page_for_display}...", flush=True)
                    ocr_text = extract_text_from_image_tesseract(image_bytes_data) 
                    print(f"    Texto OCR: {ocr_text[:100].replace(chr(10), ' ')}...", flush=True)
                    
                    # Intentar obtener la posición de la imagen en la página con bounding box
                    img_rect = None
                    try:
                        for item in page.get_drawings(): 
                            if item["type"] == "image" and item.get("xref") == xref:
                                img_rect = item["rect"]
                                break
                        if not img_rect: 
                           rects = page.get_image_bbox(img_info, transform=True)
                           if rects: img_rect = rects

                    except Exception as e_bbox:
                        print(f"Advertencia: No se pudo obtener el bounding box para la imagen {xref} en página {page_number_for_display}: {e_bbox}", flush=True)
                        img_rect = "No disponible"


                    # Construir un texto combinado para esta imagen, incluyendo su referencia, la descripción AI, el texto OCR y su posición
                    image_combined_text = (
                        f"Detalles de referencia para la Imagen: {image_index_on_page_for_display} ubicada en la Página {page_number_for_display} del documento '{pdf_id}'.\n"
                        f"Descripción visual: {ai_description}\n"
                        f"Texto contenido dentro de la imagen (OCR): {ocr_text if ocr_text and 'Error en OCR' not in ocr_text else 'No se extrajo texto o hubo un error.'}"
                        f"Posición en página (bbox): {list(img_rect) if isinstance(img_rect, fitz.Rect) else img_rect}\n"
                    )
                    
                    # 3.2 Vectorizar el texto combinado de la imagen y preparar para Pinecone
                    image_vector_id = f"{pdf_id}_page_{page_number_for_display}_img_{image_index_on_page_for_display}"
                    image_vector_embedding = embeddings_model.embed_query(image_combined_text)
                    image_metadata = {
                        "text": image_combined_text,
                        "pdf_id": pdf_id,
                        "page_number": page_number_for_display,
                        "image_index_on_page": image_index_on_page_for_display,
                        "bounding_box": [round(c, 2) for c in img_rect] if isinstance(img_rect, fitz.Rect) else str(img_rect),
                        "ai_description_raw": ai_description,
                        "ocr_text_raw": ocr_text,
                        "source_type": "image_content"
                    }
                    vectors_to_upsert.append({
                        "id": image_vector_id,
                        "values": image_vector_embedding,
                        "metadata": image_metadata
                    })
                except Exception as e_img_proc:
                    print(f"Error procesando una imagen (XREF: {xref}) en la página {page_number_for_display}: {e_img_proc}", flush=True)

        pdf_document.close()
        total_vectors = len(vectors_to_upsert)
        print(f"Total de {total_vectors} vectores generados para el PDF {pdf_id}.", flush=True)

        # 4: Subir los vectores generados a Pinecone en lotes
        if vectors_to_upsert:
            batch_size = 100
            for i in range(0, total_vectors, batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                print(f"Subiendo batch {i // batch_size + 1} de { (total_vectors + batch_size -1) // batch_size } a Pinecone (namespace: {pdf_id})...", flush=True)
                index_instance.upsert(vectors=batch, namespace=pdf_id)
            print("Todos los vectores subidos a Pinecone.", flush=True)
            response_data = {
                "status": "OK",
                "message": f"El PDF con ID '{pdf_id}' ha sido procesado y su contenido está listo para consulta en el índice '{index_name}' (namespace: '{pdf_id}').",
                "document_id_for_query": pdf_id, 
                "pinecone_index_used": index_name,
                "pinecone_namespace_for_document": pdf_id,
                "total_vectors_created": total_vectors
            }
            print(f"Response data: {json.dumps(response_data, indent=2)}", flush=True)
            return jsonify(response_data), 200
        else:
            print(f"No se generaron vectores para el PDF {pdf_id} en el índice '{index_name}'.", flush=True)
            response_data = {
                "status": "OK_NO_CONTENT", 
                "message": f"El PDF con ID '{pdf_id}' fue procesado, pero no se extrajo contenido para vectorizar o no se generaron vectores. Está técnicamente 'listo', pero puede que no haya contenido para consultar en el índice '{index_name}' (namespace: '{pdf_id}').",
                "document_id_for_query": pdf_id,
                "pinecone_index_used": index_name,
                "pinecone_namespace_for_document": pdf_id,
                "total_vectors_created": 0
            }
            return jsonify(response_data), 200

    # Manejo de errores
    except requests.exceptions.RequestException as e_req:
        print(f"Error descargando PDF: {traceback.format_exc()}", flush=True)
        return jsonify(error=f"Error descargando el PDF: {str(e_req)}", traceback=traceback.format_exc()), 500
    except RuntimeError as e_runtime:
        error_str = str(e_runtime).lower()
        if "mupdf" in error_str or "fitz" in error_str or "cannot open" in error_str or "format error" in error_str:
            print(f"PyMuPDF RuntimeError in uploadPdfChat: {traceback.format_exc()}", flush=True)
            return jsonify(error=f"Error procesando el PDF con PyMuPDF: {str(e_runtime)}", traceback=traceback.format_exc()), 500
        else:
            print(f"Unhandled RuntimeError in uploadPdfChat: {traceback.format_exc()}", flush=True)
            return jsonify(error=f"Error inesperado (RuntimeError): {str(e_runtime)}", traceback=traceback.format_exc()), 500
    except openai.APIError as e_openai: 
        print(f"OpenAI API Error in uploadPdfChat: {traceback.format_exc()}", flush=True)
        return jsonify(error=f"Error de API OpenAI: {str(e_openai)}", traceback=traceback.format_exc()), 500
    except Exception as e:
        print(f"General Exception in uploadPdfChat: {traceback.format_exc()}", flush=True)
        return jsonify(error=f"Error inesperado: {str(e)}", traceback=traceback.format_exc()), 500

load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")


# --- Prompts ---
PROMPT_SYSTEM_SQL_GENERATOR = """
Eres un asistente de bases de datos experto en MySQL. Tu tarea es convertir preguntas en lenguaje natural en consultas SQL VÁLIDAS para MySQL.
RECUERDA: Genera SOLO la consulta SQL, sin texto adicional, ni comillas ```sql``` al inicio o final, ni explicaciones. La consulta NUNCA debe empezar por la palabra 'sql'.
La base de datos se llama 'crm_eva'.
Aquí está la estructura de la tabla 'menu' con la que trabajarás:

Tabla 'menu':
- id (INT, PRIMARY KEY, AUTO_INCREMENT)
- menu_id (INT, puede ser 0 si es un menú principal)
- name (VARCHAR)
- url (VARCHAR, puede ser '#' para menús no clickeables)
- orden (INT)
- icon (VARCHAR, puede ser NULL)
- created_at (DATETIME)
- updated_at (DATETIME)


Tabla 'rol':
- id (INT, PRIMARY KEY, AUTO_INCREMENT)
- name (VARCHAR)
- description (VARCHAR)
- status (TINYINT)
- created_at (DATETIME)
- updated_at (DATETIME)

Considera que 'menu_id' se refiere al 'id' de otro registro en la misma tabla 'menu', indicando una relación padre-hijo para submenús. Un menu_id de 0 significa que es un elemento de menú de nivel superior.

SOLO devuelve la consulta SQL. No incluyas punto y coma al final a menos que sea estrictamente necesario para una consulta específica.
"""

PROMPT_SYSTEM_RESPONSE_GENERATOR = """
Eres un asistente amigable que explica los resultados de una consulta a base de datos en lenguaje natural y conciso.
"""

def generate_sql_from_question(user_question):
    """Llama a OpenAI para generar una consulta SQL."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM_SQL_GENERATOR},
                {"role": "user", "content": f"Convierte la siguiente pregunta a una consulta SQL para la tabla 'menu' descrita:\nPregunta: {user_question}\nSQL Query:"}
            ],
            temperature=0.2, 
            max_tokens=150
        )
        sql_query = response.choices[0].message.content.strip()
        # Eliminar posibles ```sql y ``` o comillas al inicio/final si el LLM las añade a pesar del prompt
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        sql_query = sql_query.strip('"; ') # Eliminar punto y coma o comillas sueltas
        return sql_query
    except Exception as e:
        app.logger.error(f"Error al generar SQL con OpenAI: {e}")
        raise

def execute_mysql_query(sql_query):
    """Ejecuta la consulta SQL en MySQL y devuelve los resultados."""
    results = []
    conn = None
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql_query)
        results = cursor.fetchall()
        return results
    except mysql.connector.Error as err:
        app.logger.error(f"Error de MySQL: {err}")
        raise ValueError(f"Error al ejecutar la consulta SQL: {err}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

def generate_natural_response(user_question, sql_query, db_results):
    """Llama a OpenAI para generar una respuesta en lenguaje natural."""
    try:
        # Convertir resultados de DB a JSON string para el prompt
        db_results_str = json.dumps(db_results, ensure_ascii=False, indent=2)

        user_prompt_for_response = f"""
        La pregunta original del usuario fue: "{user_question}"
        La consulta SQL generada y ejecutada fue: "{sql_query}"
        Los resultados COMPLETOS de la base de datos son (una lista de objetos JSON, donde cada objeto es una fila de la base de datos):
        {db_results_str}

        Por favor, proporciona una respuesta clara y en lenguaje natural al usuario basada en TODOS estos resultados y la pregunta original. Por ejemplo, si se piden nombres, lista todos los nombres encontrados.
        Si no hay resultados (la lista JSON está vacía "[]"), indícalo amablemente diciendo que no se encontraron datos para su pregunta.
        Si hubo un error evidente en los resultados (por ejemplo, si el JSON no es una lista o parece un mensaje de error), informa al usuario que no se pudo obtener la información y que podría haber un problema con la pregunta o la forma en que se procesó.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM_RESPONSE_GENERATOR},
                {"role": "user", "content": user_prompt_for_response}
            ],
            temperature=0.7,
            max_tokens=500 # Ajusta según la longitud esperada de la respuesta
        )
        natural_answer = response.choices[0].message.content.strip()
        return natural_answer
    except Exception as e:
        app.logger.error(f"Error al generar respuesta natural con OpenAI: {e}")
        raise 

@app.route('/api/consultaSql', methods=['POST'])
def consulta_sql():
    try:
        data = request.get_json()
        if not data:
            return jsonify(error="Cuerpo de la solicitud JSON vacío o malformado."), 400

        pregunta_usuario = data.get('question')
        user_id = data.get('user_id', 'default_user')

        if not pregunta_usuario:
            return jsonify(error="El parámetro 'question' (pregunta del usuario) es requerido."), 400

        app.logger.info(f"Pregunta recibida: {pregunta_usuario}")

        # Paso 1: Generar consulta SQL
        app.logger.info("Generando consulta SQL...")
        generated_sql = generate_sql_from_question(pregunta_usuario)
        app.logger.info(f"SQL Generado: {generated_sql}")

        if not generated_sql:
            return jsonify(error="No se pudo generar la consulta SQL."), 500

        # Paso 2: Ejecutar consulta SQL
        app.logger.info("Ejecutando consulta SQL en MySQL...")
        db_results = execute_mysql_query(generated_sql)
        app.logger.info(f"Resultados de la DB (primeros 5 si hay muchos): {db_results[:5]}")

        # Paso 3: Generar respuesta en lenguaje natural
        app.logger.info("Generando respuesta en lenguaje natural...")
        final_answer = generate_natural_response(pregunta_usuario, generated_sql, db_results)
        app.logger.info(f"Respuesta final generada: {final_answer}")

        return jsonify({
            "user_question": pregunta_usuario,
            "chatbot_answer": final_answer,
            "generated_sql": generated_sql, 
            "status": "success"
        }), 200
    except ValueError as ve: 
        app.logger.error(f"Error de valor procesando la solicitud: {ve}")
        return jsonify(error=str(ve)), 400 
    except Exception as e:
        app.logger.error(f"Error inesperado en /api/consultaSql: {traceback.format_exc()}")
        return jsonify(error="Ocurrió un error inesperado al procesar la consulta."), 500


app.config['JSON_AS_ASCII'] = False

EXTRACTION_SYSTEM_PROMPT = """
Eres un asistente experto en analizar documentos e imágenes. 
Tu única tarea es extraer la información solicitada por el usuario y devolverla SIEMPRE en un formato JSON válido.
No añadas texto explicativo, notas o comentarios fuera del objeto JSON.
Si no puedes encontrar la información solicitada, devuelve un JSON con un campo que indique el error, por ejemplo: {"error_analisis": "No se encontró la información solicitada en el documento."}.
"""

SUMMARY_SYSTEM_PROMPT = """
Eres un asistente eficiente. Tu tarea es tomar una pregunta original de un usuario y un conjunto de datos en formato JSON que fueron extraídos de un documento.
Basado en ambos, genera una respuesta clara, concisa y en lenguaje natural que responda directamente a la pregunta del usuario utilizando los datos proporcionados.
No te limites a listar los datos; intégralos en una respuesta según como pide.
"""

def extraer_texto_imagen(image_bytes):
    """Extrae texto de una imagen usando Tesseract OCR."""
    try:
        img = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, lang='spa+eng')
        return text.strip() if text else "No se detectó texto en la imagen."
    except Exception as e:
        print(f"Error durante OCR con Tesseract: {e}", flush=True)
        return f"Error en OCR: {str(e)}"

def extraer_texto_pdf(pdf_bytes):
    """Extrae todo el contenido textual y el texto de las imágenes (OCR) de un PDF."""
    full_document_content = []
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        print(f"Procesando PDF de {len(pdf_document)} páginas...")

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            page_number_for_display = page_num + 1

            page_text = page.get_text("text")
            if page_text.strip():
                full_document_content.append(f"--- INICIO TEXTO PÁGINA {page_number_for_display} ---\n{page_text.strip()}\n--- FIN TEXTO PÁGINA {page_number_for_display} ---")

            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    ocr_text = extraer_texto_imagen(image_bytes)
                    if "No se detectó texto" not in ocr_text and "Error en OCR" not in ocr_text:
                        full_document_content.append(f"--- INICIO TEXTO DE IMAGEN {img_idx + 1} (PÁGINA {page_number_for_display}) ---\n{ocr_text}\n--- FIN TEXTO DE IMAGEN ---")
                except Exception as e_img:
                    print(f"No se pudo procesar una imagen en la página {page_number_for_display}: {e_img}")
        
        return "\n\n".join(full_document_content)
    except Exception as e:
        print(f"Error procesando el PDF: {e}")
        return f"Error al leer el contenido del PDF: {str(e)}"

def generar_respuesta(user_prompt, extracted_json_data):
    """Toma el prompt original y el JSON extraído para generar una respuesta conversacional."""
    data_string = json.dumps(extracted_json_data, indent=2, ensure_ascii=False)
    
    summary_user_prompt = f"""
    PREGUNTA ORIGINAL DEL USUARIO:
    "{user_prompt}"

    DATOS ESTRUCTURADOS EXTRAÍDOS DEL DOCUMENTO (JSON):
    ```json
    {data_string}
    ```
    Por favor, genera una respuesta basada en la pregunta y los datos.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": summary_user_prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error al generar el resumen conversacional: {e}")
        return "Se extrajeron los datos, pero ocurrió un error al generar el resumen final."

@app.route('/api/analizardoc', methods=['POST'])
def analizardoc():
    data = request.get_json()
    if not data or 'file_url' not in data or 'prompt' not in data:
        return jsonify({"error": "Faltan los parámetros 'file_url' o 'prompt'"}), 400

    file_url = data['file_url']
    user_prompt = data['prompt']
    file_type = file_url.split('?')[0].split('.')[-1].lower()

    try:
        response = requests.get(file_url, verify=False)
        response.raise_for_status()
        file_bytes = response.content

        is_image_direct = file_type in ['png', 'jpg', 'jpeg', 'webp', 'gif']
        
        messages_for_extraction = [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}]

        if is_image_direct:
            messages_for_extraction.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": file_url}},
                ],
            })
        elif file_type == 'pdf':
            document_context = extraer_texto_pdf(file_bytes)
            full_user_prompt = f'PROMPT DEL USUARIO: "{user_prompt}"\n\nCONTEXTO DEL DOCUMENTO EXTRAÍDO:\n---\n{document_context}\n---'
            messages_for_extraction.append({"role": "user", "content": full_user_prompt})
        else:
            return jsonify({"error": f"Tipo de archivo '{file_type}' no soportado."}), 400

    except Exception as e:
        return jsonify({"error": f"Error al descargar o procesar el archivo: {str(e)}"}), 500

    try:
        extraction_response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=messages_for_extraction,
            max_tokens=4096,
        )
        
        content_string = extraction_response.choices[0].message.content
        if content_string is None:
             return jsonify({"error": "La IA no generó el JSON. Razón: " + extraction_response.choices[0].finish_reason}), 500
        
        structured_data_object = json.loads(content_string)

        conversational_response = generar_respuesta(user_prompt, structured_data_object)

        return jsonify({
            "structured_data": structured_data_object,
            "conversational_response": conversational_response
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error en la llamada a OpenAI: {str(e)}"}), 500
    
###############
# ip and port #
###############
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5015, debug=True)
    