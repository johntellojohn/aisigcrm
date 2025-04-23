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
from PyPDF2 import PdfReader
from io import BytesIO


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
    index = request.args.get('index', "chatbot")
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
    - estado_conversacion: Estado actual ('pendiente', 'esperando confirmación', 'finalizado')
    """
    
    # 1. OBTENER Y VALIDAR DATOS DE LA SOLICITUD
    try:
        data = request.get_json()
        pregunta = data.get('question')
        user_id = data.get('user_id') 
        max_histories = data.get('max_histories', 10)
        name_space = data.get('name_space', 'real') 
        json_gpt = data.get('json_gpt')
        
        # Validar campos requeridos
        if not pregunta or not user_id:
            return jsonify(response="La pregunta y el ID de usuario son requeridos."), 400
            
        user_id_int = int(user_id)  # Convertir a entero para consistencia
    
    except Exception as e:
        return jsonify(response=f"Error procesando datos de entrada: {str(e)}"), 400
    
    # 2. CONFIGURACIÓN INICIAL Y CONEXIONES
    try:
        # Inicializar conexión con Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY_PRUEBAS)
        index = pc.Index(data.get('index'))  # Usar índice proporcionado
        index_name = data.get('index')
        
        # Inicializar embeddings de OpenAI
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # 3. BUSCAR CONTEXTO RELEVANTE EN PINECONE
        # 3.1. Buscar documentos relevantes al namespace principal
        query_vector = embeddings.embed_query(pregunta)
        result = index.query(
            namespace=name_space,
            vector=query_vector,
            top_k=1,
            include_metadata=True
        )
        docs = [match['metadata']['text'] for match in result['matches'] if 'metadata' in match]
        
        # 3.2. Buscar historial previo del usuario
        prompt_history = index.query(
            namespace="user_history",
            id=str(user_id),
            top_k=1,
            include_metadata=True
        )
        user_history = [match['metadata']['text'] for match in prompt_history['matches'] if 'metadata' in match]
        user_history_string = ''.join(user_history)
        
        # Verificar si es el primer mensaje del usuario
        primeraRespuesta = not bool(user_history_string)
        
        # 4. PREPARAR LOS DOCUMENTOS PARA EL MODELO
        input_documents = (
            [Document(text) for text in docs] +    # Documentos relevantes
            [Document(text) for text in user_history]  # Historial de usuario
        )
        
        # 5. CONSTRUIR EL PROMPT COMPLETO
        if isinstance(json_gpt, dict) and json_gpt:
            # Si hay un JSON, construir prompt estructurado
            full_prompt = f"""
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
                - `"pendiente"`: si aún falta información.
                - `"esperando confirmación"`: si ya se completó todo y estás preguntando si desea modificar algo.
                - `"finalizado"`: si el usuario confirma que ya no desea cambiar nada más.
            """
        else:
            full_prompt = pregunta  # Si no hay JSON, usar la pregunta directamente
            
        # 6. EJECUTAR EL MODELO DE LENGUAJE
        llm = ChatOpenAI(model_name='gpt-4o-mini', openai_api_key=OPENAI_API_KEY, temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=input_documents, question=full_prompt)
        
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
                r'JSON:\s*(\{.*?\})\s*RESPUESTA:\s*(.*?)\s*RESUMEN:\s*(.*)',
                respuesta,
                re.DOTALL
            )
            
            if match:
                json_str = match.group(1)
                json_completado_raw = json.loads(json_str)
                
                # Si no había texto inicial, usar el conversacional
                if not respuesta_texto:
                    respuesta_texto = match.group(2).strip()

                # Si es primera vez agregar resumen
                if primeraRespuesta:
                    respuesta_texto = f"**Resumen:** {match.group(3).strip()}\n\n{respuesta_texto}"
                
                # Extraer estado de la conversación
                match_estado = re.search(r'ESTADO:\s*-\s*"([^"]+)"', respuesta)
                estado_conversacion = match_estado.group(1).strip() if match_estado else "desconocido"
        else:
            # Cuando no hay JSON involucrado
            json_completado_raw = False
            respuesta_texto = respuesta
            estado_conversacion = False
            
        # 8. ACTUALIZAR HISTORIAL DE CONVERSACIÓN
        # 8.1. Construir nuevo historial
        userHistory = (f"{user_history_string} - Respuesta: {pregunta} - Pregunta:{respuesta}")
        count = userHistory.count("- Respuesta:")
        
        # 8.2. Limitar historial según max_histories
        if count == max_histories:
            patron = re.compile(r"Historial de conversacion:(.*?- Respuesta:.*? - Pregunta:.*?)- Respuesta:", re.DOTALL)
            userHistory = re.sub(patron, "Historial de conversacion:\n-Respuesta:", userHistory, 1)
        
        # 8.3. Actualizar vector en Pinecone
        instructions_values = embeddings.embed_query(userHistory)
        existing_new_vector = index.fetch(ids=[user_id], namespace="user_history")
        current_datetime = datetime.now().isoformat()
        
        if user_id not in existing_new_vector['vectors']:
            # Crear nuevo vector de historial
            index.upsert(
                vectors=[{
                    "id": user_id,
                    "values": instructions_values,
                    "metadata": {
                        "text": "Historial de conversacion:\n" + userHistory,
                        "date": current_datetime
                    }
                }],
                namespace="user_history"
            )
        else:
            # Actualizar vector existente
            index.delete(ids=user_id, namespace="user_history")
            index.upsert(
                vectors=[{
                    "id": user_id,
                    "values": instructions_values,
                    "metadata": {
                        "text": userHistory,
                        "date": current_datetime
                    }
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
        # Manejo detallado de errores
        exc_type, exc_obj, tb = sys.exc_info()
        line_number = tb.tb_lineno
        filename = tb.tb_frame.f_code.co_filename
        return jsonify({
            "response": f"Ocurrió un error: {str(e)}",
            "archivo": filename,
            "linea": line_number,
            "tipo": str(exc_type.__name__)
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

    if not index_name or not id_vector or not file_url or not name_space or not type_file:
        return jsonify(response="Se requiere de la siguiente información (id_vector, link_file, type_file, name_space, index_name)."), 400

    try:
        # Descargar el archivo desde la URL
        response = requests.get(file_url)
        response.raise_for_status()  # Lanza una excepción si la descarga falla

        # Extraer el contenido según el tipo de archivo
        text = ""
        if type_file == "pdf":
            # Extraer texto de un PDF
            pdf_file = BytesIO(response.content)
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text()

        elif type_file == "docx":
            file_stream = BytesIO(response.content)
            result = mammoth.extract_raw_text(file_stream)
            # result.value es ya un string, no necesita to_string()
            if result.value is not None:
                text += result.value
            else:
                text += "No se pudo extraer texto del documento DOCX"

        elif type_file == "doc":
            file_stream = BytesIO(response.content)
            result = mammoth.extract_raw_text(file_stream)
            if result.value is not None:
                text += result.value
            else:
                text += "No se pudo extraer texto del documento DOC"
        
        elif type_file == "txt":
            # Extraer texto de un TXT
            text = response.text
        elif type_file in ["xls", "xlsx"]:
            # Extraer texto de un Excel
            excel_file = BytesIO(response.content)
            if type_file == "xls":
                df = pd.read_excel(excel_file, engine='xlrd')  # Para archivos .xls
            else:
                df = pd.read_excel(excel_file, engine='openpyxl')  # Para archivos .xlsx
            # Convertir el DataFrame a texto
            text = df.to_string(index=False)
        else:
            return jsonify(response="Tipo de archivo no soportado. Use pdf, docx, txt, xls o xlsx."), 400

        

        # Verificar si se extrajo texto correctamente
        if not text:
            return jsonify(response="No se pudo extraer texto del archivo."), 400
        


        # Conexión a Pinecone
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

        # Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Crear el contenido para el metadata
        values_intructions = f"""
            Contenido del archivo ({type_file}):
            {text}
        """

        
        # Buscar el vector con el ID "files"
        instructions_id = "file" + file_url_id
        existing_vector = index.fetch(ids=[instructions_id], namespace=name_space)


        # Si el vector no existe, crearlo; si existe, actualizarlo
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

        return "Información ingresada con éxito", 200

    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/api/upsertFileGeneral', methods=['POST'])
def upsert_file_general():
    data = request.get_json()
    id_vector = data.get('id_vector')
    file_url_id = data.get('link_file_id') 
    file_url = data.get('link_file')
    type_file = data.get('type_file') 
    name_space = data.get('name_space')
    index_name = data.get('index')

    if not index_name or not id_vector or not file_url or not name_space or not type_file:
        return jsonify(response="Se requiere de la siguiente información (id_vector, link_file, type_file, name_space, index_name)."), 400

    try:
        # Descargar el archivo desde la URL
        response = requests.get(file_url)
        response.raise_for_status()  # Lanza una excepción si la descarga falla

        # Extraer el contenido según el tipo de archivo
        text = ""
        if type_file == "pdf":
            # Extraer texto de un PDF
            pdf_file = BytesIO(response.content)
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text()

        elif type_file == "docx":
            file_stream = BytesIO(response.content)
            result = mammoth.extract_raw_text(file_stream)
            # result.value es ya un string, no necesita to_string()
            if result.value is not None:
                text += result.value
            else:
                text += "No se pudo extraer texto del documento DOCX"

        elif type_file == "doc":
            file_stream = BytesIO(response.content)
            result = mammoth.extract_raw_text(file_stream)
            if result.value is not None:
                text += result.value
            else:
                text += "No se pudo extraer texto del documento DOC"
        
        elif type_file == "txt":
            # Extraer texto de un TXT
            text = response.text
        elif type_file in ["xls", "xlsx"]:
            # Extraer texto de un Excel
            excel_file = BytesIO(response.content)
            if type_file == "xls":
                df = pd.read_excel(excel_file, engine='xlrd')  # Para archivos .xls
            else:
                df = pd.read_excel(excel_file, engine='openpyxl')  # Para archivos .xlsx
            # Convertir el DataFrame a texto
            text = df.to_string(index=False)
        else:
            return jsonify(response="Tipo de archivo no soportado. Use pdf, docx, txt, xls o xlsx."), 400

        

        # Verificar si se extrajo texto correctamente
        if not text:
            return jsonify(response="No se pudo extraer texto del archivo."), 400
        


        # Conexión a Pinecone
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

        # Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Crear el contenido para el metadata
        values_intructions = f"""
            Contenido del archivo ({type_file}):
            {text}
        """

        
        # Buscar el vector con el ID "files"
        instructions_id = "file" + file_url_id
        existing_vector = index.fetch(ids=[instructions_id], namespace=name_space)


        # Si el vector no existe, crearlo; si existe, actualizarlo
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

        return "Información ingresada con éxito", 200

    except Exception as e:
        return f"Error: {str(e)}", 500


###############
# ip and port #
###############
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5010, debug=True)
    