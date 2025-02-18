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
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI  
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from dotenv import load_dotenv
import sys


print("Este es un mensaje de prueba", flush=True)  # Método 1
sys.stdout.flush()  # Método 2


# Obtener credenciales desde .env
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Inicializar Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))


app = Flask(__name__)


############
# Chat Bot #
############
load_dotenv()

PINECONE_API_KEY_PRUEBAS = os.getenv('PINECONE_API_KEY_PRUEBAS')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@app.route('/index')
def index():
    return "Hello, You Human!!"


PINECONE_API_KEY_PRUEBAS = os.getenv('PINECONE_API_KEY_PRUEBAS')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

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
    
    # Cuerpo
    data = request.get_json()
    pregunta = data.get('question')
    user_id = data.get('user_id') 
    max_histories = data.get('max_histories', 10)
    name_space = data.get('name_space', 'real') 

    # Variables
    user_id_int = int(user_id)

    if not pregunta or not user_id:
        return jsonify(response="La pregunta y el ID de usuario son requeridos."), 400

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY_PRUEBAS)
        index = pc.Index(data.get('index'))  # Utilizar el índice proporcionado en la solicitud
        index_name = data.get('index')

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        query_vector = embeddings.embed_query(pregunta)
        result = index.query(
            namespace=name_space,
            vector=query_vector,
            top_k=1,
            include_metadata=True
        )

        prompt_history = index.query(
            namespace="user_history",
            id=str(user_id),
            top_k=1,
            include_metadata=True
        )

        docs = [match['metadata']['text'] for match in result['matches'] if 'metadata' in match]

        user_history = [match['metadata']['text'] for match in prompt_history['matches'] if 'metadata' in match]

        user_history_string = ''.join(user_history)

        print(f"\n\n\nSIIIIIIIIIIIIII: {user_history_string}\n\n\n")

        # Crear objetos Document de langchain con el texto de los documentos recuperados
        input_documents = (
            [Document(page_content=text) for text in docs] +
            [Document(page_content=text) for text in user_history]
        )

        #print(input_documents)

        llm = ChatOpenAI(model_name='gpt-4o-mini', openai_api_key=OPENAI_API_KEY, temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")

        # Usar el full_prompt como parte del contexto del sistema
        respuesta = chain.run(
            input_documents=input_documents,
            question=pregunta
        )

        # Historial nuevo de usuario
        userHistory = (f"{user_history_string} - Respuesta: {pregunta} - Pregunta:{respuesta}")
        count = userHistory.count("- Respuesta:")
        #print(count)
        # Eliminación de data
        if count==max_histories:
            patron = re.compile(r"Historial de conversacion:(.*?- Respuesta:.*? - Pregunta:.*?)- Respuesta:", re.DOTALL)
            userHistory_delete = re.sub(patron, "Historial de conversacion:\n-Respuesta:", userHistory, 1)
            print(f"CADENA ELIMINADA: {userHistory_delete}")
            userHistory = userHistory_delete


        # Buscar el vector con el ID "Prompt"
        instructions_values = embeddings.embed_query(userHistory)

        existing_new_vector = index.fetch(ids=[user_id], namespace="user_history")

        current_datetime = datetime.now().isoformat()

        if user_id not in existing_new_vector['vectors']:
            index.upsert(
                vectors=[
                    {
                        "id": user_id,
                        "values": instructions_values,
                        "metadata": {
                            "text": "Historial de conversacion:\n" + userHistory,
                            "date": current_datetime
                        }
                    }
                ],
                namespace="user_history"
            )

        else:
            index.delete(ids=user_id, namespace="user_history")
            index.upsert(
                vectors=[
                    {
                        "id": user_id,
                        "values": instructions_values,
                        "metadata": {
                            "text": userHistory,
                            "date": current_datetime
                        }
                    }
                ],
                namespace="user_history"
            )


        # Verificar intenciones
        intencion_detectada = False
        prompt_intentions = index.query(
            namespace=name_space,
            id="IntencionesDelBot",
            top_k=1,
            include_metadata=True
        )

        # Extraer el texto de intenciones desde la respuesta de Pinecone
        intentions_data = (
            prompt_intentions.get("matches", [{}])[0]
            .get("metadata", {})
            .get("text", "")
        )

        intenciones_formateadas = {}

        # Intentar convertir el JSON solo si `intentions_data` tiene contenido válido
        if intentions_data:
            try:
                intenciones_formateadas = json.loads(
                    intentions_data.split("Lista de intenciones:\n")[-1]
                )
            except json.JSONDecodeError as e:
                print(f"Error al procesar JSON de intenciones: {e}")

        print("Intenciones formateadas:", intenciones_formateadas)

        # Si hay intenciones, construimos el prompt
        if intenciones_formateadas:
            prompt = "A continuación, se te proporcionará una pregunta o comentario de un usuario.\n"
            prompt += "Tu tarea es determinar la intención del usuario basándote en las siguientes opciones:\n\n"

            # Agregar intenciones con sus descripciones al prompt
            for intencion, descripcion in intenciones_formateadas.items():
                prompt += f"- {intencion}: {descripcion}\n"
            
            # Instrucción explícita para devolver "Ninguna" si no hay coincidencia
            prompt += "\nSi la pregunta del usuario no coincide con ninguna de las opciones anteriores, responde con 'Ninguna'.\n"
            prompt += f'\nPregunta/Comentario del usuario: "{pregunta}"\n'
            prompt += "Intención detectada:"

            # Llamar a la API de OpenAI con el formato actualizado
            try:
                respuesta_gpt = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Usar un modelo válido
                    messages=[
                        {"role": "system", "content": "Eres un asistente que ayuda a detectar intenciones."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=50,  # Limitar la longitud de la respuesta
                    temperature=0.5,  # Controlar la creatividad (0 = más determinista, 1 = más creativo)
                )
                # Extraer la intención detectada por GPT
                intencion_detectada = respuesta_gpt.choices[0].message.content.strip()
            except Exception as e:
                print(f"\n\n\n")
                print("Error al llamar a la API de OpenAI:", str(e))
                print(f"\n\n\n")

        print(f"\n\n\nINTENTION '{intencion_detectada}'\n\n\n")

        if intencion_detectadaand and intencion_detectada.lower() != "ninguna":
            print(f"\n\n\nEl usuario tiene la intención de '{intencion_detectada}'\n\n\n")

            # Enviar solicitud a la API /close-connection
            try:

                close_connection_response = requests.post(
                    'https://sigcrm.pro/close-conection',  
                    json={"id": user_id_int, "type": intencion_detectada, "index": index_name}  
                )

                if close_connection_response.status_code == 201 or close_connection_response.status_code == 200:
                    index.delete(ids=user_id, namespace="user_history")
                    print("Conexión cerrada exitosamente")
                    
                else:
                    print(f'Error al cerrar la conexión: {close_connection_response.status_code}')

            except Exception as e:
                print(f'Ocurrió un error al cerrar la conexión: {str(e)}')

        else:
            print("\n\n\nNo se ha detectado ninguna intención específica*****\n\n\n")

        return jsonify(response=respuesta), 200
    
    except openai.error.AuthenticationError:
        return jsonify(response="La API key no es válida."), 401
    except Exception as e:
        return jsonify(response=f"Ocurrió un error: {str(e)}"), 500

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


###############
# ip and port #
###############
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5010, debug=True)
    