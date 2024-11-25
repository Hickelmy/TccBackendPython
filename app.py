import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import uuid

# Configuração do Flask
app = Flask(__name__)
CORS(app)  # Permitir CORS

# Diretório de imagens
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Função para carregar o banco de dados de rostos conhecidos
def carregar_banco_de_dados(directory: str) -> dict:
    """Carrega o banco de dados de imagens conhecidas em um dicionário."""
    database = {}
    for user_name in os.listdir(directory):
        user_dir = os.path.join(directory, user_name)
        if os.path.isdir(user_dir):
            for file_name in os.listdir(user_dir):
                file_path = os.path.join(user_dir, file_name)
                database[f"{user_name}_{file_name}"] = file_path
    return database

# Função para salvar a imagem enviada
def salvar_imagem(image_data: str, name: str) -> str:
    """Salva a imagem no diretório de imagens com nome único."""
    user_dir = os.path.join(IMAGE_DIR, name)
    os.makedirs(user_dir, exist_ok=True)

    # Gerar um nome de arquivo único
    unique_filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(user_dir, unique_filename)

    # Decodificar a imagem e salvar no disco
    image_bytes = base64.b64decode(image_data.split(",")[1])
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    return file_path

# Função para encontrar correspondência no banco de dados
def encontrar_correspondencia(face: np.ndarray, database: dict, threshold: float = 0.6, model_name: str = "Facenet") -> tuple:
    """Encontra a correspondência mais próxima para um rosto no banco de dados."""
    for identity, db_path in database.items():
        try:
            result = DeepFace.verify(face, db_path, model_name=model_name, enforce_detection=False)
            distance = result["distance"]
            if distance < threshold:  # Correspondência encontrada
                return identity.split('_')[0], distance
        except Exception as e:
            print(f"Erro ao verificar rosto com {identity}: {e}")
    return "Desconhecido", None

# Função para converter imagem para base64
def image_to_base64(image: np.ndarray) -> str:
    """Converte uma imagem OpenCV para string base64."""
    _, buffer = cv2.imencode('.jpg', image)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

# Função para processar a imagem recebida
def processar_imagem(image_data: str) -> np.ndarray:
    """Decodifica a imagem base64 para formato OpenCV."""
    image_bytes = base64.b64decode(image_data.split(",")[1])
    np_array = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

# Inicialização do banco de dados
database = carregar_banco_de_dados(IMAGE_DIR)




# Rota para deletar a pasta de um usuário
@app.route("/delete_user/<string:username>", methods=["DELETE"])
def delete_user(username):
    """Deleta a pasta de um usuário e todas as suas imagens."""
    user_dir = os.path.join(IMAGE_DIR, username)
    try:
        if os.path.exists(user_dir):
            # Remover toda a pasta
            for root, dirs, files in os.walk(user_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                os.rmdir(root)
            return jsonify({"message": f"Usuário '{username}' deletado com sucesso."}), 200
        else:
            return jsonify({"error": "Usuário não encontrado."}), 404
    except Exception as e:
        return jsonify({"error": f"Erro ao deletar o usuário: {e}"}), 500


# Rota para deletar uma imagem específica
@app.route("/delete_image/<string:username>/<string:image_name>", methods=["DELETE"])
def delete_image(username, image_name):
    """Deleta uma imagem específica de um usuário."""
    image_path = os.path.join(IMAGE_DIR, username, image_name)
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
            return jsonify({"message": f"Imagem '{image_name}' deletada com sucesso."}), 200
        else:
            return jsonify({"error": "Imagem não encontrada."}), 404
    except Exception as e:
        return jsonify({"error": f"Erro ao deletar a imagem: {e}"}), 500


# Rota para upload de imagem e armazenamento
@app.route("/upload", methods=["POST"])
def upload_image():
    """Rota para armazenar uma imagem enviada pelo cliente."""
    try:
        # Verificar se os dados foram enviados
        data = request.get_json()
        if not data or "file" not in data or "name" not in data:
            return jsonify({"error": "Os campos 'file' e 'name' são obrigatórios"}), 400

        # Salvar a imagem
        file_path = salvar_imagem(data["file"], data["name"])

        # Atualizar o banco de dados
        global database
        database = carregar_banco_de_dados(IMAGE_DIR)

        return jsonify({"message": f"Imagem salva com sucesso: {file_path}"}), 200

    except Exception as e:
        return jsonify({"error": f"Erro ao salvar a imagem: {e}"}), 500


# Função para converter uma imagem no disco para base64
def file_to_base64(file_path: str) -> str:
    """Converte uma imagem do caminho do arquivo para string base64."""
    with open(file_path, "rb") as image_file:
        return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode()}"


# Função para listar usuários e suas imagens
def listar_usuarios(directory: str) -> dict:
    """Lista todos os usuários e suas imagens em base64."""
    users = {}
    for user_name in os.listdir(directory):
        user_dir = os.path.join(directory, user_name)
        if os.path.isdir(user_dir):
            images = []
            for file_name in os.listdir(user_dir):
                file_path = os.path.join(user_dir, file_name)
                images.append(file_to_base64(file_path))
            users[user_name] = images
    return users

@app.route("/users", methods=["GET"])
def get_users():
    """Rota para listar todos os usuários e suas imagens."""
    try:
        users = listar_usuarios(IMAGE_DIR)
        return jsonify(users), 200
    except Exception as e:
        return jsonify({"error": f"Erro ao listar usuários: {e}"}), 500


# Rota para reconhecimento facial
@app.route("/recognize", methods=["POST"])
def recognize_image():
    """Rota para reconhecer um rosto a partir de uma imagem."""
    try:
        # Verificar se os dados foram enviados
        data = request.get_json()
        if not data or "file" not in data:
            return jsonify({"error": "O campo 'file' é obrigatório"}), 400

        # Processar a imagem recebida
        image = processar_imagem(data["file"])

        # Detectar rostos na imagem
        faces = DeepFace.extract_faces(img_path=image, detector_backend='opencv', enforce_detection=False)
        if not faces:
            return jsonify({"message": "Nenhum rosto detectado."}), 404

        # Selecionar a primeira região detectada
        face_region = faces[0].get('region')
        if not face_region:
            return jsonify({"message": "Erro ao identificar a região do rosto."}), 400

        x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
        face = image[y:y+h, x:x+w]  # Extrair a região do rosto

        # Encontrar correspondência no banco de dados
        identity, confidence = encontrar_correspondencia(face, database)

        # Desenhar a caixa verde ao redor do rosto
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Converter a imagem com a caixa verde para base64
        img_extracted = image_to_base64(image)

        # Retornar os resultados
        return jsonify({
            "message": f"Usuário reconhecido: {identity}" if identity != "Desconhecido" else "Rosto não reconhecido.",
            "confidence": confidence,
            "img_extracted": img_extracted
        }), 200 if identity != "Desconhecido" else 404

    except Exception as e:
        return jsonify({"error": f"Erro ao processar a imagem: {e}"}), 500

# Inicializar o servidor
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
