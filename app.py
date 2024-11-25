import os
import cv2
import base64
import uuid
import numpy as np
from deepface import DeepFace
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configuração do Flask
app = Flask(__name__)
CORS(app)  # Permitir CORS de qualquer origem de maneira mais segura

# Diretório de imagens
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Função para salvar a imagem enviada
def salvar_imagem(image_data: str, name: str) -> str:
    """Salva a imagem no diretório de imagens com nome único."""
    user_dir = os.path.join(IMAGE_DIR, name)
    os.makedirs(user_dir, exist_ok=True)

    # Gerar um nome de arquivo único
    unique_filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(user_dir, unique_filename)

    # Decodificar a imagem e salvar no disco
    try:
        image_bytes = base64.b64decode(image_data.split(",")[1])
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        return file_path
    except (IndexError, base64.binascii.Error) as e:
        raise ValueError("Erro ao decodificar a imagem: formato inválido.")

# Função para processar a imagem recebida
def processar_imagem(image_data: str) -> np.ndarray:
    """Decodifica a imagem base64 para formato OpenCV."""
    try:
        image_bytes = base64.b64decode(image_data.split(",")[1])
        np_array = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    except (IndexError, base64.binascii.Error) as e:
        raise ValueError("Erro ao decodificar a imagem: formato inválido.")

# Função para encontrar correspondência no banco de dados
def encontrar_correspondencia(face: np.ndarray, directory: str, threshold: float = 0.6, model_name: str = "Facenet") -> tuple:
    for user_name in os.listdir(directory):
        user_dir = os.path.join(directory, user_name)
        if os.path.isdir(user_dir):
            for file_name in os.listdir(user_dir):
                file_path = os.path.join(user_dir, file_name)
                try:
                    result = DeepFace.verify(face, file_path, model_name=model_name, enforce_detection=False)
                    distance = result["distance"]
                    if distance < threshold:  # Correspondência encontrada
                        return user_name, distance
                except Exception as e:
                    print(f"Erro ao verificar rosto com {file_path}: {e}")
    return "Desconhecido", None

# Rota para reconhecimento facial
@app.route("/recognize", methods=["POST"])
def recognize_image():
    try:
        # Processar a imagem enviada
        data = request.get_json()
        if not data or "file" not in data:
            return jsonify({"error": "Campo 'file' é obrigatório."}), 400

        face = processar_imagem(data["file"])
        identity, confidence = encontrar_correspondencia(face, IMAGE_DIR)

        if identity != "Desconhecido":
            return jsonify({"message": f"Usuário reconhecido: {identity}", "confidence": confidence}), 200

        return jsonify({"message": "Rosto não reconhecido."}), 404

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Erro ao processar a imagem: {e}"}), 500

# Inicializar o servidor
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
