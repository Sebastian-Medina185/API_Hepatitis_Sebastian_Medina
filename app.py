# api_hepatitis.py
import os
import pickle
import traceback
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings

warnings.filterwarnings("ignore")

# Configuración
PORT = int(os.environ.get("PORT", 5000))
app = Flask(__name__)
CORS(app)

# Función para cargar archivos
def cargar_modelo(path):
    """Carga un modelo o scaler usando joblib o pickle."""
    try:
        from joblib import load as jl_load
        return jl_load(path)
    except ImportError:
        pass

    with open(path, "rb") as f:
        return pickle.load(f)

# Rutas de modelos
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "modelo_regresion_logistica.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")


# Cargar modelo y scaler
try:
    modelo = cargar_modelo(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}")
    modelo = None

try:
    scaler = cargar_modelo(SCALER_PATH)
except Exception as e:
    print(f"[AVISO] No se pudo cargar el scaler: {e}")
    scaler = None

# Ruta principal
@app.route("/")
def index():
    return jsonify({
        "status": "activo",
        "descripcion": "API de predicción de supervivencia. POST a /predict con JSON.",
        "formato": {
            "features": ["lista de valores"],
            "o_dict": {"feature1": "...", "feature2": "..."}
        }
    })

# Procesar entrada
def preparar_entrada(data):
    """Convierte JSON a array de numpy compatible con el modelo."""
    if "features" in data:
        return np.array(data["features"], dtype=float).reshape(1, -1)

    # Si el modelo tiene nombres de features
    if hasattr(modelo, "feature_names_in_"):
        try:
            orden = modelo.feature_names_in_
            return np.array([float(data[f]) for f in orden]).reshape(1, -1)
        except Exception:
            pass

    # Por defecto, ordenar las claves
    valores = [float(data[k]) for k in sorted(data.keys())]
    return np.array(valores, dtype=float).reshape(1, -1)

# Endpoint de predicción
@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    if modelo is None:
        return jsonify({"error": "Modelo no disponible"}), 500

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "JSON inválido o ausente"}), 400

    try:
        X = preparar_entrada(data)

        if scaler:
            try:
                X = scaler.transform(X)
            except Exception as e:
                print(f"[WARN] Error usando scaler: {e}")

        etiquetas = {0: "Vive", 1: "Muere", 2: "Muere"}
        probabilidades = None

        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X)[0]
            probabilidades = {etiquetas[i]: float(p) for i, p in enumerate(proba)}
            # Elegir resultado según mayor probabilidad
            resultado = max(probabilidades, key=probabilidades.get)
            valor_crudo = int(np.argmax(proba))
        else:
            pred = int(modelo.predict(X)[0])
            resultado = etiquetas.get(pred, f"Clase {pred}")
            valor_crudo = pred

        return jsonify({
            "resultado": resultado,
            "valor_crudo": valor_crudo,
            "probabilidades": probabilidades
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Error durante la predicción",
            "detalle": str(e)
        }), 500


# Ejecutar servidor
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
