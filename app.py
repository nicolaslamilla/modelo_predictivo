from flask import Flask, request, jsonify
import os
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('modelo_predictivo.pkl')
API_KEY = os.getenv('API_KEY') 

@app.route('/')
def home():
    return "API funcionando correctamente."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener la clave API de los headers
        api_key = request.headers.get('X-API-KEY')  
        
        # Verificar si la clave API es válida
        if api_key != API_KEY:
            return jsonify({'error': 'No autorizado'}), 401
        
        data = request.get_json()
        features = data['features']  # Espera una lista de números
        prediccion = modelo.predict([features])
        return jsonify({'resultado': int(prediccion[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
