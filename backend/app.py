from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from flask_cors import CORS  # To allow cross-origin requests from frontend

app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app

# Load models (Ensure correct paths)
model_failure = joblib.load('model/classification_model.pkl')
model_rul = joblib.load('model/rul_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_failure():
    try:
        data = request.json  # Get data from frontend

        # Ensure necessary data is provided
        required_features = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Extract features
        features = np.array([
            data['air_temperature'],
            data['process_temperature'],
            data['rotational_speed'],
            data['torque'],
            data['tool_wear'],
        ]).reshape(1, -1)

        # Failure prediction
        failure_prediction = model_failure.predict(features)
        failure_probability = model_failure.predict_proba(features)[0][1]

        return jsonify({
            'failure_prediction': int(failure_prediction[0]),
            'failure_probability': failure_probability
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_rul', methods=['POST'])
def predict_rul():
    try:
        data = request.json  # Get data from frontend

        # Ensure necessary data is provided
        required_features = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Extract features
        features = np.array([
            data['air_temperature'],
            data['process_temperature'],
            data['rotational_speed'],
            data['torque'],
            data['tool_wear'],
        ]).reshape(1, -1)

        # RUL prediction
        rul_prediction = model_rul.predict(features)

        return jsonify({'rul_prediction': rul_prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/what_if', methods=['POST'])
def what_if():
    try:
        data = request.json

        # Ensure necessary data is provided
        required_features = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Simulate changed features for "what-if" analysis
        adjusted_features = np.array([
            data['air_temperature'],
            data['process_temperature'],
            data['rotational_speed'],
            data['torque'],
            data['tool_wear'],
        ]).reshape(1, -1)

        # Predict updated failure probability
        adjusted_failure_prob = model_failure.predict_proba(adjusted_features)[0][1]

        return jsonify({'adjusted_failure_probability': adjusted_failure_prob})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/timeseries/<machine_id>', methods=['GET'])
def get_time_series(machine_id):
    try:
        # Load data (make sure the path to your dataset is correct)
        machine_data = pd.read_csv('predictive_maintenance.csv')  # Adjust the path if needed
        machine_data = machine_data[machine_data['UDI'] == machine_id]
        
        if machine_data.empty:
            return jsonify({'error': 'Machine data not found'}), 404

        # Return data as JSON
        return machine_data.to_json(orient="records")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)