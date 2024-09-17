from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('diabetes_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.json
    
    # Extract the features from the data
    features = np.array([[
        data['pregnancies'], data['glucose'], data['blood_pressure'],
        data['skin_thickness'], data['insulin'], data['bmi'],
        data['dpf'], data['age']
    ]])

    # Standardize the input
    standardized_features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(standardized_features)
    
    # Convert prediction to 'Diabetic' or 'Non-diabetic'
    result = 'Diabetic' if prediction[0] == 1 else 'Non-diabetic'
    
    # Return JSON response
    return jsonify({'prediction': result})

if __name__ == "__main__":
    app.run(debug=True)
