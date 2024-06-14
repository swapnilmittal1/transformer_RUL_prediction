from flask import Flask, request, jsonify
import torch
import numpy as np
from model import Transformer  # Ensure this import matches your model's definition
import os

app = Flask(__name__)

# Define model architecture
d_model = 128
heads = 4
N = 2
m = 14
dropout = 0.1

print("Loading model...")
# Ensure the path to the model file is correct
model_path = 'rul_model.pth'
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
else:
    print(f"Model file found: {model_path}")

model = Transformer(m, d_model, N, heads, dropout)
model.load_state_dict(torch.load(model_path))
model.eval()
print("Model loaded.")

def preprocess_input(data):
    # Preprocess the input data
    X = np.array(data).astype(np.float32)
    X_tensor = torch.tensor(X).unsqueeze(0).unsqueeze(0)  # Adjust shape to match model input
    return X_tensor

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = data['input']
        X_tensor = preprocess_input(input_data)
        with torch.no_grad():
            outputs = model(X_tensor, 0)
        prediction = outputs.numpy().tolist()
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5001)  # Changed port to 5001
