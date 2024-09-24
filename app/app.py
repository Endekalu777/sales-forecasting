import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__, template_folder="./templates")

# Load the Random Forest model (assuming it is saved in joblib format)
try:
    rf_model = joblib.load("./artifacts/rf_model.pkl")
    
    # Remove the scaler-related code since it's not needed for Random Forest
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError("Failed to load the model: " + str(e))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json.get('data')
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Convert data to the required format
        input_data = np.array(list(data.values())).reshape(1, -1)
        print("Input data shape:", input_data.shape)

        # Validate input length (assuming you still want to check for correct input dimensions)
        expected_features = input_data.shape[1]  # Adjust based on your model
        if input_data.shape[1] != expected_features:
            return jsonify({"error": "Invalid input data length."}), 400

        # Make prediction (without scaling)
        output = rf_model.predict(input_data)
        return jsonify({"prediction": output[0]})  # Adjust based on your model's output shape
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.values()
        final_input = np.array([float(x) for x in data]).reshape(1, -1)
        print("Final input shape:", final_input.shape)

        # Validate input length
        expected_features = final_input.shape[1]  # Adjust based on your model
        if final_input.shape[1] != expected_features:
            return render_template("home.html", prediction_text="Invalid input data length.")

        # Make prediction (without scaling)
        output = rf_model.predict(final_input)
        return render_template("home.html", prediction_text="The predicted Sales is {:.2f}".format(output[0]))  
    except ValueError as e:
        return render_template("home.html", prediction_text="Invalid input data: {}".format(e))
    except Exception as e:
        return render_template("home.html", prediction_text="Error occurred: {}".format(e))

if __name__ == '__main__':
    app.run(debug=True)
