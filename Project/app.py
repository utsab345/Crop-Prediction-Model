import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Render the result
        return render_template('index.html', prediction_text=f"The predicted crop is: {prediction}")
    except ValueError:
        return render_template('index.html', prediction_text="Invalid input. Please enter valid numbers.")
    except Exception as e:
        return render_template('index.html', prediction_text=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
