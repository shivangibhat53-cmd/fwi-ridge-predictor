import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# AWS Elastic Beanstalk looks for a variable named 'application'
application = Flask(__name__)
app = application

# Load the machine learning models safely from your folders
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Route 1: The Main Welcome Homepage
@app.route("/")
def index():
    return render_template('index.html')

# Route 2: The Data Form and Prediction Engine
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        # 1. Capture the raw numbers from your form text boxes
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Region = float(request.form.get('Region'))
        
        # 2. Convert text input ('fire' or 'not fire') into a safe number (1.0 or 0.0)
        # This keeps the app from crashing when users type words instead of digits
        classes_input = request.form.get('Classes', '').strip().lower()
        if 'not fire' in classes_input:
            Classes = 0.0
        elif 'fire' in classes_input:
            Classes = 1.0
        else:
            Classes = 0.0  # Fallback default value if there is a typo

        # 3. Organize inputs into a matching array list and scale them
        feature_names = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
        feature_df = pd.DataFrame([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]], columns=feature_names)
        new_data_scaled = standard_scaler.transform(feature_df)
        
        # 4. Run the prediction model on the scaled data matrix
        result = ridge_model.predict(new_data_scaled)
        
        # 5. Extract the single score value and round it to 2 neat decimal points
        final_score = round(result[0], 2)
        
        # Send the rounded final score back to show up on the form dashboard
        return render_template('home.html', results=final_score)
        
    else:
        # If it is a basic GET request, just open the empty entry form cleanly
        return render_template('home.html')

if __name__ == "__main__":
    # Runs the local development server on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
