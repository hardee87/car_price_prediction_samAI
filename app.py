from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

# Initialize Flask application
app = Flask(__name__)

# Load saved machine learning model and feature list
model = joblib.load('car_model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    """Renders the main homepage UI."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles AJAX form submission, formats input data, 
    and returns predicted car price as JSON.
    """
    try:
        # 1. Extract values from the HTML form submit request
        present_price = float(request.form['present_price'])
        driven_kms = int(request.form['driven_kms'])
        car_age = int(request.form['car_age'])
        owner = int(request.form['owner'])
        fuel_type = request.form['fuel_type']
        selling_type = request.form['selling_type']
        transmission = request.form['transmission']

        # 2. Build a single-row DataFrame matching training format
        raw_data = pd.DataFrame([{
            'Present_Price': present_price,
            'Driven_kms': driven_kms,
            'Owner': owner,
            'Car_Age': car_age,
            'Fuel_Type': fuel_type,
            'Selling_type': selling_type,
            'Transmission': transmission
        }])

        # 3. Apply one-hot encoding on categorical features
        encoded_data = pd.get_dummies(raw_data)

        # 4. Reindex columns to match exact features used in training
        input_df = encoded_data.reindex(columns=model_columns, fill_value=0)

        # 5. Predict price with trained model
        prediction = model.predict(input_df)[0]
        final_price = round(max(0, prediction), 2)

        # 6. Return response to frontend JavaScript
        return jsonify({'price': final_price})

    except Exception as e:
        # Return error details if input processing fails
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run development server locally on port 5000
    app.run(debug=True)