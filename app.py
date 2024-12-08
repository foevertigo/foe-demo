from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

# Load the pre-trained model and preprocessor
with open('flood_prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

# Define the columns expected by the model
expected_columns = ['Avg_smlvl_at15cm', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 
                    'SUBDIVISION_Andaman & Nicobar', 'SUBDIVISION_Andhra Pradesh', 'SUBDIVISION_Arunachal Pradesh', 'SUBDIVISION_Bihar', 
'SUBDIVISION_Chhattisgarh', 'SUBDIVISION_Goa', 'SUBDIVISION_Gujarat', 'SUBDIVISION_Himachal Pradesh', 'SUBDIVISION_Jammu & Kashmir', 
'SUBDIVISION_Jharkhand', 'SUBDIVISION_Karnataka', 'SUBDIVISION_Kerala', 'SUBDIVISION_Lakshadweep', 'SUBDIVISION_Madhya Pradesh', 
'SUBDIVISION_Maharashtra', 'SUBDIVISION_Odisha', 'SUBDIVISION_Punjab', 'SUBDIVISION_Rajasthan', 'SUBDIVISION_Sikkim', 
'SUBDIVISION_Tamil Nadu', 'SUBDIVISION_Telangana', 'SUBDIVISION_Uttar Pradesh', 'SUBDIVISION_Uttarakhand', 
'SUBDIVISION_West Bengal', 'SUBDIVISION_Nagaland Manipur Mizoram Tripura', 'SUBDIVISION_Delhi Haryana Chandigarh', 
'SUBDIVISION_Assam Meghalaya']  # Add all possible one-hot encoded columns for subdivisions

month = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json

        # Validate input
        if 'month' not in data or 'SUBDIVISION' not in data:
            return jsonify({'error': 'Missing required fields: month or SUBDIVISION'}), 400

        month = data['month']
        subdivision = data['subdivision']
        rainfall = data.get('rainfall', 0)  # Default to 0 if not provided
        soil_moisture = data.get('soil_moisture', 80)  # Default soil moisture value if not provided

        # Create an input DataFrame
        input_data = pd.DataFrame({
            'Avg_smlvl_at15cm': [soil_moisture],
            month: [rainfall]
        })

        # Add missing columns with default values
        for col in expected_columns:
            if col.startswith('SUBDIVISION_'):
                input_data[col] = 1 if col == f'SUBDIVISION_{subdivision}' else 0
            elif col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match the model's training data
        input_data = input_data[expected_columns]

        # Preprocess the input data
        input_scaled = preprocessor.transform(input_data)

        # Predict using the trained model
        prediction = model.predict(input_scaled)

        # Return the prediction
        result = 'Flood' if prediction[0] == 1 else 'No Flood'
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
