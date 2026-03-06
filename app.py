from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model/ckd_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

# Feature names for display
feature_names = [
    'Age', 'Blood Pressure', 'Specific Gravity', 'Albumin', 'Sugar',
    'Red Blood Cells', 'Pus Cell', 'Pus Cell Clumps', 'Bacteria',
    'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine', 'Sodium',
    'Potassium', 'Hemoglobin', 'Packed Cell Volume', 'White Blood Cell Count',
    'Red Blood Cell Count', 'Hypertension', 'Diabetes Mellitus',
    'Coronary Artery Disease', 'Appetite', 'Pedal Edema', 'Anemia'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Debug: Print form data
            print("Form Data Received:", request.form)

            # Helper to safely convert to float
            def get_float(key):
                val = request.form.get(key)
                if val is None or val.strip() == '':
                    return 0.0
                try:
                    return float(val)
                except ValueError:
                    return 0.0

            features = []
            
            # Numerical features
            features.append(get_float('age'))
            features.append(get_float('bp'))
            features.append(get_float('sg'))
            features.append(get_float('al'))
            features.append(get_float('su'))
            
            # Categorical features
            features.append(1 if request.form.get('rbc') == 'normal' else 0)
            features.append(1 if request.form.get('pc') == 'normal' else 0)
            features.append(1 if request.form.get('pcc') == 'present' else 0)
            features.append(1 if request.form.get('ba') == 'present' else 0)
            
            # More numerical features
            features.append(get_float('bgr'))
            features.append(get_float('bu'))
            features.append(get_float('sc'))
            features.append(get_float('sod'))
            features.append(get_float('pot'))
            features.append(get_float('hemo'))
            features.append(get_float('pcv'))
            features.append(get_float('wc'))
            features.append(get_float('rc'))
            
            # More categorical features
            features.append(1 if request.form.get('htn') == 'yes' else 0)
            features.append(1 if request.form.get('dm') == 'yes' else 0)
            features.append(1 if request.form.get('cad') == 'yes' else 0)
            features.append(0 if request.form.get('appet') == 'good' else 1) # Good=0, Poor=1
            features.append(1 if request.form.get('pe') == 'yes' else 0)
            features.append(1 if request.form.get('ane') == 'yes' else 0)
            
            # Debug: Print features list
            print("Features List:", features)

            # STRICTLY convert list to float numpy array
            try:
                features_float = []
                for idx, x in enumerate(features):
                    try:
                        val = float(x)
                        features_float.append(val)
                    except Exception as e:
                        print(f"Failed to convert feature at index {idx} ({feature_names[idx] if idx < len(feature_names) else 'Unknown'}): {x} (type {type(x)}) to float: {e}")
                        features_float.append(0.0)
                
                features_array = np.array(features_float).reshape(1, -1)
                
                # Check for NaNs or Infs
                if np.isnan(features_array).any() or np.isinf(features_array).any():
                    print("Warning: Input contains NaNs or Infs. Replacing with 0.")
                    features_array = np.nan_to_num(features_array)
                    
            except Exception as numpy_err:
                raise ValueError(f"Error creating numpy array: {numpy_err}. Data: {features}")
            
            print("Features Array Shape:", features_array.shape)

            # Scale features
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            try:
                prediction = model.predict(features_scaled)
                probability = model.predict_proba(features_scaled)
            except Exception as model_err:
                 print(f"Model prediction error: {model_err}")
                 raise model_err

            # Get result
            if prediction[0] == 1:
                result = "Positive for CKD"
                risk_level = "High Risk"
                color = "danger"
                icon = "fas fa-exclamation-triangle"
                message = "Please consult a nephrologist immediately for further evaluation."
            else:
                result = "Negative for CKD"
                risk_level = "Low Risk"
                color = "success"
                icon = "fas fa-check-circle"
                message = "Your kidneys appear to be healthy. Maintain a healthy lifestyle!"
            
            # Prepare feature values for display
            feature_values = dict(zip(feature_names, features))
            
            return render_template('result.html', 
                                 prediction=result, 
                                 probability=probability[0][1] * 100,
                                 risk_level=risk_level, 
                                 color=color, 
                                 icon=icon, 
                                 message=message, 
                                 features=feature_values)
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            with open("error_log.txt", "w") as f:
                f.write(traceback_str)
            print("Error Traceback:", traceback_str)
            # Show the detailed error to the user to help debug
            return render_template('predict.html', error=f"Error: {str(e)} | Details: {traceback_str.splitlines()[-1]}")

    
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        features = data['features']
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Here you would typically send an email or save to database
        name = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']
        
        # For now, just flash a success message
        flash('Thank you for your message! We will get back to you soon.', 'success')
        return render_template('contact.html')
    
    return render_template('contact.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
