from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load saved models
models = {
    'Logistic Regression': pickle.load(open('logit_model.pkl', 'rb')),
    'SVM': pickle.load(open('svc_model.pkl', 'rb')),
    'K-Nearest Neighbors': pickle.load(open('k_nearest_model.pkl', 'rb')),
    'Decision Tree': pickle.load(open('dt_model.pkl', 'rb')),
    #'Random Forest': pickle.load(open('rf_model.pkl', 'rb')),
    #'XGBoost': pickle.load(open('xgb_model.pkl', 'rb')),
    #'Stacking: pickle.load(open('clf_model.pkl', 'rb'))
}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    bmi = float(request.form['bmi'])
    high_bp = int(request.form['high_bp'])
    fbs = float(request.form['fbs'])
    hba1c_level = float(request.form['hba1c_level'])
    smoking = int(request.form['smoking'])
    model_choice = request.form['model_choice']

    input_data = np.array([[age, gender, bmi, high_bp, fbs, hba1c_level, smoking]])
    model = models[model_choice]
    prediction = model.predict(input_data)[0]
    result = "Healthy" if prediction == 0 else "Unhealthy"

    return render_template('result.html', result=result, model=model_choice)
if __name__ == '__main__':
    app.run(debug=True) 
    
