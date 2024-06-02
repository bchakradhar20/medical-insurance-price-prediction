from flask import Flask, render_template, request
import numpy as np
import pickle as pkl

app = Flask(__name__)

# Load the model
model = pkl.load(open('final.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        height_cm = float(request.form['height_cm'])
        weight = float(request.form['weight'])
        known_allergies = request.form['known_allergies']
        any_chronic_diseases = request.form['any_chronic_diseases']
        diabetes = request.form['diabetes']
        history_of_cancer_in_family = request.form['history_of_cancer_in_family']
        any_transplants = request.form['any_transplants']
        blood_pressure_problems = request.form['blood_pressure_problems']
        no_of_major_surgeries = int(request.form['no_of_major_surgeries'])
        
        # Convert categorical variables to numeric
        known_allergies = 1 if known_allergies == 'Yes' else 0
        any_chronic_diseases = 1 if any_chronic_diseases == 'Yes' else 0
        diabetes = 1 if diabetes == 'Yes' else 0
        history_of_cancer_in_family = 1 if history_of_cancer_in_family == 'Yes' else 0
        any_transplants = 1 if any_transplants == 'Yes' else 0
        blood_pressure_problems = 1 if blood_pressure_problems == 'Yes' else 0

        # Convert height to meters
        #height_m = height_cm / 100

        # Calculate BMI
        #bmi = weight / (height_m ** 2)
# Age,Diabetes,BloodPressureProblems,AnyTransplants,AnyChronicDiseases,Height,Weight,KnownAllergies,HistoryOfCancerInFamily,NumberOfMajorSurgeries

        input_data = (age,diabetes, blood_pressure_problems,any_transplants,any_chronic_diseases, height_cm,weight, known_allergies,   history_of_cancer_in_family,   no_of_major_surgeries)
        input_data_array = np.asarray(input_data).reshape(1, -1)
        predicted_prem = model.predict(input_data_array)

        return render_template('result.html', premium=round(predicted_prem[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
