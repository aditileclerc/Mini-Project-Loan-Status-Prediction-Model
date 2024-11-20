# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:53:10 2024

@author: LENOVO
"""

# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
import joblib

# Create Flask app

app = Flask(__name__)

# Load the trained SVM model
model = joblib.load('trained_model.pkl')

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract user inputs from the form
    gender = request.form['Gender']
    married = request.form['Married']
    dependents= request.form['Dependents']
    education= request.form['Education']
    self_employed= request.form['Self Employed']
    applicantincome= request.form['Applicant Income']
    coapplicantincome= request.form['Coapplicant Income']
    loanamount= request.form['Loan Amount']
    loan_amount_term= request.form['Loan Amount Term']
    credit_history= request.form['Credit History']
    property_area= request.form['Property Area']
    
    
    
    
    gender_numeric = 1 if gender.lower() == 'male' else 0
    married_numeric = 1 if married.lower() == 'yes' else 0
    
    
    se_numeric = 1 if self_employed.lower() == 'yes' else 0


    if property_area.lower() == 'rural':
        proparea_numeric = 0
    elif property_area.lower() == 'semiurban':
        proparea_numeric = 1
    else:
        proparea_numeric = 2

    education_numeric = 1 if education.lower() == 'graduate' else 0
    
    features = [gender_numeric, married_numeric, dependents, education_numeric, se_numeric, applicantincome,
                coapplicantincome, loanamount, loan_amount_term, credit_history, proparea_numeric]
    
    # Make prediction using the loaded model
    prediction = model.predict([features])
    # Render prediction result back to the user
    #return render_template('index.html', prediction=prediction[0])
    
    prediction_result = "Eligible" if prediction == 1 else "Not Eligible"
    
    # Render the prediction result in a new page
    return render_template('prediction_result.html', prediction_result=prediction_result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
