from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Function to load the model
def get_model():
    model_path = os.path.join(BASE_DIR, 'model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@app.route('/')
@app.route('/index.html')
@app.route('/index.htm')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = get_model()
    if model is None:
        return "Error: model.pkl not found. Please run 'train.py' to generate and train the model first."
        
    try:
        # Get data from form
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        location = int(request.form['location'])
        
        # Create input features for prediction
        input_data = pd.DataFrame([[area, bedrooms, bathrooms, location]], 
                                 columns=['area', 'bedrooms', 'bathrooms', 'location'])
        
        # Predict house price
        prediction = model.predict(input_data)[0]
        
        return render_template('index.html', 
                               prediction=f"{prediction:,.2f}", 
                               area=area,
                               bedrooms=bedrooms,
                               bathrooms=bathrooms,
                               location=location)
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

if __name__ == "__main__":
    # Ensure templates directory exists before running
    templates_dir = os.path.join(BASE_DIR, 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    app.run(debug=True)
