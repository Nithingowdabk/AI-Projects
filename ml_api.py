import pickle
import numpy as np
from flask import Flask, request, render_template

# Load trained model
model = pickle.load(open('model/house_price_model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(request.form['area']), int(request.form['bedrooms']), int(request.form['bathrooms'])]
    prediction = model.predict([data])[0]
    return render_template('index.html', prediction_text=f'Estimated Price: ${prediction:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
