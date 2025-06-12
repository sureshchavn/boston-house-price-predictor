from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Column names used in the dataset (BostonHousing.csv)
columns = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm',
           'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']

@app.route("/")
def home():
    return render_template("index.html", columns=columns)

@app.route("/predict", methods=["POST"])
def predict():
    # Fetch input values from the form
    input_data = [float(request.form[col]) for col in columns]
    
    # Reshape data and predict
    prediction = model.predict([input_data])[0]
    
    return render_template("index.html", columns=columns, prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)