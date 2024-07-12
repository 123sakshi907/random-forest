

from flask import Flask , render_template,request
import requests
import pandas as pd
import sklearn

from sklearn.datasets import load_iris
from  sklearn.ensemble import RandomForestClassifier
import numpy as np

iris = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(iris.data,iris.target)

app =Flask(__name__)

@app.route("/iris")
def iris():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    predictions = model.predict([features])
    species = iris.target_names[predictions[0]]
    return render_template('index.html', prediction_text=f'the iris species is :{species}')




@app.route("/sakshi")
def sakshi():
    return"project start"

app.run(host='0.0.0.0',port=5000)

