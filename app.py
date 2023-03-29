from flask import Flask, escape, request, render_template
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


app = Flask(__name__,template_folder='template')

vector = pickle.load(open("vectorizer.pkl",'rb')) 
model = pickle.load(open("finalized_model.pkl",'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
       news = request.form['news']
       pred = model.predict(vector.transform([news]))
       print(pred)
       return render_template('prediction.html',prediction_text=pred)
    return None   

if __name__ == '__main__':
    app.run(debug=True)
