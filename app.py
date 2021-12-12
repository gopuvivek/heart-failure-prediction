from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    serum = request.form.get('serum')
    ejectfrac = request.form.get('ejectfrac')
    model = joblib.load('static/model.pkl')
    x_test = np.reshape([ejectfrac, serum],(1,-1))
    res = model.predict(x_test)
    col = 'red'
    if(res==1):
        pred = "You are in risk of death from heart failure"
    else:
        pred = "You have a healthy heart"
        col = 'green'
    return render_template('predict.html',pred = pred,col=col)

if __name__ == "__main__":
    app.run(debug=True,threaded=True)