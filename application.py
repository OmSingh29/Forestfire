import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regresor model and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('Relative Humidity (RH) in %: 21 to 90'))
        Ws = float(request.form.get('Wind Speed (WS) in km/h: 6 to 29'))
        Rain = float(request.form.get('Rain total day in mm: 0 to 16.8 FWI Components'))
        FFMC = float(request.form.get('Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5'))
        DMC = float(request.form.get('Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9'))
        ISI = float(request.form.get('Initial Spread Index (ISI) index from the FWI system: 0 to 18.5'))
        Classes = float(request.form.get('Classes: 1->Fire, 2->Not Fire'))
        Region = float(request.form.get('Region: 0->Bejaia region and 1->Sidi Bel-abbes region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
