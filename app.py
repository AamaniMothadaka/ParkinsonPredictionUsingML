import numpy as np
from flask import  Flask, request,render_template
import pickle
import numpy
import pandas as pd
import xgboost as xgb

app  = Flask(__name__, template_folder='templates', static_folder='static')

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    input_text_sp=[]
    d={}
    for key, value in request.form.items():
        d[key]=value
    #np_data = np.asarray(input_text_sp, dtype=np.float32)
    #prediction = model.predict(np_data.reshape(1,-1))
    print(d)

    new_data_df = pd.DataFrame([d])
    parkinson = pd.read_csv('C:/Users/USER/Downloads/parkinsonData.csv')
    X = parkinson.drop(['status','name'],axis=1)

# Ensure the order of columns matches the order of features in the training data
    new_data_df = new_data_df[X.columns]

# Ensure data types match (convert if needed)
    new_data_df = new_data_df.astype(float)

# Make predictions
    prediction = model.predict(xgb.DMatrix(new_data_df))
    
    if prediction >= 0.5:
        output = "This person has a parkinson disease"
    else:
        output = "This person has no parkinson disease"

    return render_template("index.html", message= output)

if __name__ == "__main__":
    app.run(debug=True)