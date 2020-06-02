import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
app=Flask(__name__)
model=pickle.load(open('model2.pkl','rb'))
@app.route('/',methods=['GET'])
def lpa():
    return render_template('lpa.html')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    ls=["yes","no"]
    output=ls(prediction[0],2)
    return render_template('lpa.html',prediction_text='chance of being liver patient'.print(output))
     
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.get_json(force=True)
    prediction=model.predict([np.array(list(data.values()))])
    output=prediction[0]
    return jsonify(output)
if __name__=="__main__":
    app.run(debug=True)

