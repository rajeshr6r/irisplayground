import pickle
from flask import Flask,request,jsonify, render_template,make_response
import json

app = Flask(__name__)

#useful variables
global loaded_model # declare a global variable to avoid unnecessary reloads
prediction_mapper={0:'Iris-Setosa',1:'Iris-Versicolour',2:'Iris-Virginica'}


def model_load():
    #load the classifier model 
    try:
        loaded_model = pickle.load(open('./model/iris_model.sav', 'rb'))
        return loaded_model
    except Exception as e:
        print ( f"Model Load Error {str(e)}" )

def prediction(loaded_model,array_of_features):    
    if loaded_model:# only if the model is there
        prediction_result=loaded_model.predict(array_of_features)
        try:
            species_type=prediction_mapper.get(prediction_result[0])
            return species_type
        except KeyError:
            species_type="Could Not Be Determined"
            return species_type


@app.route('/home',methods=['GET'])
@app.route('/index',methods=['GET'])
def index():
    return render_template('index.html',prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    prediction_input=request.form.to_dict()
    #attempt model load
    loaded_model=model_load()
    print(prediction_input)
    output=prediction(loaded_model,[[float(item) for item in list(prediction_input.values())]]) # as the form data conatins the values in string 
    #return jsonify(f"Species is {output}")
    return render_template('index.html',prediction_text=f"Species is {output}")

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify("Performed HealthCheck.Container is loading fine ")

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000, debug=True)
 