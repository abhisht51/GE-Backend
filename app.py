from flask import Flask,request,jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))
#Model aise load hogya, kept in the root folder 


#Me learning 
@app.route('/')
def hello_world():
    return 'hello world' 



@app.route('/json_route',methods =['POST'])
def predict():
    req_data = request.get_json() # pura request as a JSON le leta hai, cool ;_; 
       
    values1 = pd.json_normalize(req_data) #pura JSON becomes a dataframe 
    
    values1 = pd.get_dummies(values1, columns = ['Sex', 'Obesity', 'CRF', 'CVA', 'Airway disease', 'Thyroid Disease', 'CHF', 'DLP', 'Weak Peripheral Pulse','Lung rales', 'Systolic Murmur', 'Diastolic Murmur', 'Dyspnea','Atypical','Nonanginal',
                                              'Exertional CP','LVH', 'Poor R Progression', 'BBB', 'VHD'])
    
    # TODO basically feature 45 hi aa rhe even though pd.get_dummies technically should have made that 67 :| #HELP_ISHA 
    # TODO hopefully you'll figure out that part of extending the 45 features to 67 , use the json.json as postman json body ;_;, or remote desktop ;_; or teamviewer 
    
    # @bhavya 
    # TODO T/F need to be replaced by 1/0 else the model.predict function will show bakchodi, best is ki front end mein hi if else daal lena for it 

    values = values1.to_numpy() 


    prediction = model.predict(values) #where the values are actually going into the model, idk why but it seemingly has to be a numpy array and also float values 
     
    print("HELLO WORLD")
    print(prediction)
    output = prediction[0]
    # print(jsonify(output))    
    return output

#helper function hai, ignore! 
if __name__ == '__main__':
    app.run(debug=True) 
