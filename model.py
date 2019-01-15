# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 10:56:57 2019

@author: chyo8003
"""


"""import section"""
import pickle
import pandas as pd
import flask
from flask import Flask , request
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

""""""""""""""""""""""""
app = Flask(__name__)

class Classification:
    
    def __init__(self):
        self.model_path = ""
        self.features = None
        self.model= None
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.seed = 43
        print("classify created")
    
    def load_data(self,path):
        """loading csv data
        input - path
        """
        self.data = pd.read_csv(path)
        self.data = self.data.dropna()
        
    def split_data(self , dependent , _test_size= 0.3):
        X = self.data.drop(dependent,1)
        y = self.data[dependent]
        self.X_train , self.X_test , self.y_train , self.y_test = tts(X,y,test_size = _test_size ,random_state = self.seed)
        self.features = X.columns.values.tolist()
        
    def train_logistic_model(self,c_value=1):
        self.model = LogisticRegression(C = c_value)
        print("model train")
        self.model.fit(self.X_train,self.y_train)
    
    def cross_validate_model(self):
        y_pred = self.model.predict(self.X_test)
        acc_score = accuracy_score(y_pred,self.y_test)
        print( "Accuracy score without hyper parameter tunning is : "+str(acc_score))
    
    def tune_hyper_parameter(self , parameters):
        params = parameters
        grid_search = GridSearchCV(self.model,params , cv=5)
        grid_search.fit(self.X_train,self.y_train)
        return str(grid_search.best_params_)
    
    def dump_model(self,path_to_dump):
        self.model_path = path_to_dump+"\\ready_model_new.pkl"
        with open(self.model_path, 'wb') as pickle_file:
            pickle.dump(all, pickle_file)
        
    def load_model(self):
        self.model = pickle.load(self.model_path)
        return "model is loaded"
    
    def predict(self,data):
        data_df = pd.DataFrame(data, columns = self.features)
        prediction = self.model.predict(data_df)
        return prediction
        
"""model training and dumping"""""""""""""""""""""""""""
classify = Classification()
classify.load_data("C:\\yogesh\\data_sets\\loan_prediction\\train.csv")
classify.split_data("Loan_Status")
classify.train_logistic_model(c_value=1)
classify.cross_validate_model()  
param = {"C":[0.01,0.1,1]} 
classify.tune_hyper_parameter(param)
classify.dump_model("C:\\yogesh\\models")
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

@app.route("/loadmodel" , methods = ['GET'])
def load_model():
    classify.load_model()

@app.route("/predict" , methods = ['GET','POST'])
def make_predictiction():
    test_data = request.data
    result = classify.predict(test_data)
    return result

if __name__ == "__main__":
    app.run()

    
    













        
