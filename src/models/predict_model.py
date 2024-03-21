import pandas as pd
import pickle
from pathlib import Path
import yaml

#class designed to do production predictions
class DefaultPredictor:

    def __init__(self):

        #initialising all necessary paths and parameters 
        self.curr_dir = Path(__file__)
        self.home_dir = self.curr_dir.parent.parent.parent
        self.model_path = Path(str(self.home_dir) + '/models/test_bestmodel.pkl')
        #print(self.model_path)
        self.params_path = Path(str(self.home_dir) + '/params.yaml')
        #print(self.params_path)
        self.features = yaml.safe_load(open(self.params_path))['train_model']['features']
        self.features.remove('Defaulter')
        #print(self.features)


    def dict_to_df(self,dict):
        
        #converting recieved input dictionary to dataframe
        #print(pd.DataFrame(dict))
        return pd.DataFrame(dict)

   
    def buildfeatures(self):

        # condiering only required features
        self.df = self.df[self.features]
        #print(self.df)
        
    def model_load(self):

        #loading a binary model through pickle
        self.model = pickle.load(open(self.model_path, 'rb'))

    def predict_defaulter(self,dict):

          # function that performs all other function together and returns a predicted output
        try:
            self.df = self.dict_to_df(dict)
            self.buildfeatures()
            self.model_load()
            pred = self.model.predict(self.df)           
        except Exception as e:
             print(f'Prediction has been failed because of error : {e}')
        else:
            return pred
    
if __name__ == "__main__":

        #dummy example if the main file has been ran directly 
        t = DefaultPredictor()
        dict = {'Amount':[2000],'Interest Rate':[15.11],'Tenure(years)':[4],'Tier of Employment': ['A'],'Work Experience':['<1'] ,'Total Income(PA)':[1000],'Dependents':[2],'Delinq_2yrs':[1],'Total Payement ': [1500],'Received Principal': [100],'Interest Received':[50],'Number of loans':[4],'Gender':['Other'],'Married':['Yes'],'Home':['rent'],'Social Profile':['No'],'Loan Category':['Consolidation'],'Employmet type':['Salaried'],'Is_verified': ['Not Verified']}
        print(t.predict_defaulter(dict)[0])
        