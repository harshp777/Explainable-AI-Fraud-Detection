import pandas as pd
import pickle
from pathlib import Path
import yaml

#class designed to do production predictions
class DefaultPredictor:

    def __init__(self):

        try:
             

            dir_path = Path(__file__).parent
            self.model_path = Path(dir_path.as_posix() + '/models/test_bestmodel.pkl')
            self.params_path = Path(dir_path.as_posix() + 'features/.yaml')
            self.features= yaml.safe_load(open(self.params_path))['model']['features']
        
        except Exception as e:
            
            print(f"Initialization has been failed due to error: {e}")

        else:

            print("Initialization has been done successfully")


    def dict_to_df(self, dict):
        
        try:

            return pd.DataFrame(dict, index=[0])
        except Exception as e:
            print("f'Dataframe creation from input dictionary has been failed with error : {e}")


    def build_features(self):

        try:

            self.df = self.df[self.features]

        except Exception as e:
            print(f'Feature build has been failed with error : {e}')

        else:
            print('Features created successfully')

    def model_load(self):
        try: 

            self.model = pickle.load(open(self.model_path, 'rb'))

        except Exception as e:
            print(f'Model loading has been failed with error : {e}')

        else:
            print('Model loaded successfully')


    def predict_defaulter(self,dict):

        try:


            self.df = self.dict_to_dif(dict)
            self.build_features()
            self.model_load()
            pred = self.model.predict(self.df)


        except Exception as e:

            print(f'Prediction has been failed because of error : {e}')
    
        else:
            return pred


if __name__=="__main__":

    
        #dummy example if the main file has been ran directly 
        t = DefaultPredictor()
        dict = {'Amount':[2000],'Interest Rate':[15.11],'Tenure(years)':[4],'Tier of Employment': ['A'],'Work Experience':['<1'] ,'Total Income(PA)':[1000],'Dependents':[2],'Delinq_2yrs':[1],'Total Payement ': [1500],'Received Principal': [100],'Interest Received':[50],'Number of loans':[4],'Gender':['Other'],'Married':['Yes'],'Home':['rent'],'Social Profile':['No'],'Loan Category':['Consolidation'],'Employmet type':['Salaried'],'Is_verified': ['Not Verified']}
        print(t.predict_defaulter(dict)[0])
        

        
