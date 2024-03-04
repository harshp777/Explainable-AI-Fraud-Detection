from utils import model_eval_classification
from hyperparameters import get_hyperparameters

import pandas as pd 
import numpy as np 
from pathlib import Path
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import pickle
import optuna
from dvclive import Live
import yaml
import sys
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


class TrainModel:

    def __init__(self, model, trainpath, testpath, feat, ohe_feat, ord_feat, seed, output_path, home_dir,n_trials):
        self.model_cat = model
        self.trainpath = trainpath
        self.testpath = testpath
        self.features = feat
        self.ohe_feat = ohe_feat
        self.ord_feat = ord_feat
        self.seed = seed
        #self.hyperparams = hyperparams
        self.output_path = output_path
        self.home_dir = home_dir
        self.n_trials = n_trials
        self.scoredic_list = []


        try:

             
            if model == 'XGBClassifier':
                self.model_instance = xgb.XGBClassifier

            if model =='RandomForestClassifier':
                
                self.model_instance = RandomForestClassifier
        except Exception as e:
                         
                         
                         print(f"Wrong model has been passed:{e}")

        else:
            print("Model passed successfully")

    
    def read_data(self):
        
        try:
            self.df_train = pd.read_csv(self.trainpath)
            self.df_test = pd.read_csv(self.testpath)

        except Exception as e:
            print(f"Read of data from the paths has been failed due to error:{e}")

        else:

            print("Reading perfomed successfully")


        
    def feature(self):

        """Keeping only required features for modeling and spling into target and input for both test and train"""

        try:

            self.df_train = self.df_train[self.features]
            self.df_test = self.df_test[self.features]
            

        except Exception as e:


            print(f'Feature function failed with error :  {e}')

        else:

            print(f'Data with required features created successfully')



        self.x_train = self.df_train.drop(columns=['Defaulter'])
        self.y_train = self.df_train['Defaulter']
        self.x_test =self.df_test.drop(columns=['Defaulter'])
        self.y_test = self.df_test['Defaulter']



    def cat_encodings(self):
         
         
         """This function initializes the one hot encoder as well as the ordinal enoder"""

         try:
              

            # One hot encoder
            ohe_encoder=OneHotEncoder(handle_unknown='ignore')

            # Ordinal encoder
            # Create a custom mapping of categories to numerical labels
            tier_employment_order= list(self.df_train[self.ord_feat[0]].unique())
            tier_employment_order.sort()

            work_experience_order= [ '0', '<1', '1-2', '2-3', '3-5', '5-10','10+']

            custom_mapping = [tier_employment_order, work_experience_order]

            ord_encoder = OrdinalEncoder(categories=custom_mapping)

            self.encoders = ColumnTransformer(  transformers=[
            ('ohe', ohe_encoder, self.ohe_feat),  # Apply OneHotEncoder to categorical features
            ('ord', ord_encoder, self.ord_feat)  # Apply OrdinalEncoder to ordinal features
            ], remainder='passthrough'      )

         except Exception as e:
              
              print(f'Categorical encoding failed with error :  {e}')

         else:

            print(f' Categorical encoding done successfully')

              




    def objective(self, trial):
            

            """This objective function will be used as an objective function for Optuna and also for fitting models wth different parameters
            and logging it to dvclive"""


            try:


                self.hyperparams = get_hyperparameters(trial)[self.model_cat]

                self.model = self.model_instance(random_state = self.seed, **self.hyperparams)

                self.pipeline = Pipeline([('preprocess', self.encoders),('calssifier', self.model)])

                self.pipeline.fit(self.x_train, self.y_train)

                self.train_score = model_eval_classification(self.pipeline, self.x_train, self.y_train)
                self.test_score = model_eval_classification(self.pipeline, self.x_test, self.y_test)
                self.scoredic_list.append((self.train_score, self.test_score, self.hyperparams))


            except Exception as e:
                print(f'Pipeline initialization and fitting has been failed with error :  {e}')

            else:
                print("Pipeline has been fitted successfully")

            print({'accuracy': self.test_score['Accuracy'],'F1 score':self.test_score['F1 Score'], 'params':self.hyperparams})

            return self.test_score['Accuracy']



    def hyperopt_finetune(self):

            try:

                """In search of beast hyperparameters"""

                study = optuna.create_study(direction='maximize')

                study.optimize(self.objective, n_trials = self.n_trials)

                print('Number of finished trials: ', len(study.trials))
                print('Best trial:')
                self.best_hyper = study.best_trial

                print('Value: ', self.best_hyper.value)
                print('Params: ')
                for key, value in self.best_hyper.params.items():
                    print(f'    {key}: {value}')


                self.model=self.model_instance(random_state = self.seed, **self.best_hyper.params)
                self.pipeline.fit(self.x_train, self.y_train)

            except Exception as e:
                print(f"Hyperparameter has been failed due to error:{e}")
            else:
                print("Hyperparameter finetunning has been seccessfully done")


    def livelog(self):

            try:


                with Live(self.output_path, save_dvc_exp=True) as live:

                    for train_score, test_score, hyperparams in self.scoredic_list:

                        #live logging hyperparameters of model  

                        live.log_params(hyperparams)


                        #live logging all different scoring metrics of train data
                        live.log_metric('train data/Accuracy',train_score['Accuracy'])
                        live.log_metric('train data/Precission',train_score['Precision'])
                        live.log_metric('train data/Recall',train_score['Recall'])
                        live.log_metric('train data/F1 Score',train_score['F1 Score'])

                        #live logging all different scoring metrics of test data
                        live.log_metric('train data/Accuracy',test_score['Accuracy'])
                        live.log_metric('train data/Precission',test_score['Precision'])
                        live.log_metric('train data/Recall',test_score['Recall'])
                        live.log_metric('train data/F1 Score',test_score['F1 Score'])

            except Exception as e:

                #logging for error
                print(f'Live logging has been failed with error :  {e}')

            else:

                #logging for succes
                print('Logging to dvc live has been successfully done')


    def write_data(self):
            """This function saves the data and model into destination folder"""

            try:

            

                # saving a binary model in models folder

                modelfilename = '/models/trained_models/' + self.model_cat +'_pipeline.pkl'
                # Creating the directory if it doesn't exist
                directory = Path(self.home_dir) / 'models' / 'trained_models'
                directory.mkdir(parents=True, exist_ok=True)


                pickle.dump(self.pipeline, open(Path(str(self.home_dir) + modelfilename), 'wb'))


                # Store the data in the form such that it is used by the model hence converting the data into cat encodings

                encoder_step = self.pipeline.named_steps['preprocess']
                df_train_transform = encoder_step.transform(self.df_train)
                df_test_transform = encoder_step.transform(self.df_test) 

                # get the feature names beacuse of one hot encoding

                feature_names = encoder_step.get_feature_names_out()
                df_train_transform = pd.DataFrame(df_train_transform, columns=feature_names)
                df_test_transform = pd.DataFrame(df_test_transform, columns=feature_names)




                traindatafilename = '/data/processed/train_data_model.csv'
                testdatafilename =  '/data/processed/test_data_model.csv'

                traindatafilename = Path(str(self.home_dir) + traindatafilename)
                testdatafilename = Path(str(self.home_dir) + testdatafilename)

                df_train_transform.to_csv(traindatafilename)
                df_test_transform.to_csv(testdatafilename)

            except Exception as e:

            
                # Log if error in saving a model and data
                print(f'Model and data saving has been failed with error: {e}')
        
            else:

                # Log if model saved successfully
                print('Model has been saved successfully')


    def train_model(self):
                    
                    

                    '''Function to run for performing complete training process'''
                    self.read_data()
                    self.feature()
                    self.cat_encodings()
                    self.hyperopt_finetune()
                    self.livelog()
                    print(f'Best hyperparameters of the models are : {self.best_hyper.params}')
                    self.write_data()





traininput_filepath = sys.argv[1]

testinput_filepath = sys.argv[2]

#hypparam_filepath = sys.argv[3]


def main(traininput_filepath, testinput_filepath):

    """ This script trains a model.
    """

    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_dir = Path(home_dir.as_posix() + '/data')

    #path for train and test data
    traininput_path = Path(data_dir.as_posix() + traininput_filepath)
    testinput_path = Path(data_dir.as_posix() + testinput_filepath)

    #dvclive for loggging experiments
    output_path = home_dir.as_posix() + '/dvclive'

    #loading parameters of train model from params.yaml file 

    params_path = Path(home_dir.as_posix()+'/params.yaml') 
    model_params= yaml.safe_load(open(params_path))['train_model']
    model = model_params['model']
    features = model_params['features']
    ohe_feat = model_params['ohe_features']
    ord_feat = model_params['ord_features']
    seed = model_params['seed']
    n_trials = model_params['n_trials']


    # loading pickle files of hyperparameters

    #hypparam_path = Path(home_dir.as_posix() + hypparam_filepath)
    # To do
    
    #hyperparams = pickle.load(open(hypparam_path, 'rb'))[model]

    trf = TrainModel(model,traininput_path, testinput_path, features,ohe_feat,ord_feat, seed, output_path, home_dir,n_trials)

    #training a model
    trf.train_model()


if __name__ == "__main__":
    main(traininput_filepath, testinput_filepath)


    


        

                        









        





