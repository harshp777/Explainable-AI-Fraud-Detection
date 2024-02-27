# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
import yaml
import random


class TrainTestCreation:


    def __init__(self, params):

        self.test_per = params['test_par']
        self.seed = params['seed']
        


    
    def read_data(self, sheet_names, read_path):
        # This function is to read the data from the input path and store it into a dataframe after merging different sheets in the excel

        try:
            
            dfs = []
    
        # Reading each sheet from the Excel file and store it in a list of DataFrames
            for sheet_name in sheet_names:
                data_frame = pd.read_excel(read_path, sheet_name=sheet_name, engine='openpyxl')
                dfs.append(data_frame)
                loan_information = dfs[0]
                employment = dfs[1]
                personal_information = dfs[2]
                other_information = dfs[3]

                merged_df = pd.merge(loan_information, employment, left_on='User_id', right_on='User id')

                # Merging the previously merged dataframe with 'personal_information' based on 'User_id'
                merged_df = pd.merge(merged_df, personal_information, left_on='User_id', right_on='User id')

                # Merging the previously merged dataframe with 'other_information' based on 'User_id'
                merged_df = pd.merge(merged_df, other_information, left_on='User_id', right_on='User_id')

                self.df = merged_df
                
                

        except Exception as e :

             print(f"Reading failed with error:{e}")


    def drop_nulls(self,columns):
        try:
            
            # only 'four' null values hence it can be removed
            self.df= self.df.dropna(subset= columns)
        except Exception as e :
            print(f"Droppng column with null error: {e}")


    def replace_null_values_with_a_value(self, columns, value):

    # Replacing null values with "missing" in specific columns
        
        try:

            #for replacing
            for column in columns:
                self.df[column] = self.df[column].fillna(value)
            
            """
            replace_with = "missing"
            columns_to_replace = ["Social Profile", "Is_verified", "Married", "Employmet type"]
            """
               
            """
            #Replacing the null values in the 'Amount' column with the value "-1000" to differentiate them from the rest of the data.
            #replace_with= - 1000
            #columns_to_replace = ['Amount']
            """

            """
            replace_with='Z'
            columns_to_replace = ['Tier of Employment']
            """

        except Exception as e:

            print(f"Error in replacing null with a value  :{e}")

        else:
            print("Null values replaced")



    def drop_columns(self, columns_to_drop):
        try:

            for column in columns_to_drop:
                self.df.drop(column, axis=1, inplace=True)
            #columns_to_drop = ["Industry", "Role", "Pincode", 'User_id','User id_x','User id_y']
        except Exception as e:

            print(f"Error in dropping columns with a value  :{e}")

        else:
            print("Unwanted columns dropped")


    def fix_skewness(self, features):

        # features_log= ['Amount','Interest Rate','Tenure(years)','Dependents','Total Payement ','Received Principal','Interest Received']

        try:

            features_log = features
            for f in features_log:
                sk= self.df[f].skew()
                print("Inital skewness in feature: ",f," is: ", sk)

                if sk>3 or sk< -3:

                    Log_Fare = self.df[f].map(lambda i: np.log(i) if i > 0 else 0)
                    self.df[f]=Log_Fare
                    print("Final skewness in feature: ",f," is: ", Log_Fare.skew())

        except Exception as e:

            print(f"Error with defining skewness  :{e}")



    def one_hot_encoding(self, features):

        try:

            #df.select_dtypes(include='object').columns

            #features= ["Gender", "Married", "Home", "Social Profile", "Loan Category", "Employmet type","Is_verified" ]

            self.df = pd.get_dummies(self.df, columns=features)




        except Exception as e:

            print(f"Error with one hot encoding  :{e}")

        else:
            print("Done with One Hot Encoding")



        
    def ordinal_encoding(self, features):

        try:

            #ordinal_features = ["Tier of Employment", "Work Experience"]

            # Create a custom mapping of categories to numerical labels
            tier_employment_order= list(self.df["Tier of Employment"].unique())
            tier_employment_order.sort()

            work_experience_order= [ 0, '<1', '1-2', '2-3', '3-5', '5-10','10+']

            custom_mapping = [tier_employment_order, work_experience_order]
            encoder = OrdinalEncoder(categories=custom_mapping)
            self.df[features] = encoder.fit_transform(self.df[features])


        except Exception as e:

            print(f"Error with ordinal encoding  :{e}")
        else:
            print("Done with Ordinal Encoding ")




    def fix_imbalance_using_oversampling(self, target_column):

        try:

            # Separate the features and the target variable
            X = self.df.drop(target_column, axis=1)
            y = self.df[target_column]

            # Apply random oversampling using RandomOverSampler
            oversampler = RandomOverSampler(random_state=42)
            X, y = oversampler.fit_resample(X, y)

            self.resampled_df = pd.DataFrame(data=X, columns=X.columns)  # Convert back to a DataFrame if needed
            self.resampled_df[target_column] = y


        except Exception as e:
            print(f"Error in handling imbalance{e}")


    
    def split_traintestvalidate(self):



        """This function will split the whole data in the train test as per the test percent provided"""


        try:

            self.train_data, self.test_data = train_test_split(self.resampled_df, random_state = self.seed, test_size = self.test_per)
            idx_list = list(self.test_data.index)
            test_idx = random.sample(idx_list, len(idx_list)//2)
            val_idx = list(set(idx_list) - set(test_idx))
            self.validate_data = self.test_data.loc[val_idx]
            self.test_data = self.test_data.loc[test_idx]


        except Exception as e:

            print(f"Error in handling imbalance{e}")

        else:
            print("Done with the spliting of data")



    def write_data(self, write_path):

        """ This function writes the data into destination folder"""


        try:

            self.train_data.to_csv(Path(str(write_path) + '/train_data.csv'), index= False)
            self.test_data.to_csv(Path(str(write_path) + '/test_data.csv'), index=False)
            self.validate_data.to_csv(Path (str(write_path) + '/validate_data.csv'), index = False)

        except Exception as e:
            print(f"Writing data failed with error: {e}")

        else:
            print(f"Write performed successfully")


    def fit(self, read_path, write_path):

        self.read_data(['loan_information', 'Employment','Personal_information', 'Other_information' ], read_path)
        self.drop_nulls (["Industry","Work Experience"])
        self.replace_null_values_with_a_value( ["Social Profile", "Is_verified", "Married", "Employmet type"], "missing")
        self.replace_null_values_with_a_value( ['Tier of Employment'], "Z")
        self.replace_null_values_with_a_value(["Amount"],-1000)
        self.drop_columns(["Industry", "Role", "Pincode", 'User_id','User id_x','User id_y'])
        self.fix_skewness(['Amount','Interest Rate','Tenure(years)','Dependents','Total Payement ','Received Principal','Interest Received'])
        self.one_hot_encoding(["Gender", "Married", "Home", "Social Profile", "Loan Category", "Employmet type","Is_verified"])
        self.ordinal_encoding(["Tier of Employment", "Work Experience"])
        self.fix_imbalance_using_oversampling("Defaulter")
        self.write_data(write_path)

# Command-line interface using Click
@click.command()
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())



def main (input_filepath, output_filepath):

    """
    Runs data cleaning and splitting script to turn raw data into from (.../raw) into data to be feed to the model. However, if there are
    more features to be added then we have to first push the data into (.../interim) and we can use the build_features.py
    to push the data into (.../processed).
   
    """

    #set up paths

    curr_dir = Path(__file__)
    home_dir= curr_dir.parent.parent.parent
    data_dir = Path(home_dir.as_posix() + '/data')
    input_path = Path(data_dir.as_posix() + input_filepath)
    output_path = Path(data_dir.as_posix() + output_filepath)
    params_path = Path(home_dir.as_posix() + '/params.yaml')
    params = yaml.safe_load(open(params_path))['make_dataset']


    # initiating the object of TrainTestCreation

    split_data = TrainTestCreation(params)


    # Perform the steps of reading,processing , splitting, and writing data

    split_data.fit(input_path, output_path)


    #Execute the main function 
    if __name__ == '__main__' :
        main()









        

        







        









    



                
        


    






            

            




