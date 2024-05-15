import boto3
import pandas as pd
import yaml
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt




class CI_test:

    def __init__(self):

        try:

            s3 = boto3.client("s3")

            s3.download_file(

                Bucket='eai-fraud-data', Key= 'validate_data.csv', Filename="validate_data.csv"
            ),
        
            s3.download_file(

                Bucket='eai-fraud-data', Key= 'test_bestmodel.pkl', Filename="test_bestmodel.pkl"
            )


        
        except Exception as e:


            print(f"Unable to connect to S3 due to: {e}")


        else:

            self.df = pd.read_csv("validate_data.csv")




    def predict(self):

        features = yaml.safe_load(open("params.yaml"))["train_model"]['features']
        self.df = self.df[features]

        #seperating input and output data 

        self.x = self.df.drop(columns = ['Defaulter'])
        self.y = self.df['Defaulter']

        #model object to predict

        predictor = pickle.load(open("test_bestmodel.pkl",'rb'))


        #predictions

        self.y_pred = predictor.predict(self.x)



   


    def score(self):
        # Accuracy
        accuracy = round(accuracy_score(self.y, self.y_pred), 2)
        
        # Precision
        precision = round(precision_score(self.y, self.y_pred), 2)
        
        # Recall
        recall = round(recall_score(self.y, self.y_pred), 2)
        
        # F1 Score
        f1 = round(f1_score(self.y, self.y_pred), 2)
        
        # Storing metrics in a dictionary
        self.score_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
        
        return self.score_dict
    
    def test(self):

        try:
            self.predict()
            self.score()
            fig, ax = plt.subplots()
            print(list(self.score_dict.keys()))
            print(list(self.score_dict.values()))

            ax.bar(list(self.score_dict.keys()), list(self.score_dict.values()) )

            ax.set_ylabel("Scores")
            ax.set_xlabel("Metrices")

            ax.set_title('Different Scoring metrics for model')
            plt.xticks(rotation = 'vertical')
            plt.savefig('metrices_bars.png')
            ## CI-cml input --> echo '![](./metrices_bars.png "Metrices Bar plot")' >> report.md


        except Exception as e:

            print(f"Error in plotting and predicting: {e}")

        else:
            print("plotted successfully")



if __name__== "__main__":

    tst = CI_test()
    tst.test()

        










