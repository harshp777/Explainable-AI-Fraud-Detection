import pandas as pd
import os
import pickle
from pathlib import Path
import sys
from utils import model_eval_classification
from jinja2 import Template
import yaml
from push_s3 import S3Push



class VisualizeScores:

    def __init__(self,trainpath, testpath, modelpath, features, reportfig_path, outputpath, home_dir):

        self.trainpath = trainpath
        self.testpath = testpath
        self.modelpath = modelpath
        self.features = features
        self.reportfig_path = reportfig_path
        self.outputpath = outputpath 
        self.home_dir = home_dir





    def get_data(self):

        try:
            self.df_train = pd.read_csv(self.trainpath)
            self.df_test = pd.read_csv(self.testpath)

        except Exception as e:

            print(f"Could not read the data due to the errors:{e}")

        else:
            print("Reading performed successfully")


    def get_features(self):
        try:

            self.df_train = self.df_train[self.features]
            self.df_test = self.df_test[self.features]

            
        except Exception as e:

            print("Could not fetch the featires due tp the error:{e}")

        else:
            print("Featuers fetched successfully")


        # Splitting into x and y
        self.x_train = self.df_train.drop(columns =['Defaulter'])
        self.y_train = self.df_train['Defaulter']
        self.x_test = self.df_test.drop(columns=['Defaulter'])
        self.y_test = self.df_test['Defaulter']


    def get_prediction(self):


        self.metrics_dict = {'Model':[], 'Train_Accuracy':[], 'Train_Precision':[], 'Train_Recall':[], 'Train_F1_Score':[],
                             'Test_Accuracy':[], 'Test_Precision':[], 'Test_Recall':[], 'Test_F1_Score':[]}
        
        try:

            for filename in os.listdir(self.modelpath):
                print(filename)
                if filename.endswith('pkl'):
                    filepath = os.path.join(self.modelpath, filename)
                    model = pickle.load(open(filepath, 'rb'))


                    # Evaluate model on training and testing data

                    train_score = model_eval_classification(model, self.x_train, self.y_train)
                    test_score = model_eval_classification(model,self.x_test, self.y_test)

                    self.metrics_dict['Model'].append(filename.split('_')[0][:-10])
                    self.metrics_dict['Train_Accuracy'].append(train_score['Accuracy'])
                    self.metrics_dict['Train_Precision'].append(train_score['Precision'])
                    self.metrics_dict['Train_Recall'].append(train_score['Recall'])
                    self.metrics_dict['Train_F1_Score'].append(train_score['F1 Score'])
  
                    self.metrics_dict['Test_Accuracy'].append(test_score['Accuracy'])
                    self.metrics_dict['Test_Precision'].append(test_score['Precision'])
                    self.metrics_dict['Test_Recall'].append(test_score['Recall'])
                    self.metrics_dict['Test_F1_Score'].append(test_score['F1 Score'])

                    print(self.metrics_dict)
        


        

        except Exception as e:

            print("Predict function failed with error:{e}")

        else:

            print("Scoring metrics for different models created successfully")


    def visualize_results(self):



        self.metrics_df = pd.DataFrame(self.metrics_dict)
        self.metrics_df.set_index('Model', inplace = True)


        model_idx = self.metrics_df[['Test_F1_Score']].idxmax()[0]
        print(model_idx)


        for filename in os.listdir(self.modelpath):
            if model_idx in filename:

                try:

                    filepath = os.path.join(self.modelpath, filename)
                    model = pickle.load(open(filepath, 'rb'))
                    wpath = Path(str(self.home_dir) + '/models/test_bestmodel.pkl')
                    pickle.dump(model, open(wpath, 'wb'))
                
                except Exception as e:

                    print(f"Model dumping failed with error: {e}")


                else:

                    print("Model dumped successfully")


                    try:

                        s3= S3Push()
                        s3.push(wpath, 'eai-fraud-data', 'test_bestmodel.pkl')

                    except Exception as e:
                        print(f"Failed in pushing a model to S3: {e}")

                    else:
                        print("Model pushed to S3 successfullly ")

                    


        # Specify the directory path
        directory_path = Path(str(self.reportfig_path) + '/scoring_metrices')

        # Create the directory if it doesn't exist
        directory_path.mkdir(parents=True, exist_ok=True)


        # Plot and save the graphs

        ax1 = self.metrics_df[['Train_Accuracy','Test_Accuracy']].plot.bar(title='Different model train and test Accuracy')
        fig1 = ax1.get_figure()
        fig1.savefig(Path(str(self.reportfig_path) + '/scoring_metrices/Different models train and test Accuracy.png') )

        ax2 = self.metrics_df[['Train_Precision','Test_Precision']].plot.bar(title='Different model train and test Precision')
        fig2 = ax2.get_figure()
        fig2.savefig(Path(str(self.reportfig_path) + '/scoring_metrices/Different models train and test Precision.png') )

        ax3 = self.metrics_df[['Train_Recall','Test_Recall']].plot.bar(title='Different model train and test Recall')
        fig3 = ax3.get_figure()
        fig3.savefig(Path(str(self.reportfig_path) + '/scoring_metrices/Different models train and test Recall.png') )


        ax4 = self.metrics_df[['Train_F1_Score','Test_F1_Score']].plot.bar(title='Different model train and test F1_Score')
        fig4 = ax4.get_figure()
        fig4.savefig(Path(str(self.reportfig_path) + '/scoring_metrices/Different models train and test F1_Score.png') )



    def generate_report(self):

        image_folder = Path(str(self.reportfig_path) + '/scoring_metrices')
        image_files = [file for file in os.listdir(image_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        # Create an HTML template
        template_str = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Report for different models</title>
        </head>
        <body>
            <h1>Comparative charts for Different scoring metrics of the models</h1>
            {% for image_file in image_files %}
                <div>
                    <img src="{{ image_folder }}\{{ image_file }}" alt="{{ image_file }}">
                    <br>
                </div>
            {% endfor %}
        </body>
        </html>
        """
        

        # Rendering HTML content 
        template = Template(template_str)
        html_content = template.render(image_folder=image_folder, image_files=image_files)


        # Save html content to a file

        report = Path(str(self.home_dir) +'/reports/metrics_report.html' )

        with open(report, 'w') as html_file:
            html_file.write(html_content)


        
    def metrics_tracker(self):


        self.get_data()
        self.get_features()
        self.get_prediction()
        self.visualize_results()
        self.generate_report()


    




train_input_filepath = sys.argv[1]
test_input_filepath = sys.argv[2]
model_path = sys.argv[3]
fig_path = sys.argv[4]



def main(train_input_filepath, test_input_filepath, model_path, fig_path):


    # Paths for input and output data
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_dir = Path(home_dir.as_posix() + '/data')
    output_path = Path(home_dir.as_posix() + '/dvclive')


    # Path for train and test data

    train_input_path = Path(data_dir.as_posix() + train_input_filepath)
    test_input_path = Path(data_dir.as_posix() + test_input_filepath)


    #model, report, and params path
    model_path = Path(home_dir.as_posix() + model_path)
    reportfig_path = Path(home_dir.as_posix() + fig_path)
    params_path = Path(home_dir.as_posix() + '/params.yaml')
    model_params = yaml.safe_load(open(params_path))['train_model']
    features = model_params['features']


    #initialising an object
    mtrcs = VisualizeScores(train_input_path, test_input_path, model_path, features,  reportfig_path,  output_path, home_dir)


    mtrcs.metrics_tracker()



if __name__=='__main__':
    main(train_input_filepath, test_input_filepath, model_path, fig_path)






