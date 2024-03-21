import boto3
from pathlib import Path



class S3Push:

    def __init__(self):

        try:
            self.s3 = boto3.client("s3")

        except Exception as e:

            print(f"Not able to connect to S3: {e}")



    def push(self, file, bucket, name):

        try:
            self.s3.upload_file(file, bucket, name)

        except Exception as e:

            print(f"Not able to upload to S3: {e}")


        
if __name__=='__main__':

    path = Path(__file__)
    home_dir = path.parent
    modelpath = Path(str(home_dir) + '/models/test_bestmodel.pkl')


    try:
        s3 = S3Push()
        s3.push(modelpath, 'eai-fraud-data', 'bestmodel.pkl')

    except Exception as e:
        print(f"Failed to push a model to s3: {e}")

    else:
        print("Successfully pushed the model to S3")




