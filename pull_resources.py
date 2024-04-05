import boto3


try:

    s3 = boto3.client("s3")

    s3.download_file(

        Bucket = 'eai-fraud-data', Key='test_bestmodel.pkl', Filename='test_bestmodel.pkl' 
 
    )


except Exception as e:

    print(f'Not able to conect to s3: {e}')


else:

    print("Successfully loaded")