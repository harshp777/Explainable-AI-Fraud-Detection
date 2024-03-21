from flask import Flask, render_template
from predictor import DefaultPredictor


app = Flask(__name__)



@app.route('/')

def index():

    try:
        return render_template('inedx.html')

    except Exception as e:
        print(f"Home page has not been loaded with with error:", e)



@app.route('/predict', method = ['POST'])

def predict():

    form_data = dict()

    try:

        is_defaulter = DefaultPredictor()
        pred = is_defaulter.predict_defaulter(form_data)
        if pred==0:
            answer= 'Not a Defaulter'
        else:
            answer= 'defaulter'
        
        return render_template('predict.html', pred = answer)
    

    except Exception as e:

        print(f"Failed with error: {e}")
        


if __name__=='__main__':

    try:
        app.run(host='127.0.0.0', port =8000)

    except Exception as e:
        print(f"Application failure: {e}")





