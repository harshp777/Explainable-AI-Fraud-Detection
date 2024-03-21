from flask import Flask, render_template, request
from predictor import DefaultPredictor


app = Flask(__name__)



@app.route('/')
def index():

    try:
        return render_template('index.html')

    except Exception as e:
        print(f"Home page has not been loaded with with error:", e)



@app.route('/predict', methods = ['POST'])
def predict():

    form_data = dict()
    form_data['Amount'] = request.form.get('Amount', type=float)
    form_data['Interest Rate'] = request.form.get('Interest Rate', type=float)
    form_data['Tenure(years)'] = request.form.get('Tenure(years)', type=int)
    form_data['Tier of Employment'] = request.form.get('Tier of Employment')
    form_data['Work Experience'] = request.form.get('Work Experience')
    form_data['Total Income(PA)'] = request.form.get('Total Income(PA)', type=float)
    form_data['Dependents'] = request.form.get('Dependents', type=int)
    form_data['Delinq_2yrs'] = request.form.get('Delinq_2yrs', type=int)
    form_data['Total Payement '] = request.form.get('Total Payment', type=float)
    form_data['Received Principal'] = request.form.get('Received Principal', type=float)
    form_data['Interest Received'] = request.form.get('Interest Received', type=float)
    form_data['Number of loans'] = request.form.get('Number of loans', type=int)
    form_data['Gender'] = request.form.get('Gender')
    form_data['Married'] = request.form.get('Married')
    form_data['Home'] = request.form.get('Home')
    form_data['Social Profile'] = request.form.get('Social Profile')
    form_data['Loan Category'] = request.form.get('Loan Category')
    form_data['Employmet type'] = request.form.get('Employment type')
    form_data['Is_verified'] = request.form.get('Is_verified')


    try:

        is_defaulter = DefaultPredictor()
        pred = is_defaulter.predict_defaulter(form_data)
        if pred==0:
            answer= 'Not a Defaulter'
        else:
            answer= 'Defaulter'
        
        return render_template('predict.html', pred = answer)
    

    except Exception as e:

        print(f"Failed with error: {e}")
        


if __name__=='__main__':

    try:
        app.run(host='0.0.0.0', port = 8000, debug=True)

    except Exception as e:
        print(f"Application failure: {e}")





