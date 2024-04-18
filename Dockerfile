FROM python:3.12-slim

EXPOSE 8080
# Set the working directory to /app
WORKDIR /app

COPY ./Deployment/app.py /app/app.py
COPY test_bestmodel.pkl /app/test_bestmodel.pkl
COPY ./Deployment/features.yaml /app/features.yaml
COPY ./Deployment/predictor.py /app/predictor.py
COPY ./Deployment/requirements.txt /app/requirements.txt
COPY ./Deployment/static /app/static
COPY ./Deployment/templates /app/templates

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Corrected CMD syntax with square brackets
CMD ["python", "app.py"]