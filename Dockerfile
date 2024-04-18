FROM python:3.12-slim

EXPOSE 8080
# Set the working directory to /app
WORKDIR /app

COPY ./deployment/app.py /app/app.py
COPY test_bestmodel.pkl /app/test_bestmodel.pkl
COPY ./deployment/features.yaml /app/features.yaml
COPY ./deployment/predictor.py /app/predictor.py
COPY ./deployment/requirements.txt /app/requirements.txt
COPY ./deployment/static /app/static
COPY ./deployment/templates /app/templates

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Corrected CMD syntax with square brackets
CMD ["python", "app.py"]