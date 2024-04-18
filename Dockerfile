FROM python:3.12-slim

EXPOSE 8080
# Set the working directory to /app
WORKDIR /app

COPY ./deployement/app.py /app/app.py
COPY test_bestmodel.pkl /app/test_bestmodel.pkl
COPY ./deployement/features.yaml /app/features.yaml
COPY ./deployement/predictor.py /app/predictor.py
COPY ./deployement/requirements.txt /app/requirements.txt
COPY ./deployement/static /app/static
COPY ./deployement/templates /app/templates

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Corrected CMD syntax with square brackets
CMD ["python", "app.py"]