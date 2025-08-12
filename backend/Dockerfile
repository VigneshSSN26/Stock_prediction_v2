# Use the official Python 3.10 slim-buster image as a base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
# Using --no-cache-dir reduces the image size
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your application code into the container
COPY . /code/

# Expose the port that Hugging Face expects (7860)
EXPOSE 7860

# Command to run the application using Gunicorn
# This runs your Flask app. Make sure your file is 'app.py' and the Flask variable is 'app'.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers=1", "app:app"]
