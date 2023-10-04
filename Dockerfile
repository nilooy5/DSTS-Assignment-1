# using official Python runtime base image
FROM python:3.8-slim

# Setting the working directory inside the container
WORKDIR /app

# Copying the files from the local host to the container
COPY . /app

# Installing the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Running the python script
CMD ["python", "main.py"]
