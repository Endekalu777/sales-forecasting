# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the files into the container
COPY app/ ./app/  
COPY templates/ ./templates/  
COPY artifacts/ ./artifacts/  
COPY requirements.txt .

# Set the Flask environment variable
ENV FLASK_APP=app/app.py  

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run your Flask app when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
