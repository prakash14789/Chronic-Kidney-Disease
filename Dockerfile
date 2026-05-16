# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# libgomp1 is often required by LightGBM
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for data and models if they don't exist
RUN mkdir -p /app/data /app/saved_models /app/reports

# Expose ports
EXPOSE 8501 8000

# The command is overridden by docker-compose, but we provide a default
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
