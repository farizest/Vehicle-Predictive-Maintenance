# Use an official Python runtime as a parent image (slim version to save space)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    TF_CPP_MIN_LOG_LEVEL=2

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (clean up afterwards to keep image size small)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies natively
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files to the container
COPY . .

# Expose the Cloud Run port
EXPOSE 8080

# Run the FastAPI server using Uvicorn
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8080"]
