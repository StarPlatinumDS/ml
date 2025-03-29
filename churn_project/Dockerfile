# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only necessary parts
COPY requirements.txt .
COPY api/ ./api/
COPY models/ ./models/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Start API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
