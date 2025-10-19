FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

# Install Python dependencies needed by the MLflow server
RUN pip install mlflow psycopg2-binary boto3

# Install PostgreSQL client tools for migrations and connectivity checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends postgresql-client && \
    rm -rf /var/lib/apt/lists/*

# Expose MLflow UI port
EXPOSE 8888

# Start MLflow server with artifact serving enabled
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "8888", "--serve-artifacts"]
