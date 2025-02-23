FROM python:3.12-slim

# Install CA certificates
RUN apt-get update && apt-get install -y \
    gcc \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY service_bus_handler.py .
COPY prompts.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
