FROM python:3.10-slim

WORKDIR /app

# Dipendenze di sistema necessarie a compilare psycopg2
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app