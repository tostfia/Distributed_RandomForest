FROM python:3.10-slim

WORKDIR /app

# Dipendenze di sistema necessarie a compilare psycopg2, e per la simulazione di rete (tc + setcap)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    iproute2 \
    libcap2-bin \
    sudo \
    && setcap cap_net_admin+ep /usr/sbin/tc \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .