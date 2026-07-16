# =====================================================================
# STAGE 1: builder — qui installiamo tutto ciò che serve per compilare
# (gcc, libpq-dev, ecc). Questo stage NON finisce nell'immagine finale.
# =====================================================================
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Ambiente virtuale isolato: più facile da copiare "in blocco" nello stage finale
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .

# --prefer-binary: forza pip a preferire le wheel precompilate quando disponibili,
# evitando build da sorgente (spesso la causa principale di layer enormi)
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# =====================================================================
# STAGE 2: immagine finale — solo runtime, niente compilatori
# =====================================================================
FROM python:3.10-slim

# Solo le librerie RUNTIME (non i -dev/headers usati per compilare):
# - libpq5: libreria runtime di postgres (non libpq-dev, che sono solo header)
# - iproute2/libcap2-bin: servono per tc netem (solo uso locale, ma restano
#   nell'immagine visto che è condivisa da tutti i servizi del compose)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    iproute2 \
    libcap2-bin \
    sudo \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiamo l'intero virtualenv già pronto dal builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app

WORKDIR /app
COPY . .