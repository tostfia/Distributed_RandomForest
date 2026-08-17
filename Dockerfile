# =====================================================================
# STAGE 1: builder — qui installiamo tutto ciò che serve per compilare
# (gcc, libpq-dev, ecc). Questo stage NON finisce nell'immagine finale.
# =====================================================================
FROM python:3.10-slim AS builder

# Forziamo apt a usare HTTPS verso i mirror Debian: su reti con proxy/firewall
# che ispezionano o alterano il traffico HTTP in chiaro, il file InRelease può
# arrivare corrotto ("Bad header line Bad header data"), causando il fallimento
# di apt-get update. Copre sia il vecchio formato (sources.list) sia il nuovo
# formato deb822 (sources.list.d/*.sources) usato dalle immagini basate su trixie.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' \
    /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true

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

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g; s|http://security.debian.org|https://security.debian.org|g' \
    /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    iproute2 \
    libcap2-bin \
    sudo \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN setcap cap_net_admin+ep $(which tc)

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app

WORKDIR /app
COPY . .