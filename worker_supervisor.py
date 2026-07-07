"""
Supervisore di processo "restart: on-failure"-like, portabile su locale/Docker/AWS.

Perché esiste
-------------
In Docker Compose la direttiva `restart: on-failure` è gestita dal Docker Engine:
se il processo nel container esce con codice != 0, Docker lo rilancia.
Fuori da Docker (es. in locale, o su una EC2 "nuda" senza systemd) non esiste
nessun meccanismo equivalente: se il worker crasha, resta morto finché qualcuno
non lo rilancia a mano.

Questo script è un wrapper minimale che replica ESATTAMENTE la stessa semantica:
- exit code 0  -> il processo si è fermato volontariamente, NON viene riavviato
- exit code != 0 -> crash, viene riavviato dopo un backoff
- Ctrl+C / SIGTERM -> propagato al figlio, poi il supervisore esce pulito

Cosi il comportamento è identico sia che tu lanci il worker:
  - dentro un container Docker con `restart: on-failure`
  - in locale con `python worker_supervisor.py`
  - (concettualmente) come task ECS/Fargate con la sua policy di restart nativa

Uso
---
    python worker_supervisor.py -- python -m src.worker.federatedWorker

Le variabili d'ambiente (NUM_WORKERS, WORKER_NAME, ecc.) vengono ereditate
automaticamente dal supervisore al processo figlio, quindi puoi lanciare più
istanze in parallelo (una per shell/terminale) esattamente come faresti con
`docker compose up --scale worker=3`.

Configurazione (variabili d'ambiente):
    FED_SUPERVISOR_MAX_RESTARTS   numero massimo di riavvii (0 = infiniti, default 0)
    FED_SUPERVISOR_BACKOFF_SECONDS  attesa base tra un riavvio e l'altro (default 3)
    FED_SUPERVISOR_BACKOFF_MAX_SECONDS  tetto del backoff esponenziale (default 60)
"""
import os
import subprocess
import sys
import time
import signal


def main():
    if "--" not in sys.argv:
        print("Uso: python worker_supervisor.py -- <comando> [args...]")
        sys.exit(2)

    split_idx = sys.argv.index("--")
    command = sys.argv[split_idx + 1:]
    if not command:
        print("Nessun comando specificato dopo '--'.")
        sys.exit(2)

    max_restarts = int(os.environ.get("FED_SUPERVISOR_MAX_RESTARTS", "0"))          # 0 = infiniti
    backoff_base = float(os.environ.get("FED_SUPERVISOR_BACKOFF_SECONDS", "3"))
    backoff_max = float(os.environ.get("FED_SUPERVISOR_BACKOFF_MAX_SECONDS", "60"))

    attempt = 0
    current_proc = {"p": None}

    def _forward_signal(signum, _frame):
        # Propaga SIGTERM/SIGINT al processo figlio corrente e poi esce.
        p = current_proc["p"]
        if p and p.poll() is None:
            print(f"[SUPERVISOR] Ricevuto segnale {signum}, termino il processo figlio (pid={p.pid})...")
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)

    while True:
        attempt += 1
        print(f"[SUPERVISOR] Avvio tentativo #{attempt}: {' '.join(command)}")
        proc = subprocess.Popen(command)
        current_proc["p"] = proc
        exit_code = proc.wait()

        if exit_code == 0:
            print("[SUPERVISOR] Processo terminato con exit code 0 (stop volontario). Nessun riavvio.")
            sys.exit(0)

        print(f"[SUPERVISOR] Processo terminato con exit code {exit_code} (crash).")

        if max_restarts and attempt > max_restarts:
            print(f"[SUPERVISOR] Raggiunto il numero massimo di riavvii ({max_restarts}). Mi fermo.")
            sys.exit(exit_code)

        wait_time = min(backoff_base * (2 ** (attempt - 1)), backoff_max)
        print(f"[SUPERVISOR] Riavvio tra {wait_time:.1f}s...")
        time.sleep(wait_time)


if __name__ == "__main__":
    main()