#!/usr/bin/env python3
"""
Estrae o ripristina la sola sezione 'baseline_boot' di .local_storage/config.json.

Usato da cleanup.sh per far sopravvivere la boot configuration della baseline
(dataset_type, tree_type) attraverso il reset completo di .local_storage/,
mentre tutto il resto (in particolare 'last_training_request' e lo storico
delle richieste) viene comunque azzerato come da comportamento attuale.

Uso:
    preserve_baseline_boot.py save <config_path> <tmp_path>
    preserve_baseline_boot.py restore <config_path> <tmp_path>

'save' non fallisce mai (exit 0) se il file non esiste o la sezione non è
presente: in quel caso semplicemente non c'è nulla da preservare.
'restore' non fallisce mai se il file temporaneo non esiste: significa che
'save' non aveva trovato nulla da salvare.
"""
import json
import os
import sys


def save(config_path: str, tmp_path: str) -> None:
    if not os.path.exists(config_path):
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return

    if not isinstance(data, dict) or "baseline_boot" not in data:
        return

    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"baseline_boot": data["baseline_boot"]}, f, indent=2)


def restore(config_path: str, tmp_path: str) -> None:
    # Nota: se tmp_path è stato creato con 'mktemp', il file esiste già ma è
    # vuoto finché 'save' non ci scrive dentro. Un file assente o vuoto
    # significa entrambi "nessuna sezione da preservare": non è un errore.
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(tmp_path, "r", encoding="utf-8") as f:
        preserved = json.load(f)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(preserved, f, indent=2)

    os.remove(tmp_path)


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in ("save", "restore"):
        print("Uso: preserve_baseline_boot.py <save|restore> <config_path> <tmp_path>", file=sys.stderr)
        sys.exit(1)

    action, config_path, tmp_path = sys.argv[1], sys.argv[2], sys.argv[3]
    if action == "save":
        save(config_path, tmp_path)
    else:
        restore(config_path, tmp_path)


if __name__ == "__main__":
    main()