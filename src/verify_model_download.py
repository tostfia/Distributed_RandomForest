"""
Verifica rapida del fix di download del modello.

Uso:
    python verify_model_download.py <job_id> [--mode centralized|federated]

Se --mode e' omesso, usa cfg.mode (stessa logica di fallback di
download_model() in main.py quando il job non e' in requests_history.json).
"""
import argparse
import os
import pickle

import numpy as np

from src.shared.config import SystemConfig
from src.dataset.checkpoint_dao import CheckpointDAOFactory
from src.shared.utilities.model_assembly import (
    assemble_centralized_forest,
    assemble_federated_forest,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("--mode", choices=["centralized", "federated"], default=None)
    args = parser.parse_args()

    cfg = SystemConfig()
    checkpoint_dao = CheckpointDAOFactory.get_dao(cfg.env)
    mode = args.mode or cfg.mode

    print(f"[TEST] Ambiente: {cfg.env} | Modalità: {mode} | Job: {args.job_id}")

    assemble_fn = assemble_federated_forest if mode == "federated" else assemble_centralized_forest
    forest = assemble_fn(args.job_id, checkpoint_dao, cfg.env)

    print(f"[TEST] Tipo oggetto: {type(forest).__name__}")
    print(f"[TEST] n_estimators: {forest.n_estimators}")
    print(f"[TEST] n_features_in_: {forest.n_features_in_}")
    if hasattr(forest, "classes_"):
        print(f"[TEST] classes_: {forest.classes_}")

    # Round-trip pickle: replica esattamente cosa succede a un utente che
    # scarica il file e lo riapre in un ambiente pulito con solo scikit-learn.
    tmp_path = "/tmp/_verify_model.pkl"
    with open(tmp_path, "wb") as f:
        pickle.dump(forest, f)
    with open(tmp_path, "rb") as f:
        reloaded = pickle.load(f)
    os.remove(tmp_path)

    # Non e' un test di accuratezza (per quello confrontare con le metriche
    # dell'inferenza distribuita): serve solo a verificare che predict() non
    # sollevi eccezioni su un modello davvero autosufficiente.
    X_dummy = np.random.rand(5, reloaded.n_features_in_)
    preds = reloaded.predict(X_dummy)
    print(f"[TEST] predict() su input fittizio: OK -> {preds}")

    if hasattr(reloaded, "predict_proba"):
        probs = reloaded.predict_proba(X_dummy)
        print(f"[TEST] predict_proba() shape: {probs.shape}")

    print("\n[TEST] Il modello assemblato è un oggetto scikit-learn valido e utilizzabile in locale.")


if __name__ == "__main__":
    main()