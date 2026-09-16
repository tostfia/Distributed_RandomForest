"""
Assemblaggio lazy del modello scikit-learn completo a partire dagli
artefatti di storage prodotti dal training distribuito.

CONTESTO:
- In modalità 'centralized' (centralized.py, _execute_training_step), al
  termine del training viene salvato solo un MANIFESTO leggero
  (n_estimators, classes_, ecc.) in saved_models/centralized/model_{job_id}.pkl
  -- non un modello scikit-learn completo. Gli alberi veri restano nelle
  parti di checkpoint (checkpoint_trees_{job_id}.pkl.part_0000.pkl, ...),
  pensate per essere lette in streaming durante l'inferenza distribuita
  (BaseOrchestrator._iter_checkpoint_trees), non per essere scaricate
  dall'utente finale.
- In modalità 'federated' (federated.py, _reconstruct_and_save_global_model),
  ogni albero viene salvato come file separato in
  saved_models/federated/model_{job_id}_trees/tree_XXXX.pkl, più un
  meta leggero in model_meta_{job_id}.pkl. Il vecchio path monolitico
  (model_{job_id}.pkl, gestito da _resolve_model_path) non viene più
  scritto dal training corrente: sopravvive solo come fallback di lettura
  per job addestrati con una versione precedente del codice, quando era
  ancora un pickle scikit-learn autosufficiente.

Questo modulo NON tocca nessuno dei due formati di storage: si limita a
LEGGERLI (via la stessa CheckpointDAOFactory usata dagli orchestratori) e a
ricomporre, solo quando serve (cioè quando l'utente chiede il download), un
oggetto RandomForestClassifier/RandomForestRegressor scikit-learn vero e
proprio, con .estimators_ popolato -- direttamente utilizzabile con
model.predict(X) in un ambiente locale, come richiesto dal requisito
opzionale di download della traccia.

I resolver di path qui sotto DUPLICANO intenzionalmente quelli privati
delle classi CentralizedOrchestrator/FederatedOrchestrator: il client
(main.py) non istanzia un orchestratore (niente AWS services, niente
connessioni RPyC), quindi non può riusare direttamente quei metodi. Se le
convenzioni di naming dei path cambiano in centralized.py o federated.py,
vanno aggiornate anche qui.
"""

import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

BUCKET_NAME = os.environ.get(
    "DATASETS_BUCKET_NAME", "rf-distributed-datasets-383056860320-us-east-1"
)


# --------------------------------------------------------------------------
# Resolver di path (duplicati da centralized.py / federated.py, vedi docstring)
# --------------------------------------------------------------------------

def _centralized_model_path(job_id: str, environment: str) -> str:
    if environment == "aws":
        return f"s3://{BUCKET_NAME}/saved_models/centralized/model_{job_id}.pkl"
    return os.path.join("./saved_models", f"model_{job_id}.pkl")


def _centralized_trees_checkpoint_path(job_id: str, environment: str) -> str:
    if environment == "aws":
        return f"s3://{BUCKET_NAME}/checkpoints/checkpoint_trees_{job_id}.pkl"
    return f"./.local_storage/checkpoint_trees_{job_id}.pkl"


def _federated_model_dir(job_id: str, environment: str) -> str:
    if environment == "aws":
        return f"s3://{BUCKET_NAME}/saved_models/federated/model_{job_id}_trees"
    return os.path.join("./saved_models", f"model_{job_id}_trees")


def _federated_meta_path(job_id: str, environment: str) -> str:
    if environment == "aws":
        return f"s3://{BUCKET_NAME}/saved_models/federated/model_meta_{job_id}.pkl"
    return os.path.join("./saved_models", f"model_meta_{job_id}.pkl")


def _federated_legacy_model_path(job_id: str, environment: str) -> str:
    """Vecchio path monolitico, non più scritto dal training corrente (vedi
    docstring del modulo): letto solo come fallback per job pre-fix."""
    if environment == "aws":
        return f"s3://{BUCKET_NAME}/saved_models/federated/model_{job_id}.pkl"
    return os.path.join("./saved_models", f"model_{job_id}.pkl")


# --------------------------------------------------------------------------
# Lettura in streaming delle parti (stessa logica di
# BaseOrchestrator._iter_trees_checkpoint_parts / _iter_checkpoint_trees,
# reimplementata qui perché quella vive su un'istanza di orchestratore)
# --------------------------------------------------------------------------

def _iter_centralized_tree_parts(job_id: str, checkpoint_dao, environment: str):
    base = _centralized_trees_checkpoint_path(job_id, environment)
    stem = base[: -len(".pkl")] if base.endswith(".pkl") else base

    index = 0
    yielded_any = False
    while True:
        part_path = f"{stem}.part_{index:04d}.pkl"
        if not checkpoint_dao.exists(part_path):
            break
        yielded_any = True
        for tree in checkpoint_dao.load(part_path):
            yield tree
        index += 1

    if not yielded_any and checkpoint_dao.exists(base):
        # Fallback per job pre-migrazione al formato a parti: un'unica lista
        # di alberi salvata sul path monolitico (vedi _persist_trees_delta).
        for tree in checkpoint_dao.load(base):
            yield tree


# --------------------------------------------------------------------------
# Costruzione dell'oggetto scikit-learn
# --------------------------------------------------------------------------

def _build_forest_from_trees(trees, tree_type, classes_list=None, n_features_in_=None):
    if not trees:
        raise ValueError(
            "Nessun albero disponibile: impossibile assemblare il modello "
            "(checkpoint/parti mancanti o vuoti)."
        )

    forest_cls = RandomForestClassifier if tree_type == "classifier" else RandomForestRegressor
    forest = forest_cls(n_estimators=len(trees))

    forest.estimators_ = trees
    forest.n_estimators = len(trees)
    forest.n_features_in_ = (
        int(n_features_in_) if n_features_in_ is not None else int(trees[0].n_features_in_)
    )
    forest.n_outputs_ = 1
    # Template richiesto da check_is_fitted/dalla clonazione interna in
    # alcune versioni di scikit-learn: non viene mai rifittato, serve solo
    # come riferimento di struttura.
    forest.estimator_ = trees[0]

    if tree_type == "classifier":
        if classes_list:
            classes_arr = np.array(classes_list, dtype=np.int64)
        else:
            # Fallback: unione delle classi viste dai singoli alberi.
            classes_arr = np.unique(
                np.concatenate([np.asarray(t.classes_) for t in trees if hasattr(t, "classes_")])
            )
        forest.classes_ = classes_arr
        forest.n_classes_ = len(classes_arr)

    return forest


# --------------------------------------------------------------------------
# Controlli di esistenza "leggeri" (un solo checkpoint_dao.exists(), senza
# caricare/assemblare nulla) -- usati da handle_model_request() in main.py
# per il check preliminare prima di offrire il download, così quel check usa
# la stessa nozione di "esiste" dell'assemblaggio vero e proprio, invece del
# vecchio os.path.exists('model_{job_id}.pkl') che per i job federated non è
# mai vero.
# --------------------------------------------------------------------------

def centralized_artifact_exists(job_id: str, checkpoint_dao, environment: str) -> bool:
    return checkpoint_dao.exists(_centralized_model_path(job_id, environment))


def federated_artifact_exists(job_id: str, checkpoint_dao, environment: str) -> bool:
    return checkpoint_dao.exists(
        _federated_meta_path(job_id, environment)
    ) or checkpoint_dao.exists(_federated_legacy_model_path(job_id, environment))


def model_artifact_exists(job_id: str, checkpoint_dao, environment: str, training_mode: str) -> bool:
    if training_mode == "federated":
        return federated_artifact_exists(job_id, checkpoint_dao, environment)
    return centralized_artifact_exists(job_id, checkpoint_dao, environment)


# --------------------------------------------------------------------------
# API pubblica usata da main.py
# --------------------------------------------------------------------------

def assemble_centralized_forest(job_id: str, checkpoint_dao, environment: str):
    """Ricostruisce un RandomForestClassifier/Regressor scikit-learn completo
    per un job addestrato in modalità centralized, leggendo il manifesto per
    i metadati e le parti di checkpoint per gli alberi veri."""
    model_path = _centralized_model_path(job_id, environment)
    if not checkpoint_dao.exists(model_path):
        raise FileNotFoundError(f"Modello globale non trovato in '{model_path}'.")

    loaded = checkpoint_dao.load(model_path)
    if hasattr(loaded, "estimators_"):
        # Job addestrato PRIMA dell'introduzione del manifesto leggero: qui
        # 'loaded' è già un RandomForest scikit-learn autosufficiente.
        return loaded

    manifest = loaded
    tree_type = manifest.get("tree_type", "classifier")
    expected_n_estimators = manifest.get("n_estimators")

    trees = list(_iter_centralized_tree_parts(job_id, checkpoint_dao, environment))
    if expected_n_estimators is not None and len(trees) != expected_n_estimators:
        print(
            f"[model_assembly] [WARN] Il manifesto dichiara {expected_n_estimators} "
            f"alberi, ma dalle parti di checkpoint ne sono stati letti {len(trees)}. "
            f"Il modello assemblato userà solo quelli effettivamente trovati."
        )

    return _build_forest_from_trees(
        trees,
        tree_type=tree_type,
        classes_list=manifest.get("classes_"),
        n_features_in_=manifest.get("n_features_in_"),
    )


def assemble_federated_forest(job_id: str, checkpoint_dao, environment: str):
    """Ricostruisce un RandomForestClassifier/Regressor scikit-learn completo
    per un job addestrato in modalità federated, leggendo i metadati leggeri
    e gli alberi separati (un file per albero)."""
    meta_path = _federated_meta_path(job_id, environment)

    if checkpoint_dao.exists(meta_path):
        meta = checkpoint_dao.load(meta_path)
        model_dir = meta.get("model_dir") or _federated_model_dir(job_id, environment)
        num_trees = meta["num_trees"]

        trees = []
        for i in range(num_trees):
            tree_path = os.path.join(model_dir, f"tree_{i:04d}.pkl")
            if not checkpoint_dao.exists(tree_path):
                print(f"[model_assembly] [WARN] Albero mancante: '{tree_path}' (saltato).")
                continue
            trees.append(checkpoint_dao.load(tree_path))

        return _build_forest_from_trees(
            trees,
            tree_type=meta.get("tree_type", "classifier"),
            classes_list=meta.get("classes"),
            n_features_in_=None,  # non presente nel meta federato: dedotto dagli alberi
        )

    # Fallback per job addestrati prima dell'introduzione degli alberi
    # separati (vedi _resolve_model_path in federated.py, non più scritto
    # dal training corrente ma ancora letto qui per compatibilità).
    legacy_path = _federated_legacy_model_path(job_id, environment)
    if checkpoint_dao.exists(legacy_path):
        loaded = checkpoint_dao.load(legacy_path)
        if hasattr(loaded, "estimators_"):
            return loaded

    raise FileNotFoundError(
        f"Nessun modello trovato per il job '{job_id}'. Cercati: metadati "
        f"'{meta_path}' (modalità ad alberi separati) o file monolitico "
        f"legacy '{legacy_path}'."
    )