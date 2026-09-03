"""
Modulo UNICO per la giustificazione empirica di n_estimators, sia per la
classificazione (dataset reale CICIDS) sia per la regressione (dataset
sintetico) -- sostituisce analyze_classification_n_estimators.py e
consolida in un solo posto una logica prima sparsa su tre livelli diversi
di run_baseline.py (Optuna categorico dentro il tuning, un secondo Optuna
GridSampler di raffinamento senza warm_start, una scelta interattiva con
soglia di tolleranza) più questo stesso script per la sola classificazione.

RUOLO NELLA PIPELINE -- SOLA DIAGNOSTICA, MAI NEL HOT PATH:
    Questo script NON tocca tuning né feature selection. Li dà per fatti:
      - classificazione: legge gli iperparametri (incluso max_features, che
        determina la feature selection) da 'outputs_baseline/config_real.json',
        prodotto da run_baseline.py -- optuna_oob_hyperparameter_search;
      - regressione: legge gli iperparametri da
        'outputs_baseline/config_synthetic_regressor.json' -- di default
        sklearn (nessun tuning: il sintetico serve solo da stress-test di
        scalabilità, non a massimizzare l'R² -- vedi REGRESSOR_DEFAULT_HP in
        run_baseline.py), tranne n_estimators, che è ESATTAMENTE il valore
        che QUESTO script calcola: run_baseline.py lo rilegge da qui al giro
        successivo.
    Fa crescere SOLO n_estimators via warm_start, a parità di tutto il
    resto, e produce grafico + tabella. Il criterio di scelta resta la
    lettura VISIVA della curva (vedi sotto): questo script non scrive né
    sovrascrive alcun manifesto (l'aggiornamento di n_estimators nel
    manifesto, dopo aver letto il grafico, è manuale).

METODOLOGIA -- fedele all'esempio ufficiale scikit-learn:
    "OOB Errors for Random Forests"
    https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
    (The scikit-learn developers, BSD-3-Clause)

    Una sola foresta cresce incrementalmente con warm_start=True (si
    aggiungono alberi via via), non tante foreste indipendenti create da
    zero per ogni n_estimators: con fit indipendenti lo schema di seeding
    interno di scikit-learn assegna semi diversi ai singoli alberi a
    seconda del numero totale di stimatori richiesto, quindi la curva
    confronterebbe foreste leggermente diverse punto per punto, non la
    stessa foresta osservata a stadi di crescita diversi.

    UNA SOLA CURVA per task (non un confronto tra più configurazioni come
    nell'esempio ufficiale, che affianca diversi max_features): qui gli
    altri iperparametri sono già stati scelti altrove (tuning OOB per la
    classificazione, configurazione dichiarata a priori per la
    regressione) -- l'obiettivo di questo script è raffinare/giustificare
    solo n_estimators a valle di quella scelta, non riaprire lo spazio
    degli altri iperparametri.

    METRICA -- OOB error rate = 1 - oob_score_, l'attributo NATIVO di
    scikit-learn, esattamente come nell'esempio ufficiale (che per la
    classificazione traccia `1 - clf.oob_score_`). Per la regressione,
    dove oob_score_ è l'R² OOB, la stessa formula `1 - oob_score_`
    fornisce una metrica di errore comparabile (frazione di varianza OOB
    non spiegata), usata qui per coerenza di stile con la classificazione.

    LETTURA: il criterio primario per scegliere n_estimators resta la
    lettura VISIVA del grafico (dove la curva si appiattisce). Nessun
    algoritmo di knee-detection automatico (rimosso in una versione
    precedente di questo script per la classificazione: il Kneedle
    semplificato segnala sistematicamente un ginocchio troppo basso su
    curve che saturano rapidamente).

LIMITAZIONE DICHIARATA -- bias a n_estimators basso:
    `oob_score_` nativo di scikit-learn (verificato empiricamente su
    1.6.1) valuta anche le righe che non sono MAI out-of-bag per nessun
    albero, usando per esse un valore di riempimento di default invece di
    escluderle:
      - CLASSIFICAZIONE: oob_decision_function_[riga] = [0, 0, ...] per le
        righe mai OOB; l'argmax su un vettore di soli zeri restituisce
        sempre la prima classe (indice 0), quindi quelle righe vengono
        contate come classificate in classe 0 indipendentemente dalla
        vera etichetta;
      - REGRESSIONE: oob_prediction_[riga] = 0.0 per le righe mai OOB,
        confrontato con il vero target nel calcolo di oob_score_ (R²).
    Questo introduce un errore sistematico (non casuale) più marcato ai
    valori BASSI di n_estimators nella griglia, dove più righe non sono
    ancora mai state OOB per nessun albero -- cioè proprio nella parte
    della curva più delicata per stimare quanti alberi bastano. Scelta
    consapevole di questo script: dare priorità alla fedeltà diretta
    all'esempio ufficiale scikit-learn piuttosto che a una correzione
    manuale del bias (rimossa in una versione precedente di questo
    modulo, che ricostruiva la copertura OOB riga per riga). Da citare
    esplicitamente in relazione come limite noto della stima ai valori
    più bassi della griglia.

Uso:
    python -m src.baseline.analyze_n_estimators --task classifier
    python -m src.baseline.analyze_n_estimators --task regressor
    python -m src.baseline.analyze_n_estimators --task both   (default)
"""
import os
import json
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.datasplitter import StratifiedDataSplitter
from src.shared.utilities.featureselection import CICIDSFeatureSelector
from src.shared.utilities.undersampling import undersample_majority_class

RANDOM_SEED = 123
TEST_SIZE = 0.2

MIN_ESTIMATORS = 5
MAX_ESTIMATORS = 200
STEP_ESTIMATORS = 5
DEFAULT_GRID = list(range(MIN_ESTIMATORS, MAX_ESTIMATORS + 1, STEP_ESTIMATORS))

# ---------------------------------------------------------------------------
# CLASSIFICAZIONE -- dataset reale CICIDS
# ---------------------------------------------------------------------------
CLF_TARGET_COL = "Label"
CLF_TARGET_ROWS_PER_DAY = 100_000  # stesso valore di run_baseline.py
CLF_UNDERSAMPLING_RATIO = 1.0
CLF_CONFIG_PATH = os.path.join("outputs_baseline", "config_real.json")
CLF_DIAGNOSTIC_SUBSAMPLE_SIZE = 100_000
# Stessi valori di run_baseline.py (MULTICOLLINEARITY_DISTANCE_THRESHOLD e
# VALIDATION_SIZE_FOR_THRESHOLD): se li cambi lì, cambiali identici anche
# qui, altrimenti il train set su cui si misura la curva OOB non è più lo
# stesso che run_baseline.py userebbe davvero per l'addestramento finale.
CLF_MULTICOLLINEARITY_DISTANCE_THRESHOLD = 0.2
CLF_VALIDATION_SIZE_FOR_THRESHOLD = 0.15

# ---------------------------------------------------------------------------
# REGRESSIONE -- dataset sintetico
#
# Iperparametri: default sklearn tranne n_estimators (vedi REGRESSOR_DEFAULT_HP
# in run_baseline.py) -- nessun tuning, il sintetico serve solo da stress-test
# di scalabilità. Persistiti in REG_CONFIG_PATH, questo script si limita a
# leggerli.
#
# RICETTA DEL DATASET: letta anch'essa da REG_CONFIG_PATH (n_samples,
# n_features, noise), NON ridichiarata qui con valori più piccoli. La curva
# OOB va misurata sul dataset di PRODUZIONE esatto: un sottocampione
# diagnostico più piccolo darebbe una stima di n_estimators non trasferibile
# al modello che verrà davvero usato nei test di scalabilità -- stesso motivo
# per cui il tuning stesso è stato tolto dal percorso "dataset più piccolo".
# ---------------------------------------------------------------------------
REG_TARGET_COL = "Target"
# Stesso file già scritto da run_baseline.py per OGNI run sintetico di
# regressione (copia per-task, non un file dedicato al tuning): niente
# artefatto aggiuntivo da tenere sincronizzato con chi consuma già
# outputs_baseline/.
REG_CONFIG_PATH = os.path.join("outputs_baseline", "config_synthetic_regressor.json")


# ---------------------------------------------------------------------------
# Caricamento dati
# ---------------------------------------------------------------------------
def load_tuned_classifier_hp():
    """
    Legge gli iperparametri TUNATI (tranne n_estimators, che qui è la
    variabile indipendente) da config_real.json. Non decide nulla: si
    limita a leggere cosa optuna_oob_hyperparameter_search ha già scelto.
    """
    if not os.path.exists(CLF_CONFIG_PATH):
        raise FileNotFoundError(
            f"'{CLF_CONFIG_PATH}' non trovato: esegui prima run_baseline.py "
            f"(con tuning, non SKIP_TUNING) almeno una volta."
        )
    with open(CLF_CONFIG_PATH, "r") as f:
        config = json.load(f)
    hp = config.get("hyperparameters")
    if not hp:
        raise ValueError(f"'{CLF_CONFIG_PATH}' trovato ma senza sezione 'hyperparameters'.")
    print(f"[INFO] Iperparametri tunati letti da '{CLF_CONFIG_PATH}': {hp}")
    return hp


def prepare_classifier_dataset(tuned_hp):
    """
    Ricostruisce lo STESSO train set che run_baseline.py userebbe con questi
    iperparametri: campionamento ribilanciato per giorno, binarizzazione,
    split, preprocessing, split del validation set per la calibrazione
    della soglia (scartato qui: non serve a questa diagnostica, ma va
    comunque rimosso dal train PRIMA dell'undersampling per restare fedeli
    al volume/composizione realmente usati in addestramento), under-
    sampling, feature selection con lo stesso max_features vincente del
    tuning -- poi un sottocampionamento SOLO diagnostico (vedi
    CLF_DIAGNOSTIC_SUBSAMPLE_SIZE).
    """
    data_folder = os.environ.get("DATASET_LOCAL_PATH", "./dataset_cache")
    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            f"Cartella dataset non trovata: '{data_folder}'. "
            f"Imposta DATASET_LOCAL_PATH come per run_baseline.py."
        )

    print(f"[1/5] Caricamento dati da '{data_folder}' (campionamento ribilanciato per giorno, "
          f"target ~{CLF_TARGET_ROWS_PER_DAY} righe/giorno)...")
    loader = RawCSVDataLoader(
        data_url=data_folder, dataset_seed=RANDOM_SEED,
        target_rows_per_day=CLF_TARGET_ROWS_PER_DAY,
    )
    df_raw = loader.load()

    print("[2/5] Binarizzazione + split stratificato + preprocessing...")
    preprocessor = CICIDSPreprocessor(target_column=CLF_TARGET_COL)
    splitter = StratifiedDataSplitter(target_column=CLF_TARGET_COL, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    df_binarized = preprocessor.binarize_target(df_raw)
    train_df, _ = splitter.split(df_binarized)
    train_df = preprocessor.process(train_df)

    # Stesso split del validation set fatto in run_baseline.py, PRIMA
    # dell'undersampling (vedi VALIDATION_SIZE_FOR_THRESHOLD lì): il
    # validation stesso non serve a questa diagnostica (non si sceglie
    # nessuna soglia qui), ma va comunque tolto dal train per riottenere
    # lo stesso volume/composizione su cui run_baseline.py addestra
    # davvero il modello finale -- altrimenti la curva OOB misurata qui
    # userebbe più righe di quelle che il modello di produzione vede.
    print(f"[3/5] Split di un validation set ({CLF_VALIDATION_SIZE_FOR_THRESHOLD*100:.0f}% del train, "
          f"come in run_baseline.py -- scartato qui, non serve a questa diagnostica)...")
    validation_splitter = StratifiedDataSplitter(
        target_column=CLF_TARGET_COL, test_size=CLF_VALIDATION_SIZE_FOR_THRESHOLD, random_state=RANDOM_SEED
    )
    train_df, _ = validation_splitter.split(train_df)

    print("[4/5] Under-sampling della classe maggioritaria (solo train set) + feature selection "
          f"(max_features='{tuned_hp.get('max_features', 'sqrt')}', dal tuning)...")
    train_df = undersample_majority_class(
        train_df, target_column=CLF_TARGET_COL,
        majority_class=0, minority_class=1,
        ratio=CLF_UNDERSAMPLING_RATIO, random_state=RANDOM_SEED,
    )
    fs = CICIDSFeatureSelector(
        target_column=CLF_TARGET_COL, rf_random_state=RANDOM_SEED,
        rf_max_features=tuned_hp.get("max_features", "sqrt"),
        reduce_multicollinearity=True,
        multicollinearity_distance_threshold=CLF_MULTICOLLINEARITY_DISTANCE_THRESHOLD,
        dendrogram_plot_path=f"feature_correlation_dendrogram_{tuned_hp.get('max_features', 'sqrt')}_n_est_diagnostica.png",
    )
    train_df = fs.fit_transform(train_df)

    if len(train_df) > CLF_DIAGNOSTIC_SUBSAMPLE_SIZE:
        print(f"[5/5] Sottocampionamento diagnostico: {len(train_df):,} -> "
              f"{CLF_DIAGNOSTIC_SUBSAMPLE_SIZE:,} righe (stratificato, "
              f"seed={RANDOM_SEED}).".replace(",", "."))
        train_df, _ = train_test_split(
            train_df, train_size=CLF_DIAGNOSTIC_SUBSAMPLE_SIZE,
            stratify=train_df[CLF_TARGET_COL], random_state=RANDOM_SEED,
        )
        train_df = train_df.reset_index(drop=True)
    else:
        print(f"[5/5] Train set ({len(train_df):,} righe) sotto la soglia di sottocampionamento: "
              f"uso tutte le righe disponibili.".replace(",", "."))

    X = train_df.drop(columns=[CLF_TARGET_COL]).to_numpy()
    y = train_df[CLF_TARGET_COL].to_numpy()
    return X, y


def load_tuned_regressor_hp():
    """
    Legge da REG_CONFIG_PATH sia gli iperparametri (tranne n_estimators, che
    qui è la variabile indipendente) sia la ricetta del dataset (n_samples,
    n_features, noise) -- prodotti da run_baseline.py. Analoga a
    load_tuned_classifier_hp(), stessa idea: non decide nulla, legge solo
    cosa è già stato deciso altrove.
    """
    if not os.path.exists(REG_CONFIG_PATH):
        raise FileNotFoundError(
            f"'{REG_CONFIG_PATH}' non trovato: esegui prima run_baseline.py "
            f"con dataset_type='synthetic' e tree_type='regressor' almeno una volta."
        )
    with open(REG_CONFIG_PATH, "r") as f:
        config = json.load(f)
    hp = config.get("hyperparameters")
    if not hp:
        raise ValueError(f"'{REG_CONFIG_PATH}' trovato ma senza sezione 'hyperparameters'.")
    dataset_recipe = {
        "n_samples": config.get("n_samples"),
        "n_features": config.get("n_features"),
        "noise": config.get("noise"),
    }
    if any(v is None for v in dataset_recipe.values()):
        raise ValueError(
            f"'{REG_CONFIG_PATH}' non contiene la ricetta completa del dataset "
            f"(n_samples/n_features/noise): {dataset_recipe}."
        )
    print(f"[INFO] Iperparametri letti da '{REG_CONFIG_PATH}': {hp}")
    print(f"[INFO] Ricetta dataset letta da '{REG_CONFIG_PATH}': {dataset_recipe}")
    return hp, dataset_recipe


def prepare_regressor_dataset(dataset_recipe):
    """
    Genera il dataset sintetico di regressione con la ricetta ESATTA di
    produzione (n_samples, n_features, noise letti da REG_CONFIG_PATH, non
    ridichiarati qui) -- Friedman #1 (Friedman 1991; Breiman 1996), stessa
    scelta di SyntheticDataLoader. Nessuna feature selection: sul sintetico
    la separazione segnale/rumore è nota a priori dal generatore (5 feature
    informative fisse, il resto rumore puro).
    """
    n_samples = dataset_recipe["n_samples"]
    n_features = dataset_recipe["n_features"]
    noise = dataset_recipe["noise"]
    print(f"[1/2] Generazione dataset sintetico di regressione (Friedman #1) "
          f"({n_samples:,} righe, {n_features} feature, 5 informative fisse, "
          f"noise={noise})...".replace(",", "."))
    loader = SyntheticDataLoader(
        task="regression",
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_seed=RANDOM_SEED,
        target_column=REG_TARGET_COL,
    )
    df = loader.load()

    print("[2/2] Split train/test (solo il train serve per la curva OOB)...")
    train_df, _ = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    X = train_df.drop(columns=[REG_TARGET_COL]).to_numpy()
    y = train_df[REG_TARGET_COL].to_numpy()
    return X, y


# ---------------------------------------------------------------------------
# Curva OOB error rate con warm_start (nucleo comune ai due task)
# ---------------------------------------------------------------------------
def compute_oob_curve(task, X, y, base_kwargs, grid, marker_n_estimators=None):
    """
    Fa crescere UNA foresta con warm_start=True lungo `grid`, calcolando ad
    ogni checkpoint l'OOB error rate = 1 - oob_score_ (attributo nativo
    scikit-learn), esattamente come nell'esempio ufficiale. `task` è
    'classifier' o 'regressor'. Ritorna la lista di righe (una per
    checkpoint di griglia): {n, error_rate, cum_time}.

    NOTA (limitazione dichiarata): oob_score_ nativo include nel calcolo
    anche le righe mai OOB per nessun albero, usando per esse un valore di
    riempimento di default -- vedi la sezione "LIMITAZIONE DICHIARATA"
    nella docstring del modulo. L'effetto è più marcato ai valori bassi
    della griglia.
    """
    is_clf = task == "classifier"
    estimator_cls = RandomForestClassifier if is_clf else RandomForestRegressor

    print(f"\n[CURVA OOB ERROR RATE] task='{task}', warm_start, griglia {grid[0]}..{grid[-1]} "
          f"(step {grid[1] - grid[0] if len(grid) > 1 else '-'}), altri iperparametri fissi: "
          f"{base_kwargs}")
    print("=" * 100)

    rf = estimator_cls(warm_start=True, oob_score=True, n_jobs=-1,
                        random_state=RANDOM_SEED, **base_kwargs)

    rows = []
    cumulative_time = 0.0
    for i, n in enumerate(grid, start=1):
        print(f"  [{i}/{len(grid)}] fit warm_start fino a n_estimators={n} ...", flush=True)
        rf.set_params(n_estimators=n)
        t0 = time.perf_counter()
        rf.fit(X, y)
        elapsed = time.perf_counter() - t0
        cumulative_time += elapsed
        print(f"  [{i}/{len(grid)}] completato in {elapsed:.2f}s (cumulato: {cumulative_time:.2f}s)", flush=True)

        error_rate = 1 - rf.oob_score_
        rows.append(dict(n=n, error_rate=error_rate, cum_time=cumulative_time))

    # --- Tabella ---
    print(f"\n  {'n_est':<7} | {'OOB error rate':<16} | {'t cum(s)'}")
    print("  " + "-" * 44)
    for r in rows:
        marker = "  <-- valore di riferimento" if r["n"] == marker_n_estimators else ""
        print(f"  {r['n']:<7} | {r['error_rate']:<16.5f} | {r['cum_time']:8.2f}{marker}")

    return rows


def plot_oob_curve(rows, task, marker_n_estimators, out_prefix):
    """
    Grafico a pannello singolo, in stile identico all'esempio ufficiale
    scikit-learn: OOB error rate (asse Y) vs n_estimators (asse X), una
    sola curva.
    """
    grid_vals = [r["n"] for r in rows]
    error_vals = [r["error_rate"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(grid_vals, error_vals, color='#2563eb', linewidth=2, label="OOB error rate")
    if marker_n_estimators:
        ax.axvline(x=marker_n_estimators, color='#16a34a', linestyle='--', linewidth=1.5,
                   label=f'n_estimators={marker_n_estimators} (valore di riferimento)')
    ax.set_xlim(grid_vals[0], grid_vals[-1])
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("OOB error rate")
    ax.set_title(f"OOB error rate al crescere di n_estimators (warm_start) — {task}\n"
                 "leggi a occhio dove la curva si appiattisce")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

    fig.tight_layout()
    out_path = f"{out_prefix}_oob_error_rate_warmstart.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\n  Grafico salvato in: {out_path}")


# ---------------------------------------------------------------------------
# Entry point per task
# ---------------------------------------------------------------------------
def run_classifier_analysis():
    print("\n" + "#" * 100)
    print("  TASK: CLASSIFICAZIONE (dataset reale CICIDS)")
    print("#" * 100)
    tuned_hp = load_tuned_classifier_hp()
    X, y = prepare_classifier_dataset(tuned_hp)

    bootstrap = bool(tuned_hp.get("bootstrap", True))
    if not bootstrap:
        raise ValueError(
            "La stima OOB richiede bootstrap=True (Breiman 2001, Definition 1.1): "
            f"config_real.json indica bootstrap={bootstrap}, incompatibile con questa diagnostica."
        )
    base_kwargs = dict(
        max_depth=tuned_hp.get("max_depth"),
        min_samples_split=int(tuned_hp.get("min_samples_split", 2)),
        max_features=tuned_hp.get("max_features", "sqrt"),
        criterion=tuned_hp.get("criterion", "gini"),
        class_weight=tuned_hp.get("class_weight"),
        bootstrap=True,
        max_samples=float(tuned_hp.get("max_samples", 1.0)),
    )
    tuned_n = int(tuned_hp["n_estimators"]) if tuned_hp.get("n_estimators") is not None else None
    grid = sorted(set(DEFAULT_GRID) | ({tuned_n} if tuned_n else set()))

    rows = compute_oob_curve("classifier", X, y, base_kwargs, grid, marker_n_estimators=tuned_n)
    plot_oob_curve(rows, "Classificazione (CICIDS)", tuned_n, out_prefix="classifier")
    return rows


def run_regressor_analysis():
    print("\n" + "#" * 100)
    print("  TASK: REGRESSIONE (dataset sintetico)")
    print("#" * 100)
    hp, dataset_recipe = load_tuned_regressor_hp()
    X, y = prepare_regressor_dataset(dataset_recipe)

    bootstrap = bool(hp.get("bootstrap", True))
    if not bootstrap:
        raise ValueError(
            "La stima OOB richiede bootstrap=True (Breiman 2001, Definition 1.1): "
            f"'{REG_CONFIG_PATH}' indica bootstrap={bootstrap}, incompatibile con questa diagnostica."
        )
    raw_max_samples = hp.get("max_samples")
    base_kwargs = dict(
        max_depth=hp.get("max_depth"),
        min_samples_split=int(hp.get("min_samples_split", 2)),
        max_features=hp.get("max_features", 1 / 3),
        criterion=hp.get("criterion", "squared_error"),
        bootstrap=True,
        max_samples=float(raw_max_samples) if raw_max_samples is not None else None,
    )
    reference_n = int(hp["n_estimators"]) if hp.get("n_estimators") is not None else None
    grid = sorted(set(DEFAULT_GRID) | ({reference_n} if reference_n else set()))

    rows = compute_oob_curve("regressor", X, y, base_kwargs, grid, marker_n_estimators=reference_n)
    plot_oob_curve(rows, "Regressione (sintetico)", reference_n, out_prefix="regressor")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Curva OOB error rate (warm_start, stile ufficiale sklearn) per la "
                     "giustificazione empirica di n_estimators."
    )
    parser.add_argument("--task", choices=["classifier", "regressor", "both"], default="both",
                         help="Quale task analizzare (default: both).")
    args = parser.parse_args()

    if args.task in ("classifier", "both"):
        run_classifier_analysis()
    if args.task in ("regressor", "both"):
        run_regressor_analysis()

    print("\n" + "=" * 100)
    print("  PROSSIMO PASSO")
    print("=" * 100)
    print("  1. Apri i grafici '*_oob_error_rate_warmstart.png' e leggi a occhio dove la curva")
    print("     si appiattisce -- unico criterio.")
    print("  2. Ricorda la limitazione dichiarata: oob_score_ nativo include anche le righe mai")
    print("     OOB per nessun albero, con un bias più marcato ai valori bassi della griglia")
    print("     (vedi sezione 'LIMITAZIONE DICHIARATA' nella docstring del modulo) -- non")
    print("     scegliere n_estimators guardando solo i primi checkpoint della curva.")
    print("  3. Questo script NON scrive alcun manifesto. Se decidi di cambiare n_estimators:")
    print("     - classificazione: rilancia run_baseline.py con SKIP_TUNING (riusa gli altri")
    print("       iperparametri) e modifica manualmente 'n_estimators' in config_real.json,")
    print("       oppure rilancia il tuning includendo il nuovo valore nella griglia Optuna;")
    print("     - regressione: stesso principio, sul manifesto")
    print("       'config_synthetic_regressor.json' (o rilancia il tuning includendo")
    print("       il nuovo valore in REGRESSOR_SEARCH_N_ESTIMATORS, in run_baseline.py).")
    print("     In entrambi i casi cita in relazione:")
    print("     (a) l'esempio ufficiale scikit-learn per il metodo warm_start e la metrica")
    print("         OOB error rate,")
    print("     (b) la limitazione dichiarata sul bias a basso n_estimators (sezione dedicata")
    print("         nella docstring di questo script),")
    print("     (c) Breiman 2001 Sec. 3.1 per la stima OOB come base del criterio stesso.")


if __name__ == "__main__":
    main()