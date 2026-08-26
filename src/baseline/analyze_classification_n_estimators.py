"""
Diagnostica per giustificare gli iperparametri EFFETTIVAMENTE usati nella
classificazione sul dataset reale.

Importante: questo script NON sceglie un valore al posto tuo. Legge
n_estimators da 'outputs_baseline/config_real.json' (prodotto da
run_baseline.py) e mostra dove cade quel valore sulla curva di
stabilizzazione dell'errore OOB. NOTA: dopo l'introduzione
dell'override deliberato in run_baseline.py (n_estimators impostato a un
valore diverso da quello grezzo trovato dalla ricerca OOB, sulla base di
QUESTO stesso script), il valore letto qui potrebbe non essere l'output
originale del tuning ma una scelta successiva — per questo il grafico e la
tabella etichettano il valore come "presente in config_real.json", non più
genericamente "scelto dal tuning": è un'etichetta neutra, corretta in
entrambi i casi.

Se config_real.json non esiste ancora (tuning non ancora lanciato), lo script
gira comunque e produce la curva "neutra" su una griglia di default, così puoi
anche usarlo PRIMA del tuning per decidere il range della griglia di ricerca,
invece che solo dopo per giustificare il risultato.

Cosa fa:
  1. Riproduce la stessa pipeline di run_baseline.py fino al train set finale
     (stesso sample_fraction, stesso binarize+split+preprocess+feature
     selection via OOB permutation importance).
  2. Se disponibile, legge n_estimators (e gli altri iperparametri) da
     config_real.json, per addestrare la foresta di diagnostica con la STESSA
     configurazione presente nel manifesto (a parte n_estimators, che è la
     variabile indipendente dell'analisi).
  3. Addestra una foresta con oob_score=True su una griglia di n_estimators
     che include sempre il valore letto dal manifesto (se presente),
     misurando OOB accuracy e OOB F1.
  4. Stampa una tabella con i delta tra un valore e il successivo, e marca
     esplicitamente la riga corrispondente al valore presente nel manifesto.
  5. Salva un grafico (oob_accuracy_f1_vs_n_estimators_classification.png).

Uso:
    python analyze_classification_n_estimators.py
"""
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.datasplitter import StratifiedDataSplitter
from src.shared.utilities.featureselection import CICIDSFeatureSelector

# Stessi valori di run_baseline.py — aggiornare qui se li cambi anche lì.
RANDOM_SEED = 123
TEST_SIZE = 0.2
SAMPLE_FRACTION = 0.05
TARGET_COL = "Label"
CONFIG_REAL_PATH = os.path.join("outputs_baseline", "config_real.json")
DEFAULT_GRID = [5, 10, 20, 30, 40, 60, 80, 120, 160, 200]


def load_tuned_hyperparameters():
    """
    Ritorna il dizionario 'hyperparameters' di config_real.json se esiste,
    altrimenti None. Non decide nulla: si limita a leggere cosa il tuning ha
    già scelto, per permettere di costruire la diagnostica intorno a quel
    valore specifico invece che intorno a un'ipotesi arbitraria.
    """
    if not os.path.exists(CONFIG_REAL_PATH):
        print(f"[INFO] '{CONFIG_REAL_PATH}' non trovato: il tuning non è ancora stato "
              f"eseguito (o è stato eseguito con un output diverso). Uso la griglia di "
              f"default per un'esplorazione preliminare, senza un valore 'tunato' da marcare.")
        return None
    with open(CONFIG_REAL_PATH, "r") as f:
        config = json.load(f)
    hp = config.get("hyperparameters")
    if not hp:
        print(f"[ATTENZIONE] '{CONFIG_REAL_PATH}' trovato ma senza sezione 'hyperparameters'.")
        return None
    print(f"[INFO] Iperparametri tunati letti da '{CONFIG_REAL_PATH}': {hp}")
    return hp


def prepare_train_set():
    data_folder = os.environ.get("DATASET_LOCAL_PATH", "./dataset_cache")
    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            f"Cartella dataset non trovata: '{data_folder}'. "
            f"Imposta DATASET_LOCAL_PATH come per run_baseline.py."
        )

    print(f"[1/3] Caricamento dati da '{data_folder}' (sample_fraction={SAMPLE_FRACTION})...")
    loader = RawCSVDataLoader(data_url=data_folder, sample_fraction=SAMPLE_FRACTION, dataset_seed=RANDOM_SEED)
    df_raw = loader.load()

    print("[2/3] Binarizzazione + split stratificato + preprocessing + feature selection "
          "(identico a run_baseline.py, incluso il criterio OOB permutation importance)...")
    preprocessor = CICIDSPreprocessor(target_column=TARGET_COL)
    splitter = StratifiedDataSplitter(target_column=TARGET_COL, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    df_binarized = preprocessor.binarize_target(df_raw)
    train_df, _ = splitter.split(df_binarized)
    train_df = preprocessor.process(train_df)

    fs = CICIDSFeatureSelector(target_column=TARGET_COL, rf_random_state=RANDOM_SEED)
    train_df = fs.fit_transform(train_df)

    X_train = train_df.drop(columns=[TARGET_COL]).to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    return X_train, y_train


def analyze_n_estimators(X, y, tuned_hp):
    # Gli altri iperparametri (max_features, criterion, class_weight,
    # min_samples_split, max_depth, bootstrap, max_samples) vengono presi
    # DAL TUNING se disponibili, così la foresta di diagnostica è la stessa
    # che il tuning ha selezionato — solo n_estimators varia lungo la griglia.
    # Se il tuning non è ancora stato eseguito, si usano valori di default
    # neutri, solo per esplorare la forma della curva.
    if tuned_hp:
        bootstrap = bool(tuned_hp.get("bootstrap", True))
        base_kwargs = dict(
            max_depth=tuned_hp.get("max_depth"),
            min_samples_split=int(tuned_hp.get("min_samples_split", 2)),
            max_features=tuned_hp.get("max_features", "sqrt"),
            criterion=tuned_hp.get("criterion", "gini"),
            class_weight=tuned_hp.get("class_weight"),
            bootstrap=bootstrap,
        )
        if bootstrap:
            base_kwargs["max_samples"] = float(tuned_hp.get("max_samples", 1.0))
        tuned_n_estimators = int(tuned_hp.get("n_estimators")) if tuned_hp.get("n_estimators") is not None else None
    else:
        base_kwargs = dict(
            max_depth=None, min_samples_split=2, max_features="sqrt",
            criterion="gini", class_weight="balanced", bootstrap=True,
        )
        tuned_n_estimators = None

    base_kwargs["n_jobs"] = -1
    base_kwargs["random_state"] = RANDOM_SEED
    base_kwargs["oob_score"] = True

    grid = sorted(set(DEFAULT_GRID) | ({tuned_n_estimators} if tuned_n_estimators else set()))

    print(f"\n[3/3] Stabilizzazione OOB accuracy/F1 al crescere di n_estimators "
          f"(altri iperparametri: {base_kwargs})...")
    print("=" * 78)
    print(f"  {'n_estimators':<14} | {'OOB Accuracy':<13} | {'OOB F1':<10} | {'Delta F1':<10} | {'Tempo (s)'}")
    print("  " + "-" * 70)

    results = []
    prev_f1 = None
    for n in grid:
        kwargs = dict(base_kwargs)
        kwargs["n_estimators"] = n
        rf = RandomForestClassifier(**kwargs)
        start = time.perf_counter()
        rf.fit(X, y)
        elapsed = time.perf_counter() - start

        oob_acc = rf.oob_score_
        oob_decision = rf.oob_decision_function_
        valid_mask = oob_decision.sum(axis=1) != 0
        n_missing_oob = int((~valid_mask).sum())
        oob_pred = np.argmax(oob_decision[valid_mask], axis=1)
        y_valid = y[valid_mask]
        oob_f1 = f1_score(y_valid, oob_pred, zero_division=0)
        if n_missing_oob > 0:
            print(f"     [OOB] n_estimators={n}: {n_missing_oob} righe "
                  f"({n_missing_oob/len(y)*100:.2f}%) senza copertura OOB, escluse dall'F1.")

        delta = "" if prev_f1 is None else f"{oob_f1 - prev_f1:+.5f}"
        marker = "  <-- VALORE IN config_real.json" if n == tuned_n_estimators else ""
        print(f"  {n:<14} | {oob_acc:<13.5f} | {oob_f1:<10.5f} | {delta:<10} | {elapsed:8.2f}{marker}")
        results.append((n, oob_acc, oob_f1, elapsed))
        prev_f1 = oob_f1

    print("\n  Lettura del risultato (nessuna scelta implicita dello script):")
    print("  • Se il delta F1 intorno al valore scelto dal tuning è già piccolo,")
    print("    è evidenza che il tuning ha convergere su un n_estimators dove il")
    print("    rendimento marginale è basso: puoi citarlo al professore come")
    print("    giustificazione empirica, non arbitraria, della scelta.")
    print("  • Se invece il delta è ancora ampio in quella zona, è un'informazione")
    print("    altrettanto utile: puoi discuterlo come limitazione nota (spazio di")
    print("    ricerca ristretto per vincoli di tempo/CPU) invece di lasciarla implicita.")

    grid_vals = [r[0] for r in results]
    accs = [r[1] for r in results]
    f1s = [r[2] for r in results]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    ax1.plot(grid_vals, accs, marker='o', color='#2563eb', linewidth=2, label='OOB Accuracy')
    ax1.plot(grid_vals, f1s, marker='s', color='#dc2626', linewidth=2, label='OOB F1')
    if tuned_n_estimators:
        ax1.axvline(x=tuned_n_estimators, color='#16a34a', linestyle='--', linewidth=1.5,
                    label=f'n_estimators={tuned_n_estimators} (valore in config_real.json)')
    ax1.set_xlabel("n_estimators")
    ax1.set_ylabel("Score OOB")
    ax1.set_title("Stabilizzazione OOB al crescere di n_estimators\n"
                   "(Random Forest Classifier, dataset reale CICIDS)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    fig.tight_layout()
    out_path = "oob_accuracy_f1_vs_n_estimators_classification.png"
    fig.savefig(out_path)
    print(f"\n  Grafico salvato in: {out_path}")

    return results


def main():
    tuned_hp = load_tuned_hyperparameters()
    X, y = prepare_train_set()
    analyze_n_estimators(X, y, tuned_hp)


if __name__ == "__main__":
    main()