"""
Misura empirica della correlazione media rho tra gli alberi della foresta
(Breiman 2001, Sec. 2, Teorema 2.3: PE* <= rho_bar * (1-s^2)/s^2), a
sostegno quantitativo dell'argomento già usato nella relazione per spiegare
perche' il punto di stabilizzazione OOB rispetto a n_estimators non si e'
spostato nonostante il campionamento ribilanciato per giorno abbia reso il
problema oggettivamente piu' difficile (F1 sceso da ~96% a ~88.8%, vedi
run_baseline.py e analyze_classification_n_estimators.py).

METODOLOGIA -- proxy standard della correlazione di Breiman:
    Per ciascun albero t della foresta, si calcola il "margine grezzo" sul
    TEST SET (non OOB: ogni albero puo' essere valutato su tutto il test
    set, che nessun albero ha mai visto in training, quindi non serve
    gestire alcuna copertura parziale come nel caso OOB):
        rmg[t, i] = +1  se l'albero t classifica correttamente il campione i
        rmg[t, i] = -1  altrimenti
    rho_bar e' la correlazione di Pearson media tra rmg[t] e rmg[t'] su
    tutte le coppie di alberi (t, t') della foresta -- proxy diretto della
    "mean correlation between raw margin functions" definita da Breiman
    (Sec. 2), qui approssimata con la correlazione di Pearson standard
    invece della forma esatta pesata per deviazione standard usata nel
    paper originale (semplificazione dichiarata, non un errore: la
    differenza tra le due e' marginale quando le rmg sono vicine a +-1
    quasi ovunque, come nel nostro caso di classificazione binaria con
    accuracy elevata).

    "s" (forza, strength) e' stimata come la margine media dell'insieme:
    s = mean_i( 2*accuracy_ensemble(i) - 1 ), un proxy della definizione
    di Breiman per il caso binario (dove il margine si riduce a
    P(corretto) - P(scorretto) = 2*P(corretto) - 1).

Usa gli STESSI iperparametri e la STESSA pipeline di preparazione dati di
run_baseline.py e analyze_classification_n_estimators.py (stesso
TARGET_ROWS_PER_DAY, stesso seed) -- aggiornare qui se li cambi anche
altrove.

Uso:
    python -m src.baseline.analyze_tree_correlation
"""
import os
import json
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.datasplitter import StratifiedDataSplitter
from src.shared.utilities.featureselection import CICIDSFeatureSelector
from src.shared.utilities.deduplication import remove_near_duplicate_rows

# Stessi valori di run_baseline.py / analyze_classification_n_estimators.py
RANDOM_SEED = 123
TEST_SIZE = 0.2
TARGET_COL = "Label"
TARGET_ROWS_PER_DAY = 100_000
CONFIG_REAL_PATH = os.path.join("outputs_baseline", "config_real.json")

# Numero di alberi usato per la stima di rho_bar: non e' necessario che
# coincida con n_estimators di produzione (30) -- una foresta piu' grande
# da' una stima piu' stabile della correlazione media, dato che il numero
# di coppie cresce quadraticamente. 100 alberi -> 4.950 coppie, sufficiente
# per una stima stabile senza tempi di fit eccessivi.
N_TREES_FOR_ESTIMATE = 100

# Se il test set e' molto grande, calcolare N_TREES_FOR_ESTIMATE^2/2 paia di
# correlazioni su ogni singolo campione ha un costo che cresce con la
# dimensione del test set: sottocampioniamo il test set SOLO per questa
# stima (non tocca in alcun modo il test set reale usato altrove).
MAX_TEST_SAMPLES_FOR_CORR = 30_000


def load_tuned_hyperparameters():
    if not os.path.exists(CONFIG_REAL_PATH):
        raise FileNotFoundError(
            f"'{CONFIG_REAL_PATH}' non trovato: esegui prima run_baseline.py "
            f"(almeno una volta con tuning) per avere una configurazione da cui partire."
        )
    with open(CONFIG_REAL_PATH, "r") as f:
        config = json.load(f)
    hp = config["hyperparameters"]
    print(f"[INFO] Iperparametri letti da '{CONFIG_REAL_PATH}': {hp}")
    return hp


def prepare_train_test():
    """
    Ricostruisce train E test set con la STESSA pipeline deterministica di
    run_baseline.py (stesso seed, stesso TARGET_ROWS_PER_DAY, stessa
    deduplicazione, stesso split, stessa feature selection incluso il secondo
    passo di riduzione multicollinearità) -- necessario
    perche' qui serve anche il test set, che run_baseline.py non
    serializza nel pickle del modello.
    """
    data_folder = os.environ.get("DATASET_LOCAL_PATH", "./dataset_cache")
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Cartella dataset non trovata: '{data_folder}'.")

    print(f"[1/5] Caricamento dati da '{data_folder}' (campionamento ribilanciato per "
          f"giorno, target ~{TARGET_ROWS_PER_DAY} righe/giorno)...")
    loader = RawCSVDataLoader(
        data_url=data_folder, dataset_seed=RANDOM_SEED, target_rows_per_day=TARGET_ROWS_PER_DAY,
    )
    df_raw = loader.load()

    print("[2/5] Binarizzazione + deduplicazione (identico a run_baseline.py)...")
    preprocessor = CICIDSPreprocessor(target_column=TARGET_COL)
    df_binarized = preprocessor.binarize_target(df_raw)
    df_binarized = remove_near_duplicate_rows(df_binarized, target_column=TARGET_COL)

    print("[3/5] Split stratificato...")
    splitter = StratifiedDataSplitter(target_column=TARGET_COL, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_df, test_df = splitter.split(df_binarized)
    train_df = preprocessor.process(train_df)
    test_df = preprocessor.process(test_df)

    print("[4/5] Feature selection (fit SOLO su train, transform su test)...")
    fs = CICIDSFeatureSelector(
        target_column=TARGET_COL, rf_random_state=RANDOM_SEED,
        reduce_multicollinearity=True, multicollinearity_distance_threshold=0.3,
    )
    train_df = fs.fit_transform(train_df)
    test_df = fs.transform(test_df)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].to_numpy()
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].to_numpy()

    if len(X_test) > MAX_TEST_SAMPLES_FOR_CORR:
        print(f"[5/5] Sottocampionamento del TEST SET solo per la stima di rho_bar: "
              f"{len(X_test):,} -> {MAX_TEST_SAMPLES_FOR_CORR:,} righe (stratificato, "
              f"seed={RANDOM_SEED}). Il test set usato altrove nella pipeline NON e' "
              f"toccato da questo sottocampionamento.".replace(",", "."))
        from sklearn.model_selection import train_test_split
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test, train_size=MAX_TEST_SAMPLES_FOR_CORR,
            stratify=y_test, random_state=RANDOM_SEED,
        )

    return X_train, y_train, X_test, y_test


def compute_tree_correlation(forest: RandomForestClassifier, X_test, y_test):
    """
    Ritorna (rho_bar, strength_s, rmg_matrix) -- vedi docstring del modulo
    per la definizione. rmg_matrix ha shape (n_estimators, n_test_samples).
    """
    X_test_arr = X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test
    n_trees = len(forest.estimators_)
    n_samples = len(y_test)

    print(f"\n[CORR] Calcolo margine grezzo per {n_trees} alberi su {n_samples} "
          f"campioni di test...")
    rmg = np.empty((n_trees, n_samples), dtype=np.int8)
    for t, tree in enumerate(forest.estimators_):
        preds = tree.predict(X_test_arr)
        rmg[t] = np.where(preds == y_test, 1, -1)

    # Forza (strength) dell'ensemble: margine medio del VOTO DI MAGGIORANZA
    # (non del singolo albero) -- proxy della definizione di Breiman per il
    # caso binario: s = E[2*P(corretto) - 1].
    ensemble_correct_rate = np.mean(rmg == 1, axis=0)  # per campione, frazione di alberi corretti
    strength_s = float(np.mean(2 * ensemble_correct_rate - 1))

    print(f"[CORR] Calcolo correlazione di Pearson media su tutte le "
          f"{n_trees * (n_trees - 1) // 2} coppie di alberi...")
    # Matrice di correlazione (n_trees x n_trees): np.corrcoef sulle righe di rmg.
    corr_matrix = np.corrcoef(rmg)
    # Media SOLO della parte triangolare superiore, esclusa la diagonale
    # (che vale sempre 1 per definizione, non e' informativa qui).
    iu = np.triu_indices(n_trees, k=1)
    rho_bar = float(np.mean(corr_matrix[iu]))
    rho_std = float(np.std(corr_matrix[iu]))

    return rho_bar, rho_std, strength_s, rmg


def main():
    tuned_hp = load_tuned_hyperparameters()
    X_train, y_train, X_test, y_test = prepare_train_test()

    rf_kwargs = dict(
        n_estimators=N_TREES_FOR_ESTIMATE,
        max_depth=tuned_hp.get("max_depth"),
        min_samples_split=int(tuned_hp.get("min_samples_split", 2)),
        max_features=tuned_hp.get("max_features", "sqrt"),
        criterion=tuned_hp.get("criterion", "gini"),
        class_weight=tuned_hp.get("class_weight"),
        bootstrap=True,
        max_samples=float(tuned_hp.get("max_samples", 1.0)),
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    print(f"\n[FIT] Addestramento foresta con {N_TREES_FOR_ESTIMATE} alberi "
          f"(stessi iperparametri del tuning, tranne n_estimators) su "
          f"{len(X_train):,} righe di train...".replace(",", "."))
    forest = RandomForestClassifier(**rf_kwargs)
    forest.fit(X_train, y_train)

    rho_bar, rho_std, strength_s, rmg = compute_tree_correlation(forest, X_test, y_test)

    print("\n" + "=" * 90)
    print("  CORRELAZIONE MEDIA TRA ALBERI (Breiman 2001, Sec. 2 / Teorema 2.3)")
    print("=" * 90)
    print(f"  rho_bar (correlazione media tra coppie di alberi) : {rho_bar:.4f}  (dev.std tra coppie: {rho_std:.4f})")
    print(f"  s (forza dell'ensemble, margine medio del voto)   : {strength_s:.4f}")
    print(f"  Limite superiore PE* <= rho_bar*(1-s^2)/s^2        : {rho_bar * (1 - strength_s**2) / strength_s**2:.4f}")
    print(f"  Alberi usati per la stima: {N_TREES_FOR_ESTIMATE} ({N_TREES_FOR_ESTIMATE*(N_TREES_FOR_ESTIMATE-1)//2} coppie)")
    print(f"  Campioni di test usati per la stima: {rmg.shape[1]}")
    print("=" * 90)
    print("  Da citare in relazione insieme al confronto tra il plateau OOB del")
    print("  campionamento uniforme e quello del campionamento ribilanciato per giorno:")
    print("  un rho_bar comparabile nei due regimi, a fronte di un F1 molto diverso,")
    print("  e' l'evidenza empirica diretta che la velocita' di stabilizzazione OOB")
    print("  dipende dalla correlazione tra alberi, non dalla difficolta' assoluta")
    print("  del problema.")


if __name__ == "__main__":
    main()