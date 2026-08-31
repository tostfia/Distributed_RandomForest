"""
Diagnostica per giustificare n_estimators nella classificazione sul dataset
REALE, con la STESSA metodologia già applicata alla regressione
(analyze_regression_reference_config.py) — qui si riportano solo le
differenze specifiche alla classificazione. Vedi quel file per la
descrizione estesa del metodo.

RUOLO NELLA PIPELINE (per confronto con la regressione): a differenza della
regressione, qui il "Passo 1" (ricerca congiunta degli iperparametri,
n_estimators incluso) è già dentro run_baseline.py
(oob_hyperparameter_search) e non viene toccato da questo refactor. Questo
script resta un raffinamento POST-ricerca: parte dalla combinazione
vincente in config_real.json e verifica se un n_estimators più basso di
quello trovato dalla ricerca congiunta mantiene prestazioni equivalenti,
per risparmiare tempo di training (stesso principio già applicato in
run_baseline.py tramite scegli_n_estimators(), che ricalcola la soglia
ad ogni run invece di usare un valore fisso — vedi quella funzione per la
motivazione completa). A differenza della regressione, qui n_estimators FA
parte dello spazio di ricerca del Passo 1, quindi la combinazione vincente
non è mai stata scelta ignorando l'effetto dell'ensemble size — la
differenza tra i due task è quindi solo su COME n_estimators viene
raffinato ulteriormente dopo, non su come viene scelto la prima volta.

Questo script NON sceglie un valore al posto tuo: produce la curva OOB e
un punto di verifica Kneedle, ma il criterio primario resta la lettura
VISIVA del grafico (come dichiarato esplicitamente nell'esempio ufficiale
scikit-learn, vedi sotto).

METODOLOGIA -- allineata all'esempio ufficiale di scikit-learn:
    "OOB Errors for Random Forests"
    https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
    (The scikit-learn developers, BSD-3-Clause)

    Come per la regressione, la curva OOB è costruita con warm_start=True:
    UNA SOLA foresta cresce incrementalmente (si aggiungono alberi via via),
    non tante foreste indipendenti create da zero per ogni n_estimators
    (a differenza della VERSIONE PRECEDENTE di questo script, che faceva
    fit indipendenti sulla griglia — approccio meno corretto: con fit
    indipendenti lo schema di seeding interno di sklearn assegna semi
    diversi a seconda di quanti stimatori totali vengono richiesti, quindi
    la curva confrontava foreste diverse punto per punto, non la stessa
    foresta osservata a diversi stadi di crescita). Anche la
    giustificazione teorica dell'uso della stima OOB al posto della k-fold
    Cross-Validation resta quella di Breiman 2001, Sec. 3.1 (vedi
    run_baseline.py, oob_hyperparameter_search, per la citazione completa):
    la stima OOB è quella di UNA foresta con bootstrap=True, non un
    artefatto di più foreste indipendenti confrontate tra loro.

    A DIFFERENZA dell'esempio ufficiale scikit-learn — che calcola l'errore
    OOB con `1 - clf.oob_score_`, senza alcuna correzione per le righe mai
    out-of-bag — questo script corregge esplicitamente quel bias (vedi
    sezione "CORREZIONE DEL BIAS OOB" più sotto). Sul dataset giocattolo
    dell'esempio ufficiale (500 campioni, griglia che parte da 15 alberi)
    l'effetto è verosimilmente trascurabile; sui nostri dati, con una
    griglia che parte da 5 alberi apposta per rendere il bias visibile,
    NON lo è (misurato: fino all'1% di righe scoperte a n_estimators=5).

    LETTURA: il criterio per scegliere n_estimators è la lettura VISIVA del
    grafico. Non viene usato alcun algoritmo di knee-detection automatico
    (rimosso: il Kneedle semplificato usato in una versione precedente di
    questo script segnala sistematicamente un ginocchio troppo basso su
    curve che saturano rapidamente -- individua il punto di massima
    curvatura, non il punto di reale stabilità, e su un plateau lungo la
    corda del chord-diagram passa quasi rasente alla curva stessa, facendo
    apparire "massima" la distanza proprio subito dopo la salita iniziale).
    Come riferimento numerico supplementare resta il punto in cui la
    copertura OOB raggiunge il 100% (limite INFERIORE assoluto, non un
    target).

UNDER-SAMPLING SULLA CLASSE MAGGIORITARIA -- stesso passo di run_baseline.py
(sostituisce la deduplicazione, rimossa da questo lavoro):

    Una diagnostica dedicata (analyze_dedup_by_day.py) ha misurato che la
    deduplicazione a quasi-duplicati (rimossa da questo lavoro) toglieva
    sproporzionatamente più righe di classe Attacco (fino al 75% su
    giorni con traffico ripetitivo per costruzione, es. bruteforce via
    Patator) che di classe Benign, peggiorando lo sbilanciamento delle
    classi invece di correggerlo — coerente con la recall relativamente
    bassa (80.98%) osservata sul test set. Il bilanciamento delle classi è
    ora affidato esclusivamente all'under-sampling della classe
    maggioritaria (Benign), applicato SOLO al train set, subito dopo lo
    split e PRIMA della feature selection — vedi undersampling.py per la
    motivazione completa dell'ordine e del rapporto scelto.

CAMPIONAMENTO RIBILANCIATO PER GIORNO -- stesso passo di run_baseline.py,
PRIMA mancante anche qui (bug corretto in questa versione, stessa classe di
problema della deduplicazione sopra):

    run_baseline.py non usa più una sample_fraction UNIFORME su tutti i 10
    CSV: usa target_rows_per_day (RawCSVDataLoader), che punta a un numero
    di righe comparabile per ciascun giorno di cattura invece che alla
    stessa percentuale — necessario perché i 10 giorni hanno volumi molto
    diversi tra loro (un solo giorno, Thuesday-20-02-2018, pesa da solo
    quasi la metà del dataset), e diversi file risultano troncati a monte
    al limite di riga di Excel (1.048.575), quindi il loro conteggio non
    riflette il vero volume di traffico di quel giorno (vedi commento
    esteso in run_baseline.py, sezione TARGET_ROWS_PER_DAY). Una versione
    precedente di questo script campionava ancora con sample_fraction
    uniforme: la curva OOB prodotta descriveva quindi un dataset diverso —
    meno eterogeneo per giorno di cattura — da quello che run_baseline.py
    userà davvero in produzione. Stesso principio della deduplicazione:
    l'obiettivo di questo script è restare rappresentativo di
    run_baseline.py, quindi ogni cambiamento nella preparazione dati fatto
    lì va replicato anche qui.

CORREZIONE DEL BIAS OOB A n_estimators BASSO -- stessa correzione già
applicata in analyze_regression_reference_config.py e nel bug fix di
oob_hyperparameter_search (vedi commento "BUG FIX" in run_baseline.py),
ma con una differenza tecnica importante rispetto alla regressione:

    Quando una riga di training non è MAI out-of-bag per nessun albero,
    sklearn riempie oob_decision_function_[riga] con [0, 0, ...] (tutte le
    classi a zero) invece di NaN. rf.oob_score_ (accuracy) e qualunque
    metrica calcolata con np.argmax su queste righe SENZA controllo
    assegnano silenziosamente la classe 0 (Benign) come se fosse una
    predizione valida -- esattamente il bug già corretto in
    oob_hyperparameter_search e nello script di regressione.

    A DIFFERENZA della regressione, però, qui NON serve la ricostruzione
    manuale della copertura via sklearn.ensemble._forest._generate_sample_indices
    (API privata): per la classificazione il segnale "riga mai OOB" è
    INEQUIVOCABILE, perché una riga davvero valutata da almeno un albero ha
    sempre almeno un voto (somma della decision function > 0), mentre una
    riga mai valutata ha somma ESATTAMENTE zero su tutte le classi. Per la
    regressione questo non è possibile: una predizione di regressione
    vera può valere esattamente 0.0, indistinguibile dal riempimento di
    default -- da qui la necessità, solo lì, di ricostruire la copertura
    dagli indici di bootstrap. Qui basta:
        valid_mask = oob_decision_function_.sum(axis=1) != 0
    Questo script calcola quindi DUE curve, per ACCURACY e per F1:
      • "naive" -- calcolata su TUTTE le righe (le righe mai OOB contano
        come predette classe 0 via np.argmax, cosi' come fa rf.oob_score_);
      • "corretta" -- calcolata SOLO sulle righe con valid_mask=True.
    Il grafico mostra entrambe le curve, cosi' la lettura visiva si fa
    sulle curve corrette, non su quelle distorte.

Gli iperparametri diversi da n_estimators (max_depth, min_samples_split,
max_features, criterion, class_weight, bootstrap, max_samples) restano
quelli letti da 'outputs_baseline/config_real.json' (prodotto dal tuning
OOB in run_baseline.py, Breiman 2001 Sec. 3.1): SOLO n_estimators è la
variabile indipendente di questa diagnostica, fatta crescere
incrementalmente via warm_start. Se config_real.json non esiste ancora
(tuning non ancora lanciato), si usano valori di default neutri, così lo
script gira anche PRIMA del tuning per orientare il range della griglia di
ricerca, non solo dopo per giustificarne il risultato.

SOTTOCAMPIONAMENTO DIAGNOSTICO -- stessa tecnica e stessa giustificazione
già usate in analyze_permutation_importance_config.py (vedi il commento su
DIAGNOSTIC_SUBSAMPLE_SIZE lì): con warm_start, sklearn ricalcola
oob_decision_function_ da zero su TUTTI gli alberi correnti a OGNI
checkpoint della griglia, non solo sui nuovi -- con una griglia 5..200 a
step 5 (40 checkpoint) il costo equivalente e' 5+10+...+200 = 4100
"alberi-equivalenti" di valutazione OOB, cioe' 20 volte il costo di un
singolo fit a 200 alberi. Sull'intero train set reale (~645k righe) questo
e' il motivo per cui una prima esecuzione di questo script puo' richiedere
ore. La STABILITA' della curva (dove si appiattisce l'errore OOB) e'
governata dal numero di alberi, non dalla dimensione del dataset -- stessa
logica gia' applicata sia alla diagnostica di regressione
(N_SAMPLES=30.000 invece di 300.000) sia a quella di permutation
importance (100.000 righe invece del train set completo). Il risultato di
produzione (n_estimators scelto per run_baseline.py) resta valido sul
train set completo per la stessa ragione: la forma della curva OOB non
dipende dalla dimensione del campione. NOTA: con la deduplicazione ora
applicata (vedi sopra), il numero di righe disponibili prima del
sottocampionamento sarà inferiore a quello di run precedenti a parità di
sample_fraction — è atteso, non un errore.

Uso:
    python -m src.baseline.analyze_classification_n_estimators
"""
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.datasplitter import StratifiedDataSplitter
from src.shared.utilities.featureselection import CICIDSFeatureSelector
from src.shared.utilities.undersampling import undersample_majority_class

# Stessi valori di run_baseline.py — aggiornare qui se li cambi anche lì.
RANDOM_SEED = 123
TEST_SIZE = 0.2
SAMPLE_FRACTION = 0.05
# Campionamento ribilanciato per giorno di cattura, identico a
# run_baseline.py (vedi lì per la motivazione completa: i 10 CSV di
# CIC-IDS2018 hanno volumi molto disomogenei tra loro, e diversi file
# risultano troncati a monte al limite di riga di Excel — vedi commento
# "TARGET_ROWS_PER_DAY" in run_baseline.py). SAMPLE_FRACTION resta definita
# sopra ma viene ignorata quando TARGET_ROWS_PER_DAY non è None, esattamente
# come in run_baseline.py.
TARGET_ROWS_PER_DAY = 100_000
# Under-sampling della classe maggioritaria, identico a run_baseline.py —
# vedi undersampling.py per la motivazione completa (sostituisce la
# deduplicazione, rimossa da questo lavoro dopo la diagnostica
# analyze_dedup_by_day.py).
UNDERSAMPLING_RATIO = 1.0
TARGET_COL = "Label"
CONFIG_REAL_PATH = os.path.join("outputs_baseline", "config_real.json")

# Sottocampione SOLO per questa diagnostica (non per run_baseline.py, che
# resta sul train set completo) — stessa soglia e stessa motivazione di
# DIAGNOSTIC_SUBSAMPLE_SIZE in analyze_permutation_importance_config.py.
# Vedi il commento "SOTTOCAMPIONAMENTO DIAGNOSTICO" in cima al file.
DIAGNOSTIC_SUBSAMPLE_SIZE = 100_000

# Griglia uniforme (stile esempio ufficiale sklearn: min/max/step) — stessi
# estremi usati in analyze_regression_reference_config.py, partendo da 5
# apposta per rendere visibile il bias OOB a basso n_estimators invece di
# aggirarlo implicitamente (come farebbe partire la griglia da 15, soglia
# a cui la copertura è già ~99.9% per la classificazione).
MIN_ESTIMATORS = 5
MAX_ESTIMATORS = 200
STEP_ESTIMATORS = 5
DEFAULT_GRID = list(range(MIN_ESTIMATORS, MAX_ESTIMATORS + 1, STEP_ESTIMATORS))


def load_tuned_hyperparameters():
    """
    Ritorna il dizionario 'hyperparameters' di config_real.json se esiste,
    altrimenti None. Non decide nulla: si limita a leggere cosa il tuning ha
    già scelto, per costruire la diagnostica intorno a quella configurazione
    specifica invece che intorno a un'ipotesi arbitraria.
    """
    if not os.path.exists(CONFIG_REAL_PATH):
        print(f"[INFO] '{CONFIG_REAL_PATH}' non trovato: il tuning non è ancora stato "
              f"eseguito (o è stato eseguito con un output diverso). Uso valori di "
              f"default neutri per un'esplorazione preliminare.")
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

    print(f"[1/4] Caricamento dati da '{data_folder}' (campionamento ribilanciato per giorno, "
          f"target ~{TARGET_ROWS_PER_DAY} righe/giorno)...")
    loader = RawCSVDataLoader(
        data_url=data_folder,
        sample_fraction=SAMPLE_FRACTION,  # ignorata: target_rows_per_day ha priorità
        dataset_seed=RANDOM_SEED,
        target_rows_per_day=TARGET_ROWS_PER_DAY,
    )
    df_raw = loader.load()

    print("[2/4] Binarizzazione + split stratificato + preprocessing + under-sampling + "
          "feature selection (identico a run_baseline.py, incluso il criterio OOB "
          "permutation importance)...")
    preprocessor = CICIDSPreprocessor(target_column=TARGET_COL)
    splitter = StratifiedDataSplitter(target_column=TARGET_COL, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    df_binarized = preprocessor.binarize_target(df_raw)

    train_df, _ = splitter.split(df_binarized)
    train_df = preprocessor.process(train_df)

    # Under-sampling della classe maggioritaria, identico a run_baseline.py
    # (vedi undersampling.py per la motivazione completa) — necessario
    # perché questo script deve restare rappresentativo di ciò che
    # run_baseline.py usa in produzione. Sostituisce la deduplicazione,
    # rimossa da questo lavoro dopo la diagnostica analyze_dedup_by_day.py
    # (che toglieva sproporzionatamente più righe Attacco che Benign).
    print("[3/4] Under-sampling della classe maggioritaria (solo train set)...")
    train_df = undersample_majority_class(
        train_df, target_column=TARGET_COL,
        majority_class=0, minority_class=1,
        ratio=UNDERSAMPLING_RATIO, random_state=RANDOM_SEED,
    )

    # reduce_multicollinearity=True: identico a run_baseline.py (vedi lì per
    # la motivazione completa) — necessario perché questo script deve
    # restare rappresentativo di ciò che run_baseline.py usa in produzione.
    fs = CICIDSFeatureSelector(
        target_column=TARGET_COL, rf_random_state=RANDOM_SEED,
        reduce_multicollinearity=True, multicollinearity_distance_threshold=0.3,
    )
    train_df = fs.fit_transform(train_df)

    if len(train_df) > DIAGNOSTIC_SUBSAMPLE_SIZE:
        print(f"[4/4] Sottocampionamento diagnostico: {len(train_df):,} -> "
              f"{DIAGNOSTIC_SUBSAMPLE_SIZE:,} righe (stratificato per classe, "
              f"seed={RANDOM_SEED}). Solo per questa diagnostica — vedi commento "
              f"su DIAGNOSTIC_SUBSAMPLE_SIZE.".replace(",", "."))
        from sklearn.model_selection import train_test_split
        train_df, _ = train_test_split(
            train_df,
            train_size=DIAGNOSTIC_SUBSAMPLE_SIZE,
            stratify=train_df[TARGET_COL],
            random_state=RANDOM_SEED,
        )
        train_df = train_df.reset_index(drop=True)
        print(f"       Dimensione dopo il sottocampionamento: {len(train_df):,} righe "
              f"({train_df[TARGET_COL].value_counts().to_dict()})".replace(",", "."))
    else:
        print(f"[4/4] Train set ({len(train_df):,} righe) sotto la soglia di "
              f"sottocampionamento ({DIAGNOSTIC_SUBSAMPLE_SIZE:,}): uso tutte le righe "
              f"disponibili.".replace(",", "."))

    X_train = train_df.drop(columns=[TARGET_COL]).to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    return X_train, y_train


def analyze_n_estimators(X, y, tuned_hp):
    # Gli altri iperparametri vengono presi DAL TUNING se disponibili, così
    # la foresta di diagnostica è la stessa che il tuning ha selezionato —
    # solo n_estimators varia, fatto crescere incrementalmente via
    # warm_start (non rifatto da zero a ogni punto della griglia).
    if tuned_hp:
        bootstrap = bool(tuned_hp.get("bootstrap", True))
        if not bootstrap:
            raise ValueError(
                "La stima OOB richiede bootstrap=True (Breiman 2001, Definition 1.1): "
                f"config_real.json indica bootstrap={bootstrap}, incompatibile con questa "
                "diagnostica."
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
        tuned_n_estimators = int(tuned_hp.get("n_estimators")) if tuned_hp.get("n_estimators") is not None else None
    else:
        base_kwargs = dict(
            max_depth=None, min_samples_split=2, max_features="sqrt",
            criterion="gini", class_weight="balanced", bootstrap=True, max_samples=1.0,
        )
        tuned_n_estimators = None

    grid = sorted(set(DEFAULT_GRID) | ({tuned_n_estimators} if tuned_n_estimators else set()))

    print(f"\n[3/3] Curva OOB con warm_start (metodo: esempio ufficiale scikit-learn), "
          f"griglia {grid[0]}..{grid[-1]}, altri iperparametri: {base_kwargs}...")
    print("=" * 100)

    rf = RandomForestClassifier(
        warm_start=True, oob_score=True, n_jobs=-1, random_state=RANDOM_SEED, **base_kwargs,
    )

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

        oob_decision = rf.oob_decision_function_
        # Righe mai OOB per nessun albero: somma delle "probabilità" OOB
        # esattamente zero su tutte le classi (a differenza della
        # regressione, qui il segnale è inequivocabile — vedi docstring).
        valid_mask = oob_decision.sum(axis=1) != 0
        n_missing = int((~valid_mask).sum())
        coverage_pct = valid_mask.mean() * 100

        # "naive": esattamente cosa farebbe rf.oob_score_ / np.argmax senza
        # controllo — le righe mancanti contano come predette classe 0.
        oob_pred_naive = np.argmax(oob_decision, axis=1)
        acc_naive = accuracy_score(y, oob_pred_naive)
        f1_naive = f1_score(y, oob_pred_naive, zero_division=0)

        # "corretta": solo sulle righe realmente coperte da almeno un albero.
        if valid_mask.any():
            oob_pred_corr = np.argmax(oob_decision[valid_mask], axis=1)
            y_valid = y[valid_mask]
            acc_corr = accuracy_score(y_valid, oob_pred_corr)
            f1_corr = f1_score(y_valid, oob_pred_corr, zero_division=0)
        else:
            acc_corr = np.nan
            f1_corr = np.nan

        if n_missing > 0:
            print(f"     [OOB] n_estimators={n}: {n_missing} righe "
                  f"({100 - coverage_pct:.3f}%) senza copertura OOB, escluse dalla curva corretta.")

        rows.append(dict(
            n=n, coverage_pct=coverage_pct,
            acc_naive=acc_naive, acc_corr=acc_corr,
            f1_naive=f1_naive, f1_corr=f1_corr,
            cum_time=cumulative_time,
        ))

    print(f"\n  {'n_est':<7} | {'copert.%':<9} | {'Acc naive':<10} | {'Acc corr.':<10} | "
          f"{'F1 naive':<9} | {'F1 corr.':<9} | {'Δ F1 corr.':<10} | {'t cum(s)'}")
    print("  " + "-" * 96)
    prev_f1_corr = None
    for r in rows:
        f1c_str = f"{r['f1_corr']:.5f}" if not np.isnan(r['f1_corr']) else "n/d"
        accc_str = f"{r['acc_corr']:.5f}" if not np.isnan(r['acc_corr']) else "n/d"
        delta = "" if prev_f1_corr is None or np.isnan(r['f1_corr']) else f"{r['f1_corr'] - prev_f1_corr:+.5f}"
        marker = "  <-- VALORE IN config_real.json" if r["n"] == tuned_n_estimators else ""
        print(f"  {r['n']:<7} | {r['coverage_pct']:<9.3f} | {r['acc_naive']:<10.5f} | {accc_str:<10} | "
              f"{r['f1_naive']:<9.5f} | {f1c_str:<9} | {delta:<10} | {r['cum_time']:8.2f}{marker}")
        if not np.isnan(r['f1_corr']):
            prev_f1_corr = r['f1_corr']

    fully_covered = [r for r in rows if r["coverage_pct"] >= 100.0 - 1e-9]
    first_full = fully_covered[0]["n"] if fully_covered else None
    print(f"\n  Copertura OOB 100% raggiunta per la prima volta a n_estimators={first_full}.")
    print("  PRIMA di quel punto, le curve naive e corrette DIVERGONO: la naive è")
    print("  artificialmente ottimistica sull'accuracy (i mancanti valgono come classe 0,")
    print("  la classe maggioritaria) — verificare nella tabella quanto è ampio lo scarto")
    print("  ai valori più bassi di n_estimators.")

    grid_vals = [r["n"] for r in rows]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 11), dpi=150, sharex=True)

    ax1.plot(grid_vals, [r["acc_naive"] for r in rows], marker='o', markersize=3,
              color='#9ca3af', linewidth=1.2, linestyle=':', label='Accuracy OOB naive')
    ax1.plot(grid_vals, [r["acc_corr"] for r in rows], marker='o', markersize=3,
              color='#2563eb', linewidth=2, label='Accuracy OOB corretta')
    ax1.plot(grid_vals, [r["f1_naive"] for r in rows], marker='s', markersize=3,
              color='#fca5a5', linewidth=1.2, linestyle=':', label='F1 OOB naive')
    ax1.plot(grid_vals, [r["f1_corr"] for r in rows], marker='s', markersize=3,
              color='#dc2626', linewidth=2, label='F1 OOB corretta')
    if tuned_n_estimators:
        ax1.axvline(x=tuned_n_estimators, color='#16a34a', linestyle='--', linewidth=1.5,
                    label=f'n_estimators={tuned_n_estimators} (valore in config_real.json)')
    ax1.set_ylabel("Score OOB")
    ax1.set_title("Stabilizzazione OOB al crescere di n_estimators (warm_start)\n"
                   "Random Forest Classifier, dataset reale CICIDS — leggi a occhio dove le curve piene si appiattiscono")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=7)

    ax2.plot(grid_vals, [r["coverage_pct"] for r in rows], marker='o', markersize=3,
              color='#7c3aed', linewidth=2)
    ax2.axhline(y=100, color='#9ca3af', linestyle=':', linewidth=1)
    ax2.set_ylabel("Copertura OOB (%)")
    ax2.grid(True, alpha=0.3)

    ax3.plot(grid_vals, [r["cum_time"] for r in rows], marker='o', markersize=3,
              color='#dc2626', linewidth=2)
    ax3.set_xlabel("n_estimators")
    ax3.set_ylabel("Tempo cumulato di training (s)")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = "oob_curve_warmstart_classification.png"
    fig.savefig(out_path)
    print(f"\n  Grafico salvato in: {out_path}")
    print("  Usa questo grafico per la lettura visiva del punto di stabilizzazione:")
    print("  guarda le curve piene (corrette) nel pannello superiore -- dove smettono di")
    print("  salire in modo apprezzabile e' il valore da riportare in run_baseline.py.")
    plt.close(fig)

    # ------------------------------------------------------------------
    # VERSIONE ZOOMATA -- stessi dati, solo asse Y ristretto alla parte
    # STABILE della curva (esclude il tratto di salita iniziale). Su
    # dataset grandi (qui: fino a 100.000 campioni per la stima OOB)
    # l'errore standard della stima OOB scala come 1/sqrt(n_campioni):
    # rispetto all'esempio ufficiale scikit-learn (500 campioni) la
    # variabilita' punto-per-punto e' quindi ridotta di un fattore
    # ~sqrt(100000/500) = ~14x -- ma quella variabilita' e' visibile solo
    # DOPO che la curva ha smesso di salire: se i limiti dell'asse Y sono
    # calcolati su TUTTA la curva (inclusa la salita iniziale, che ha
    # un'ampiezza molto maggiore delle micro-oscillazioni successive), lo
    # zoom resta comunque "schiacciato" dal salto iniziale e il plateau
    # appare piatto per contrasto -- esattamente il problema di una prima
    # versione di questo grafico. La correzione: calcolare i limiti
    # dell'asse Y solo sull'ultimo 75% dei checkpoint esplorati (euristica
    # generica, non specifica a questo dataset: assume che il plateau
    # occupi la maggioranza della griglia, coerente con il fatto che la
    # griglia stessa e' stata dimensionata per rendere visibile la salita
    # iniziale nel primo quarto, non per durare a lungo). I punti della
    # salita iniziale restano comunque disegnati per intero: usciranno
    # semplicemente dalla parte inferiore del grafico invece di comprimere
    # la scala, il che e' di per se' informativo (mostra quanto la curva
    # si sia gia' alzata rispetto alla finestra di zoom).
    # ------------------------------------------------------------------
    plateau_start_idx = len(rows) // 4
    plateau_rows = rows[plateau_start_idx:]
    acc_corr_vals = [r["acc_corr"] for r in plateau_rows if not np.isnan(r["acc_corr"])]
    f1_corr_vals = [r["f1_corr"] for r in plateau_rows if not np.isnan(r["f1_corr"])]
    y_min_acc, y_max_acc = min(acc_corr_vals), max(acc_corr_vals)
    y_min_f1, y_max_f1 = min(f1_corr_vals), max(f1_corr_vals)
    y_pad_acc = (y_max_acc - y_min_acc) * 0.25 if y_max_acc > y_min_acc else 0.001
    y_pad_f1 = (y_max_f1 - y_min_f1) * 0.25 if y_max_f1 > y_min_f1 else 0.001
    print(f"\n  [ZOOM] Limiti calcolati sugli ultimi {len(plateau_rows)}/{len(rows)} checkpoint "
          f"(n_estimators >= {plateau_rows[0]['n']}):")
    print(f"    Accuracy corretta: [{y_min_acc:.5f}, {y_max_acc:.5f}]")
    print(f"    F1 corretta:       [{y_min_f1:.5f}, {y_max_f1:.5f}]")

    fig_zoom, (ax1z, ax2z) = plt.subplots(2, 1, figsize=(8, 8), dpi=150, sharex=True)

    ax1z.plot(grid_vals, [r["acc_corr"] for r in rows], marker='o', markersize=3,
              color='#2563eb', linewidth=1.5, label='Accuracy OOB corretta')
    if tuned_n_estimators:
        ax1z.axvline(x=tuned_n_estimators, color='#16a34a', linestyle='--', linewidth=1.5,
                     label=f'n_estimators={tuned_n_estimators} (valore in config_real.json)')
    ax1z.set_ylabel("Accuracy OOB corretta")
    ax1z.set_ylim(y_min_acc - y_pad_acc, y_max_acc + y_pad_acc)
    ax1z.set_title("Stabilizzazione OOB al crescere di n_estimators — VERSIONE ZOOMATA\n"
                    "Stessi dati del grafico a scala piena, asse Y ristretto alla sola parte stabile "
                    "(la salita iniziale esce dal basso)")
    ax1z.grid(True, alpha=0.3)
    ax1z.legend(loc='lower right', fontsize=7)

    ax2z.plot(grid_vals, [r["f1_corr"] for r in rows], marker='s', markersize=3,
              color='#dc2626', linewidth=1.5, label='F1 OOB corretta')
    if tuned_n_estimators:
        ax2z.axvline(x=tuned_n_estimators, color='#16a34a', linestyle='--', linewidth=1.5)
    ax2z.set_xlabel("n_estimators")
    ax2z.set_ylabel("F1 OOB corretta")
    ax2z.set_ylim(y_min_f1 - y_pad_f1, y_max_f1 + y_pad_f1)
    ax2z.grid(True, alpha=0.3)
    ax2z.legend(loc='lower right', fontsize=7)

    fig_zoom.tight_layout()
    out_path_zoom = "oob_curve_warmstart_classification_zoom.png"
    fig_zoom.savefig(out_path_zoom)
    plt.close(fig_zoom)
    print(f"  Grafico ZOOMATO salvato in: {out_path_zoom}")
    print(f"  (asse Accuracy: [{y_min_acc - y_pad_acc:.5f}, {y_max_acc + y_pad_acc:.5f}], "
          f"asse F1: [{y_min_f1 - y_pad_f1:.5f}, {y_max_f1 + y_pad_f1:.5f}] — stessi dati, "
          f"solo scala diversa)")

    return rows


def main():
    tuned_hp = load_tuned_hyperparameters()
    X, y = prepare_train_set()
    rows = analyze_n_estimators(X, y, tuned_hp)

    print("\n" + "=" * 100)
    print("  PROSSIMO PASSO")
    print("=" * 100)
    print("  1. Apri oob_curve_warmstart_classification.png e leggi a occhio dove le curve")
    print("     BLU/ROSSA piene (accuracy/F1 OOB corrette) si appiattiscono — unico criterio.")
    print("  2. Verifica che il valore scelto sia >= al n_estimators a cui la copertura OOB")
    print("     raggiunge il 100% (stampato sopra): sotto quella soglia la stima OOB è")
    print("     matematicamente incompleta, a prescindere da come appare la curva.")
    print("  3. Il valore finale NON va più scritto a mano in run_baseline.py: alla")
    print("     prossima esecuzione della baseline (run_baseline.py), la funzione")
    print("     scegli_n_estimators() propone automaticamente, sulla base della ricerca")
    print("     OOB di QUEL run, il più piccolo n_estimators entro OOB_F1_TOLERANCE dal")
    print("     migliore — puoi confermarlo, accettarlo, o inserire un valore custom letto")
    print("     da questa tabella. Usa questo grafico come verifica indipendente (dataset")
    print("     dedotto identico a run_baseline.py, incluse dedup e feature selection) di")
    print("     ciò che quella funzione propone, citando:")
    print("     (a) l'esempio ufficiale scikit-learn per il metodo warm_start,")
    print("     (b) la correzione del bias di copertura OOB spiegata in questo script,")
    print("     (c) Breiman 2001 Sec. 3.1 per la stima OOB come base del criterio stesso.")


if __name__ == "__main__":
    main()