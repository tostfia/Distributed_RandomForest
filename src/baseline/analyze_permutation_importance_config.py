"""
Diagnostica per gli iperparametri della foresta PRELIMINARE usata dalla OOB
permutation importance in CICIDSFeatureSelector: rf_n_estimators=200 e
importance_threshold=0.0. Nessuno dei due era stato giustificato con
evidenza empirica finora — stesso trattamento già dato a n_estimators per
la regressione (curva di stabilizzazione) e per la classificazione (tuning).

Due evidenze prodotte, entrambe sul dataset reale (serve DATASET_LOCAL_PATH,
come per run_baseline.py):

  1. STABILITA' DEL RANKING al crescere di rf_n_estimators. Non basta
     guardare l'OOB score della foresta preliminare (quello si stabilizza
     in fretta, come qualunque Random Forest — vedi Breiman Sec. 3.1): quello
     che conta per la feature selection è se l'ORDINE delle feature per
     importanza è affidabile. Per ogni valore di rf_n_estimators nella
     griglia, calcola la correlazione di Spearman tra il ranking ottenuto e
     quello di una foresta di riferimento molto più grande (400 alberi), più
     la Jaccard similarity tra l'insieme di feature scartate a
     importance_threshold=0.0. Se rf_n_estimators=200 dà già un ranking e un
     insieme di feature scartate quasi identici al riferimento, è
     l'evidenza che giustifica il valore.

  2. SENSITIVITY DI importance_threshold: usando la foresta più grande come
     riferimento (stima più stabile), replica lo stesso tipo di analisi già
     fatta per CORRELATION_THRESHOLD — istogramma della distribuzione delle
     importanze OOB e tabella di quante feature verrebbero scartate a soglie
     diverse, per verificare che 0.0 non cada in una zona "affollata" della
     distribuzione.

Uso:
    python analyze_permutation_importance_config.py

Nota tempi: la griglia arriva fino a 400 alberi per il riferimento; su
dataset reali di dimensioni non piccole può richiedere alcuni minuti totali
(una foresta preliminare per ogni valore della griglia).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.datasplitter import StratifiedDataSplitter
from src.shared.utilities.featureselection import CICIDSFeatureSelector

# Stessi valori di run_baseline.py — aggiornare qui se li cambi anche lì.
RANDOM_SEED = 123
TEST_SIZE = 0.2
SAMPLE_FRACTION = 0.05
TARGET_COL = "Label"

# Sottocampione SOLO per questa diagnostica (non per run_baseline.py, che
# resta sul train set completo). Motivazione: la griglia sotto addestra 7
# foreste preliminari COMPLETE (fit + intero ciclo di permutation
# importance) — sul train set pieno (~645k righe), il solo caso
# rf_n_estimators=100 ha già richiesto ~13 minuti nel run reale; scalando
# linearmente sull'intera griglia fino a 400 alberi il tempo totale stimato
# supera le 2 ore. La STABILITA' DEL RANKING (quello che questa diagnostica
# misura: Spearman rho e Jaccard tra le feature scartate a diversi
# rf_n_estimators) è governata dal numero di alberi, non dalla dimensione
# del dataset — stessa logica già applicata alla diagnostica di regressione
# per n_estimators. Riduce il tempo totale a un ordine di grandezza gestibile
# (~15-20 minuti), pur restando un campione ampio (100k righe, non un
# giocattolo). Il risultato di produzione (quali feature scartare davvero)
# resta quello calcolato da run_baseline.py sul dataset completo.
DIAGNOSTIC_SUBSAMPLE_SIZE = 100_000

RF_N_ESTIMATORS_GRID = [20, 50, 100, 150, 200, 300, 400]
REFERENCE_N_ESTIMATORS = 400  # il più grande della griglia, usato come "verità" approssimata
CHOSEN_RF_N_ESTIMATORS = 200  # valore attuale in CICIDSFeatureSelector
CHOSEN_THRESHOLD = 0.0
CANDIDATE_THRESHOLDS = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0]


def prepare_preprocessed_train_set():
    """
    Prepara il train set FINO al preprocessing incluso, ma PRIMA della
    feature selection (che è proprio l'oggetto di questa diagnostica).
    """
    data_folder = os.environ.get("DATASET_LOCAL_PATH", "./dataset_cache")
    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            f"Cartella dataset non trovata: '{data_folder}'. "
            f"Imposta DATASET_LOCAL_PATH come per run_baseline.py."
        )

    print(f"[1/3] Caricamento dati da '{data_folder}' (sample_fraction={SAMPLE_FRACTION})...")
    loader = RawCSVDataLoader(data_url=data_folder, sample_fraction=SAMPLE_FRACTION, dataset_seed=RANDOM_SEED)
    df_raw = loader.load()

    print("[2/3] Binarizzazione + split stratificato + preprocessing...")
    preprocessor = CICIDSPreprocessor(target_column=TARGET_COL)
    splitter = StratifiedDataSplitter(target_column=TARGET_COL, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    df_binarized = preprocessor.binarize_target(df_raw)
    train_df, _ = splitter.split(df_binarized)
    train_df = preprocessor.process(train_df)

    if len(train_df) > DIAGNOSTIC_SUBSAMPLE_SIZE:
        print(f"[3/3] Sottocampionamento diagnostico: {len(train_df):,} -> "
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

    return train_df


def jaccard(set_a, set_b):
    set_a, set_b = set(set_a), set(set_b)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def analyze_rf_n_estimators_stability(train_df):
    print("\n[3a/3] STABILITA' DEL RANKING al crescere di rf_n_estimators")
    print("=" * 78)

    importance_by_n = {}
    for n in RF_N_ESTIMATORS_GRID:
        print(f"\n  --- Foresta preliminare con rf_n_estimators={n} ---")
        fs = CICIDSFeatureSelector(
            target_column=TARGET_COL,
            rf_n_estimators=n,
            rf_random_state=RANDOM_SEED,
            importance_threshold=CHOSEN_THRESHOLD,
        )
        fs.fit(train_df)
        importance_by_n[n] = fs.importance_scores_

    reference = importance_by_n[REFERENCE_N_ESTIMATORS]
    reference_dropped = set(reference[reference <= CHOSEN_THRESHOLD].index)

    print("\n" + "=" * 78)
    print(f"  CONFRONTO CON RIFERIMENTO ({REFERENCE_N_ESTIMATORS} alberi)")
    print("=" * 78)
    print(f"  {'rf_n_estimators':<17} | {'Spearman rho':<14} | {'Jaccard scartate':<17} | {'N. scartate'}")
    print("  " + "-" * 68)
    rho_list = []
    jac_list = []
    for n in RF_N_ESTIMATORS_GRID:
        series = importance_by_n[n]
        # Allinea sugli stessi indici (stesse feature candidate in tutti i run)
        common_idx = series.index.intersection(reference.index)
        rho, _ = spearmanr(series.loc[common_idx], reference.loc[common_idx])
        dropped = set(series[series <= CHOSEN_THRESHOLD].index)
        jac = jaccard(dropped, reference_dropped)
        rho_list.append(rho)
        jac_list.append(jac)
        marker = "  <-- config attuale" if n == CHOSEN_RF_N_ESTIMATORS else ""
        marker += "  (riferimento)" if n == REFERENCE_N_ESTIMATORS else ""
        print(f"  {n:<17} | {rho:<14.4f} | {jac:<17.4f} | {len(dropped)}{marker}")

    print("\n  Interpretazione: rho vicino a 1.0 significa che il ranking delle feature")
    print("  per importanza è già stabile a quel numero di alberi (non cambierebbe")
    print("  sostanzialmente aggiungendone altri). Jaccard vicino a 1.0 significa che")
    print(f"  l'insieme di feature scartate a soglia {CHOSEN_THRESHOLD} è già lo stesso")
    print(f"  del riferimento a {REFERENCE_N_ESTIMATORS} alberi: usare rf_n_estimators="
          f"{CHOSEN_RF_N_ESTIMATORS} non cambierebbe quali feature vengono eliminate.")

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    ax.plot(RF_N_ESTIMATORS_GRID, rho_list, marker='o', color='#2563eb', linewidth=2,
             label='Spearman rho (ranking vs riferimento)')
    ax.plot(RF_N_ESTIMATORS_GRID, jac_list, marker='s', color='#dc2626', linewidth=2,
             label='Jaccard (feature scartate vs riferimento)')
    ax.axvline(x=CHOSEN_RF_N_ESTIMATORS, color='#16a34a', linestyle='--', linewidth=1.5,
               label=f'rf_n_estimators={CHOSEN_RF_N_ESTIMATORS} (config attuale)')
    ax.set_xlabel("rf_n_estimators (foresta preliminare)")
    ax.set_ylabel("Similarità col riferimento (400 alberi)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Stabilità del ranking di importanza al crescere di rf_n_estimators\n"
                  "(OOB permutation importance, dataset reale CICIDS)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()
    stability_plot_path = "permutation_importance_rf_n_estimators_stability.png"
    fig.savefig(stability_plot_path)
    print(f"\n  Grafico salvato in: {stability_plot_path}")

    return importance_by_n, reference


def analyze_importance_threshold(reference_importance):
    print("\n[3b/3] SENSITIVITY DI importance_threshold (foresta di riferimento, "
          f"{REFERENCE_N_ESTIMATORS} alberi)")
    print("=" * 78)

    print("\n  DISTRIBUZIONE delle importanze OOB (percent increase in misclassification rate):")
    bins = np.linspace(
        min(reference_importance.min(), -5),
        max(reference_importance.max(), 10),
        21,
    )
    counts, edges = np.histogram(reference_importance.values, bins=bins)
    max_count = counts.max() if len(counts) else 1
    for i, c in enumerate(counts):
        bar = "█" * int(50 * c / max_count) if max_count else ""
        marker = "  <-- soglia scelta" if edges[i] <= CHOSEN_THRESHOLD < edges[i + 1] else ""
        print(f"  [{edges[i]:7.2f}, {edges[i+1]:7.2f}) : {c:3d} {bar}{marker}")

    print(f"\n  Statistiche: min={reference_importance.min():.2f}%  "
          f"median={reference_importance.median():.2f}%  "
          f"mean={reference_importance.mean():.2f}%  max={reference_importance.max():.2f}%")

    print("\n  SENSITIVITY: feature scartate al variare della soglia")
    print("  " + "-" * 60)
    print(f"  {'Soglia':<10} | {'N. scartate':<12} | {'% scartate':<12} | {'N. rimaste'}")
    print("  " + "-" * 60)
    n_total = len(reference_importance)
    for t in CANDIDATE_THRESHOLDS:
        n_dropped = int((reference_importance <= t).sum())
        marker = "  <-- SCELTA ATTUALE" if abs(t - CHOSEN_THRESHOLD) < 1e-9 else ""
        print(f"  {t:<10.1f} | {n_dropped:<12d} | {n_dropped/n_total*100:<11.1f}% | {n_total - n_dropped}{marker}")

    print("\n  Feature al confine della soglia scelta (±2 punti percentuali):")
    near_border = reference_importance[
        (reference_importance >= CHOSEN_THRESHOLD - 2) & (reference_importance <= CHOSEN_THRESHOLD + 2)
    ]
    if near_border.empty:
        print("  Nessuna feature nell'intorno: la soglia cade in una zona 'vuota' "
              "della distribuzione (buon segno, scelta robusta a piccole variazioni).")
    else:
        for feat, val in near_border.sort_values().items():
            status = "SCARTATA" if val <= CHOSEN_THRESHOLD else "TRATTENUTA"
            print(f"    {feat:<40} {val:+7.2f}%  [{status}]")

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    ax.hist(reference_importance.values, bins=bins, color='#2563eb', alpha=0.75, edgecolor='white')
    ax.axvline(x=CHOSEN_THRESHOLD, color='#dc2626', linestyle='--', linewidth=1.5,
               label=f'importance_threshold={CHOSEN_THRESHOLD} (soglia scelta)')
    ax.set_xlabel("Importanza OOB (percent increase in misclassification rate)")
    ax.set_ylabel("Numero di feature")
    ax.set_title("Distribuzione dell'importanza OOB delle feature\n"
                  f"(foresta di riferimento, {REFERENCE_N_ESTIMATORS} alberi)")
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')
    fig.tight_layout()
    histogram_plot_path = "permutation_importance_threshold_distribution.png"
    fig.savefig(histogram_plot_path)
    print(f"\n  Grafico salvato in: {histogram_plot_path}")


def main():
    train_df = prepare_preprocessed_train_set()
    importance_by_n, reference = analyze_rf_n_estimators_stability(train_df)
    analyze_importance_threshold(reference)
    print("\n[OK] Copia le due tabelle e i due grafici nella relazione come giustificazione "
          "empirica di rf_n_estimators=200 e importance_threshold=0.0.")


if __name__ == "__main__":
    main()