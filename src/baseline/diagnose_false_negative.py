"""
Diagnostica sul modello già addestrato da run_baseline.py (nessun
retraining). Tre analisi, tutte sullo stesso test set ricostruito:

  1. FALSI NEGATIVI per SOTTO-TIPO di attacco (etichetta multi-classe
     originale, prima della binarizzazione Benign/Attacco) -- su quali
     sotto-tipi il modello manca di più.
  2. FALSI POSITIVI per GIORNO DI CATTURA -- se il traffico Benign
     scambiato per Attacco si concentra sui due giorni "protetti"
     (Infiltration) o è distribuito su tutti i giorni.
  3. SCAN DI SOGLIA -- precision/recall/F1 a soglie diverse da 0.5
     sulle probabilità già predette dal modello: nessun retraining,
     serve solo a capire se un'altra soglia dà un compromesso
     migliore di quella di default per l'obiettivo che ti interessa
     (più recall o più precision).

Perché questo script e non un'analisi dentro run_baseline.py: la
binarizzazione (CICIDSPreprocessor.binarize_target) sovrascrive la colonna
"Label" con 0/1, perdendo il sotto-tipo originale (es. "DDoS attack-HOIC",
"Infiltration", "Bot", ...) -- questo script rifà il caricamento e lo split
con la STESSA identica configurazione (stesso seed, stesso campionamento per
giorno) usata da run_baseline.py, tenendo però da parte l'etichetta
originale e il giorno di cattura come colonne aggiuntive, così può
ricongiungerle alle predizioni del test set senza dover ritoccare la
pipeline principale né rifare il tuning.

Precondizioni:
  - run_baseline.py deve essere già stato eseguito almeno una volta sul
    dataset REALE (dataset_type != "synthetic"), con lo stesso identico
    RANDOM_SEED/TARGET_ROWS_PER_DAY/data_folder impostati qui sotto --
    altrimenti il modello caricato dal pickle si aspetta feature/righe che
    non corrispondono al test set ricostruito qui (l'eccezione
    "Colonne attese dal modello ma assenti" segnala proprio questo).
  - Il file 'outputs_baseline/baseline_random_forest_completa.pkl' deve
    esistere (prodotto dalla FASE finale di run_baseline.py).

Output:
  - Stampa a schermo le tre tabelle.
  - Salva su disco in 'outputs_baseline/':
      diagnostica_falsi_negativi_per_sottotipo.csv
      diagnostica_falsi_positivi_per_giorno.csv
      diagnostica_scan_soglia.csv
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader, SOURCE_DAY_COLUMN
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.datasplitter import StratifiedDataSplitter

# STESSI valori di run_baseline.py per il dataset reale -- se li cambi lì,
# cambiali identici anche qui, altrimenti il modello caricato dal pickle si
# troverà davanti un test set diverso da quello su cui è stato valutato.
RANDOM_SEED = 123
TEST_SIZE = 0.2
TARGET_ROWS_PER_DAY = 100_000
TARGET_COL = "Label"

ORIGINAL_LABEL_COL = "Original_Label_Multiclasse"
OUTPUT_DIR = "./outputs_baseline"
PICKLE_PATH = os.path.join(OUTPUT_DIR, "baseline_random_forest_completa.pkl")

# Soglie esplorate per lo scan (passo 3): 0.5 è il default di
# RandomForestClassifier.predict(), incluso qui come riferimento.
THRESHOLDS_TO_SCAN = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def main() -> None:
    data_folder = os.environ.get("DATASET_LOCAL_PATH", "./dataset_cache")
    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            f"Cartella del dataset reale non trovata: '{data_folder}'. Imposta "
            f"DATASET_LOCAL_PATH come per run_baseline.py."
        )
    if not os.path.exists(PICKLE_PATH):
        raise FileNotFoundError(
            f"'{PICKLE_PATH}' non trovato: esegui prima run_baseline.py sul "
            f"dataset reale (deve completare la FASE finale che salva il "
            f".pkl) prima di lanciare questa diagnostica."
        )

    # 1. Ricarica i dati grezzi con la STESSA identica configurazione di
    #    run_baseline.py (RANDOM_SEED, TARGET_ROWS_PER_DAY invariati -> stesso
    #    identico DataFrame e stesso identico split), con l'unica differenza
    #    di tag_source_day=True: attivato SOLO qui, serve a tracciare da
    #    quale giorno viene ogni riga per l'analisi dei Falsi Positivi
    #    (passo 2). Non tocca in alcun modo il .pkl né run_baseline.py.
    print(f"[1/6] Ricaricamento dati grezzi da '{data_folder}' (stesso seed di run_baseline.py)...")
    loader = RawCSVDataLoader(
        data_url=data_folder,
        dataset_seed=RANDOM_SEED,
        target_rows_per_day=TARGET_ROWS_PER_DAY,
        tag_source_day=True,
    )
    df_raw = loader.load()

    # 2. Salva l'etichetta multi-classe originale PRIMA della binarizzazione
    #    (che sovrascrive "Label" con 0/1, perdendo il sotto-tipo).
    df_raw = df_raw.copy()
    df_raw[ORIGINAL_LABEL_COL] = df_raw[TARGET_COL].astype(str).str.strip()

    preprocessor = CICIDSPreprocessor(target_column=TARGET_COL)
    splitter = StratifiedDataSplitter(
        target_column=TARGET_COL, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    print("[2/6] Binarizzazione e split stratificato (identici a run_baseline.py)...")
    df_binarized = preprocessor.binarize_target(df_raw)
    train_df, test_df = splitter.split(df_binarized)
    del train_df  # non serve per questa diagnostica, solo il test set

    # 3. Isola le due colonne extra (etichetta originale + giorno di cattura)
    #    del test set PRIMA di richiamare i passi di process(): altrimenti
    #    _convert_feature_columns_to_numeric proverebbe a convertirle a
    #    numerico e le distruggerebbe in NaN (stesso rischio già individuato
    #    per una colonna di tag stringa in generale).
    extra_cols_test = test_df[[ORIGINAL_LABEL_COL, SOURCE_DAY_COLUMN]].reset_index(drop=True)
    test_df_no_extra = test_df.drop(columns=[ORIGINAL_LABEL_COL, SOURCE_DAY_COLUMN])

    # 4. Preprocessing del test set IDENTICO a quello di run_baseline.py
    #    (stessi tre passi interni di process()), ma eseguito passo-passo
    #    invece che con process() per poter tracciare quali righe
    #    sopravvivono a _drop_invalid_rows e tenerle allineate alle colonne
    #    extra (process() fa un reset_index interno che farebbe perdere
    #    l'allineamento riga-per-riga).
    print("[3/6] Preprocessing del test set (drop metadata, cast numerico, feature ingegnerizzate)...")
    test_features = preprocessor._drop_metadata_columns(test_df_no_extra)
    test_features = preprocessor._convert_feature_columns_to_numeric(test_features)
    test_features = preprocessor._add_engineered_features(test_features)

    valid_mask = ~test_features.replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        print(f"      [NOTA] {n_dropped} righe scartate per NaN/inf (stesso criterio di "
              f"_drop_invalid_rows), escluse anche dalle colonne extra per restare allineate.")

    test_features = test_features[valid_mask.values].reset_index(drop=True)
    extra_cols_test = extra_cols_test[valid_mask.values].reset_index(drop=True)
    original_label_test = extra_cols_test[ORIGINAL_LABEL_COL]
    capture_day_test = extra_cols_test[SOURCE_DAY_COLUMN]

    # 5. Carica il modello già addestrato e la lista di feature che si aspetta
    #    (dopo la feature selection) dal pickle prodotto da run_baseline.py.
    print(f"[4/6] Caricamento modello già addestrato da '{PICKLE_PATH}'...")
    with open(PICKLE_PATH, "rb") as f:
        metadata_pipeline = pickle.load(f)

    model = metadata_pipeline["modello_addestrato"]
    feature_columns = metadata_pipeline["features_mappate"]

    missing = set(feature_columns) - set(test_features.columns)
    if missing:
        raise KeyError(
            f"Colonne attese dal modello ma assenti nel test set ricostruito: {sorted(missing)}. "
            f"Verifica che RANDOM_SEED/TARGET_ROWS_PER_DAY/data_folder coincidano esattamente "
            f"con l'ultimo run di run_baseline.py che ha prodotto il .pkl."
        )

    X_test = test_features[feature_columns]
    y_test = test_features[TARGET_COL].astype(int)
    y_pred = model.predict(X_test)
    # Probabilità della classe "Attacco" (colonna 1), usate sia per il ROC-AUC
    # implicito sia per lo scan di soglia al passo 6.
    y_proba = model.predict_proba(X_test)[:, 1]

    diag = pd.DataFrame({
        "original_label": original_label_test,
        "capture_day": capture_day_test,
        "y_true": y_test.to_numpy(),
        "y_pred": y_pred,
        "y_proba": y_proba,
    })

    # 6a. FALSI NEGATIVI per sotto-tipo di attacco originale.
    print("[5/6] Calcolo Falsi Negativi per sotto-tipo di attacco originale...\n")
    attacks = diag[diag["y_true"] == 1].copy()
    attacks["is_fn"] = attacks["y_pred"] == 0

    per_subtype = (
        attacks.groupby("original_label")
        .agg(n_totale=("is_fn", "size"), n_falsi_negativi=("is_fn", "sum"))
    )
    per_subtype["recall_sottoclasse"] = 1 - per_subtype["n_falsi_negativi"] / per_subtype["n_totale"]
    per_subtype = per_subtype.sort_values("recall_sottoclasse")

    pd.set_option("display.width", 120)
    print("=" * 90)
    print("  1) FALSI NEGATIVI PER SOTTO-TIPO DI ATTACCO (etichetta multi-classe originale)")
    print("=" * 90)
    print(per_subtype.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 90)

    total_fn = int(attacks["is_fn"].sum())
    total_attacks = len(attacks)
    print(f"\nTotale Falsi Negativi: {total_fn} su {total_attacks} attacchi nel test set "
          f"({total_fn / total_attacks * 100:.2f}%).")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fn_path = os.path.join(OUTPUT_DIR, "diagnostica_falsi_negativi_per_sottotipo.csv")
    per_subtype.to_csv(fn_path)
    print(f"[OK] Tabella salvata in: '{fn_path}'")

    # 6b. FALSI POSITIVI per giorno di cattura (i Benign non hanno un
    #     sotto-tipo utile come gli attacchi, ma sapere SE i FP si
    #     concentrano sui due giorni "protetti" -- vedi
    #     DEFAULT_PROTECTED_MINORITY_DAYS in raw_csvdataloader.py -- o sono
    #     distribuiti ovunque è comunque diagnostico per capire se
    #     l'aumento di FP dopo la fix sul loading è un effetto collaterale
    #     concentrato o diffuso).
    print("\n[6/6] Calcolo Falsi Positivi per giorno di cattura...\n")
    benign = diag[diag["y_true"] == 0].copy()
    benign["is_fp"] = benign["y_pred"] == 1

    per_day = (
        benign.groupby("capture_day")
        .agg(n_benign_totale=("is_fp", "size"), n_falsi_positivi=("is_fp", "sum"))
    )
    per_day["fp_rate"] = per_day["n_falsi_positivi"] / per_day["n_benign_totale"]
    per_day = per_day.sort_values("fp_rate", ascending=False)

    print("=" * 90)
    print("  2) FALSI POSITIVI PER GIORNO DI CATTURA (traffico Benign scambiato per Attacco)")
    print("=" * 90)
    print(per_day.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 90)

    total_fp = int(benign["is_fp"].sum())
    total_benign = len(benign)
    print(f"\nTotale Falsi Positivi: {total_fp} su {total_benign} record Benign nel test set "
          f"({total_fp / total_benign * 100:.2f}%).")

    fp_path = os.path.join(OUTPUT_DIR, "diagnostica_falsi_positivi_per_giorno.csv")
    per_day.to_csv(fp_path)
    print(f"[OK] Tabella salvata in: '{fp_path}'")

    # 6c. SCAN DI SOGLIA: nessun retraining, solo ricalcolo di
    #     precision/recall/F1 sulle probabilità già predette a soglie
    #     diverse da 0.5 (il default implicito di model.predict()). Utile
    #     per capire se esiste un compromesso migliore dell'87.02% di F1
    #     osservato a soglia 0.5, spostando il punto operativo verso più
    #     recall o più precision a seconda di cosa serve.
    print("\n[SCAN SOGLIA] Precision/Recall/F1 a soglie diverse (nessun retraining)...\n")
    y_true_arr = diag["y_true"].to_numpy()
    y_proba_arr = diag["y_proba"].to_numpy()

    scan_rows = []
    for t in THRESHOLDS_TO_SCAN:
        y_pred_t = (y_proba_arr >= t).astype(int)
        scan_rows.append({
            "soglia": t,
            "precision": precision_score(y_true_arr, y_pred_t, zero_division=0),
            "recall": recall_score(y_true_arr, y_pred_t, zero_division=0),
            "f1": f1_score(y_true_arr, y_pred_t, zero_division=0),
        })
    scan_df = pd.DataFrame(scan_rows).set_index("soglia")

    print("=" * 90)
    print("  3) SCAN DI SOGLIA (0.50 = soglia di default usata da model.predict())")
    print("=" * 90)
    print(scan_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 90)

    best_f1_threshold = scan_df["f1"].idxmax()
    print(f"\nSoglia con F1 migliore nello scan: {best_f1_threshold:.2f} "
          f"(F1={scan_df.loc[best_f1_threshold, 'f1']:.4f}, contro F1={scan_df.loc[0.50, 'f1']:.4f} "
          f"a soglia 0.50). Scegli comunque la soglia in base a cosa conta di più per il tuo caso "
          f"d'uso (più recall = meno intrusioni mancate ma più falsi allarmi, e viceversa), non "
          f"solo in base al massimo di F1.")

    scan_path = os.path.join(OUTPUT_DIR, "diagnostica_scan_soglia.csv")
    scan_df.to_csv(scan_path)
    print(f"[OK] Tabella salvata in: '{scan_path}'")


if __name__ == "__main__":
    main()