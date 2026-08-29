"""
Diagnostica: verifica QUANTITATIVA del leakage train/test nel dataset REALE
(CICIDS2018) e conferma se la deduplicazione (deduplication.py) lo risolve.

Questa versione esegue l'INTERA diagnostica DUE VOLTE sullo stesso dataset
caricato una sola volta da disco:
  - PRIMA: pipeline attuale, senza deduplicazione (per confronto/storico).
  - DOPO: stessa pipeline con remove_near_duplicate_rows() applicata prima
    dello split, esattamente come ora avviene in run_baseline.py,
    centralized.py e FederatedDataSplitter.split_and_shard().

Se la correzione funziona, il leakage misurato nel blocco "DOPO" deve
crollare rispetto al blocco "PRIMA" (dal ~25% osservato verso un valore
residuo vicino allo zero, quello atteso per puro caso). Il confronto finale
in coda allo script riassume i due risultati fianco a fianco.

CONTESTO (invariato dalla versione precedente):
I CSV sorgente sono divisi per giorno di cattura (10 file). RawCSVDataLoader
concatena tutti i CSV in un unico DataFrame SENZA tracciare la provenienza
per riga (nessuna colonna 'source_day' salvata), poi StratifiedDataSplitter
esegue uno split casuale stratificato SOLO sulla classe target
(StratifiedShuffleSplit), ignorando completamente tempo/giorno/sessione.

Flussi di rete vicini nel tempo (stessa sessione TCP, stesso attacco in
corso durante la cattura) tendono ad avere feature quasi identiche (Flow
Duration, IAT, Init Win Bytes, ecc.). Se uno di questi flussi finisce in
train e uno molto simile (o identico) finisce in test, il modello non sta
generalizzando su quel caso: lo sta riconoscendo. Le metriche sul test set
risultano quindi artificialmente ottimistiche in proporzione a quanti casi
così emergono.

COSA FA QUESTO SCRIPT, PER OGNI SCENARIO (PRIMA/DOPO):
  1. Replica la pipeline di run_baseline.py fino al preprocessing incluso
     (stesso loader, stesso sample_fraction, stesso seed, stesso split),
     MA tenendo traccia in più del file/giorno sorgente di ogni riga.
  2. Confronta X_train e X_test cercando righe duplicate o quasi-duplicate
     sulle feature numeriche, a più livelli di tolleranza (arrotondamento).
     Il confronto è O(n) via hashing (non un confronto quadratico
     riga-per-riga).
  3. ANALISI A -- STESSO GIORNO vs CROSS-GIORNO: per ogni riga di test
     duplicata, verifica se esiste almeno un match in train dello STESSO
     giorno di origine, oppure se tutti i match trovati provengono da
     giorni DIVERSI.
  4. ANALISI B -- RIDONDANZA INTRA-SET: misura quanti duplicati/near-
     duplicati esistono GIÀ dentro il train set da solo (e dentro il test
     set da solo), indipendentemente da come è stato fatto lo split.

INTERPRETAZIONE (vedi anche output a runtime):
  - Un tasso di duplicati ESATTI (arrotondamento a molti decimali) basso è
    normale anche con split perfettamente casuale, per puro caso su feature
    a bassa cardinalità -- non è di per sé prova di leakage.
  - Un tasso alto di NEAR-duplicati (arrotondamento permissivo, pochi
    decimali) è il segnale da cercare: indica righe di test praticamente
    indistinguibili da righe di train nello spazio delle feature usate dal
    modello.
  - Questo script misura CORRELAZIONE (duplicati), non causalità: non prova
    da solo che le metriche siano gonfiate, ma quantifica quanto è ampio il
    fenomeno che potrebbe gonfiarle -- il numero stesso è già informativo
    per la relazione, a prescindere da come lo si interpreta poi.

CITAZIONI CORRETTE per la relazione (vedi verifica bibliografica fatta in
chat -- la citazione "Engelen et al. 2021" di una versione precedente di
questo docstring era imprecisa, è sul dataset 2017 non 2018):
  - Lanvin et al. (2023), "Errors in the CICIDS2017 dataset and the
    significant differences in detection performances it makes": trovano
    ~497.000 pacchetti duplicati/giorno; correggerli fa scendere il recall
    del 20%, prova diretta che punteggi "quasi perfetti" derivano da
    overfitting sugli artefatti di duplicazione, non da vera capacità di
    rilevamento.
  - Liu et al. (2022), "Error prevalence in NIDS datasets: A case study on
    CIC-IDS-2017 and CSE-CIC-IDS-2018": copre esplicitamente la versione
    2018 usata in questo progetto.

Uso:
    python analyze_train_test_leakage.py

Nota: richiede DATASET_LOCAL_PATH puntato ai CSV reali, come run_baseline.py.
"""
import os
import pandas as pd

from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.datasplitter import StratifiedDataSplitter
from src.shared.utilities.deduplication import remove_near_duplicate_rows, DEFAULT_METADATA_KEYWORDS

RANDOM_SEED = 123
TEST_SIZE = 0.2
SAMPLE_FRACTION = 0.05
TARGET_COL = "Label"

# Livelli di arrotondamento testati sulle feature numeriche prima
# dell'hashing. 6 decimali cattura solo duplicati praticamente esatti
# (rumore di rappresentazione float); 1 decimale è molto più permissivo e
# cattura anche near-duplicate con piccole differenze reali tra loro.
ROUNDING_LEVELS = [6, 3, 1]

# Le analisi A (stesso giorno vs cross-giorno) e B (ridondanza intra-set)
# richiedono calcoli aggiuntivi (hash train<->train, train<->giorno): girano
# solo sui due livelli estremi (quasi-esatto e permissivo), non su tutti e
# tre, per contenere il tempo totale.
DETAILED_ROUNDING_LEVELS = [6, 1]


def load_with_day_tracking() -> pd.DataFrame:
    """
    Replica RawCSVDataLoader.load() ma aggiunge due colonne extra per ogni
    riga: 'source_day' (nome del file di origine) e '_diag_orig_row_id'
    (indice originale come valore, non come indice pandas). La seconda è
    necessaria perché CICIDSPreprocessor._drop_invalid_rows() fa
    df.dropna().reset_index(drop=True): l'indice pandas viene azzerato e
    rinumerato, quindi non può essere usato per riallineare 'source_day'
    dopo il preprocessing. '_diag_orig_row_id' sopravvive invece perché è
    una colonna DATI come le altre (non un'etichetta di indice), e non
    corrisponde a nessuna delle metadata_keywords del preprocessor
    ('timestamp', 'flow id', 'ip', 'port', 'mac'), quindi non viene scartata
    da _drop_metadata_columns().
    """
    data_folder = os.environ.get("DATASET_LOCAL_PATH", "./dataset_cache")
    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            f"Cartella dataset non trovata: '{data_folder}'. "
            f"Imposta DATASET_LOCAL_PATH come per run_baseline.py."
        )

    loader = RawCSVDataLoader(data_url=data_folder, sample_fraction=SAMPLE_FRACTION, dataset_seed=RANDOM_SEED)
    sources = loader._discover_sources()
    print(f"[1/4] Sorgenti trovate: {len(sources)}")

    chunks = []
    for source in sources:
        print(f"   - Lettura: {source}")
        df_temp = loader._read_single_csv(source=source).copy()
        df_temp["source_day"] = os.path.basename(source)
        chunks.append(df_temp)

    df = pd.concat(chunks, ignore_index=True)
    df["_diag_orig_row_id"] = df.index.values
    print(f"[OK] Caricamento con tracciamento giorno completato: {df.shape[0]} righe totali.")
    return df


def prepare_train_test_with_day(df_raw: pd.DataFrame, apply_dedup: bool):
    print(f"[2/4] Binarizzazione + {'DEDUPLICAZIONE + ' if apply_dedup else ''}"
          f"split stratificato + preprocessing...")
    preprocessor = CICIDSPreprocessor(target_column=TARGET_COL)
    splitter = StratifiedDataSplitter(target_column=TARGET_COL, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    # Mappa indipendente dall'indice pandas: sopravvive a qualunque
    # reset_index() successivo, incluso quello dentro preprocessor.process().
    day_lookup = df_raw.set_index("_diag_orig_row_id")["source_day"]

    df_binarized = preprocessor.binarize_target(df_raw)

    if apply_dedup:
        # Stesso punto esatto in cui va chiamata nella pipeline reale:
        # PRIMA dello split, sul dataset intero non ancora diviso.
        #
        # IMPORTANTE (solo per QUESTO script diagnostico, non per la
        # pipeline reale): '_diag_orig_row_id' e 'source_day' sono colonne
        # extra aggiunte qui per il tracciamento del giorno -- non esistono
        # in run_baseline.py/centralized.py. '_diag_orig_row_id' in
        # particolare è per costruzione UNICA per ogni riga: se la si
        # lasciasse nel confronto, ogni riga risulterebbe "diversa da tutte
        # le altre" per definizione, e la deduplicazione non troverebbe mai
        # nulla da rimuovere. Vanno quindi escluse esplicitamente dal
        # confronto (restano comunque nel DataFrame restituito, servono
        # dopo per riallineare 'source_day').
        df_binarized = remove_near_duplicate_rows(
            df_binarized,
            target_column=TARGET_COL,
            metadata_keywords=DEFAULT_METADATA_KEYWORDS + ["_diag_orig_row_id", "source_day"],
        )

    train_df, test_df = splitter.split(df_binarized)

    # 'source_day' non è una feature del modello e va tolta prima di
    # process() (altrimenti pd.to_numeric la trasformerebbe in NaN,
    # facendo scartare TUTTE le righe come "invalide"). '_diag_orig_row_id'
    # invece resta: è già numerica, sopravvive intatta a process() e serve
    # a riallineare 'source_day' subito dopo.
    train_df = preprocessor.process(train_df.drop(columns=["source_day"]))
    test_df = preprocessor.process(test_df.drop(columns=["source_day"]))

    train_day = train_df["_diag_orig_row_id"].astype("int64").map(day_lookup)
    test_day = test_df["_diag_orig_row_id"].astype("int64").map(day_lookup)

    train_df = train_df.drop(columns=["_diag_orig_row_id"])
    test_df = test_df.drop(columns=["_diag_orig_row_id"])

    return train_df, test_df, train_day, test_day


def compute_row_hashes(df: pd.DataFrame, target_col: str, decimals: int) -> pd.Series:
    """
    Arrotonda le feature numeriche a 'decimals' decimali e calcola un hash
    per riga (pandas.util.hash_pandas_object, vettorizzato). Usato sia per
    il confronto train<->test sia per le analisi A e B, così l'hashing non
    viene ripetuto più volte per lo stesso (df, decimals).
    """
    feature_cols = [c for c in df.columns if c != target_col]
    rounded = df[feature_cols].round(decimals)
    return pd.util.hash_pandas_object(rounded, index=False)


def find_duplicate_mask(train_hashes: pd.Series, test_hashes: pd.Series) -> pd.Series:
    """
    Confronto O(n): verifica quali righe di test hanno lo stesso hash di
    almeno una riga di train. Nessun confronto quadratico riga-per-riga.
    """
    train_hash_set = set(train_hashes.values)
    return test_hashes.isin(train_hash_set)


def intra_set_duplicate_stats(hashes: pd.Series):
    """
    ANALISI B: quota di righe che hanno almeno un'ALTRA riga con lo stesso
    hash all'interno dello STESSO insieme (train da solo, o test da solo).
    Misura la ridondanza strutturale del dataset grezzo, indipendente da
    come è stato fatto lo split train/test.
    """
    counts = hashes.value_counts()
    dup_hashes = counts[counts > 1].index
    n_dup = int(hashes.isin(dup_hashes).sum())
    pct = n_dup / len(hashes) * 100 if len(hashes) else float("nan")
    return n_dup, pct


def same_day_vs_cross_day(test_hashes: pd.Series, test_day: pd.Series,
                           train_hashes: pd.Series, train_day: pd.Series,
                           dup_mask: pd.Series):
    """
    ANALISI A: per le righe di test duplicate (dup_mask=True), determina se
    esiste almeno un match in train dello STESSO giorno di origine, oppure
    se tutti i match trovati provengono da giorni DIVERSI. Risponde a "uno
    split per giorno risolverebbe il leakage misurato?": se la maggioranza
    dei match è stesso-giorno, sì (l'intero giorno finirebbe da un lato
    solo dello split, eliminando la coppia); se è cross-giorno, no.
    """
    train_tmp = pd.DataFrame({"hash": train_hashes.values, "day": train_day.values})
    hash_to_days = train_tmp.groupby("hash")["day"].apply(set)

    days_available = test_hashes.map(hash_to_days)

    n_same_day = 0
    n_cross_only = 0
    for is_dup, own_day, avail_days in zip(dup_mask.values, test_day.values, days_available.values):
        if not is_dup:
            continue
        if isinstance(avail_days, set) and own_day in avail_days:
            n_same_day += 1
        else:
            n_cross_only += 1

    return n_same_day, n_cross_only


def run_leakage_analysis(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          train_day: pd.Series, test_day: pd.Series, label: str) -> dict:
    """
    Esegue l'intera diagnostica (tabella train<->test, analisi A, analisi B)
    su una coppia (train_df, test_df) già pronta, e ritorna un riepilogo
    numerico per il confronto finale prima/dopo la deduplicazione.
    """
    print("\n" + "#" * 78)
    print(f"# {label}")
    print("#" * 78)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    print("\nRicerca duplicati/near-duplicati train<->test per livello di arrotondamento:")
    print("=" * 78)
    print(f"  {'Decimali':<10} | {'Test duplicate':<15} | {'% del test set'}")
    print("  " + "-" * 50)

    hashes_by_level = {}
    dup_mask_by_level = {}
    dup_pct_by_level = {}
    for decimals in ROUNDING_LEVELS:
        train_hashes = compute_row_hashes(train_df, TARGET_COL, decimals)
        test_hashes = compute_row_hashes(test_df, TARGET_COL, decimals)
        dup_mask = find_duplicate_mask(train_hashes, test_hashes)

        hashes_by_level[decimals] = (train_hashes, test_hashes)
        dup_mask_by_level[decimals] = dup_mask

        n_dup = int(dup_mask.sum())
        pct = n_dup / len(test_df) * 100 if len(test_df) else float("nan")
        dup_pct_by_level[decimals] = pct
        print(f"  {decimals:<10} | {n_dup:<15} | {pct:.3f}%")

    summary = {
        "label": label,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "dup_pct_by_level": dup_pct_by_level,
    }

    last_decimals = ROUNDING_LEVELS[-1]
    last_dup_mask = dup_mask_by_level[last_decimals]
    if not last_dup_mask.any():
        print("\n  Nessun duplicato/near-duplicato trovato nemmeno al livello più permissivo.")
        return summary

    dup_test_days = test_day[last_dup_mask.values]
    print(f"\n  Distribuzione per giorno delle {int(last_dup_mask.sum())} righe di test "
          f"duplicate (decimali={last_decimals}):")
    print(dup_test_days.value_counts().to_string())

    # --- ANALISI B: ridondanza intra-set ---
    print("\n" + "=" * 78)
    print("  B. RIDONDANZA INTRA-SET")
    print("=" * 78)
    print(f"  {'Decimali':<10} | {'Train dup %':<14} | {'Test dup %'}")
    print("  " + "-" * 50)
    for decimals in DETAILED_ROUNDING_LEVELS:
        train_hashes, test_hashes = hashes_by_level[decimals]
        _, pct_train = intra_set_duplicate_stats(train_hashes)
        _, pct_test = intra_set_duplicate_stats(test_hashes)
        print(f"  {decimals:<10} | {pct_train:<14.3f} | {pct_test:.3f}")

    # --- ANALISI A: stesso giorno vs cross-giorno ---
    print("\n" + "=" * 78)
    print("  A. STESSO GIORNO vs GIORNI DIVERSI")
    print("=" * 78)
    print(f"  {'Decimali':<10} | {'Stesso giorno':<15} | {'Solo cross-giorno':<19} | {'% stesso giorno'}")
    print("  " + "-" * 70)
    for decimals in DETAILED_ROUNDING_LEVELS:
        train_hashes, test_hashes = hashes_by_level[decimals]
        dup_mask = dup_mask_by_level[decimals]
        n_same, n_cross_only = same_day_vs_cross_day(test_hashes, test_day, train_hashes, train_day, dup_mask)
        n_total = n_same + n_cross_only
        pct_same = (n_same / n_total * 100) if n_total else float("nan")
        print(f"  {decimals:<10} | {n_same:<15} | {n_cross_only:<19} | {pct_same:.1f}%")

    return summary


def print_comparison(summary_before: dict, summary_after: dict):
    print("\n\n" + "=" * 78)
    print("  CONFRONTO FINALE: PRIMA vs DOPO LA DEDUPLICAZIONE")
    print("=" * 78)
    print(f"  {'':<12} | {'Train shape':<16} | {'Test shape':<16}")
    print("  " + "-" * 50)
    print(f"  {'PRIMA':<12} | {str(summary_before['train_shape']):<16} | {str(summary_before['test_shape']):<16}")
    print(f"  {'DOPO':<12} | {str(summary_after['train_shape']):<16} | {str(summary_after['test_shape']):<16}")

    print(f"\n  {'Decimali':<10} | {'% dup. PRIMA':<14} | {'% dup. DOPO':<13} | {'Riduzione'}")
    print("  " + "-" * 60)
    for decimals in ROUNDING_LEVELS:
        pct_before = summary_before["dup_pct_by_level"].get(decimals, float("nan"))
        pct_after = summary_after["dup_pct_by_level"].get(decimals, float("nan"))
        riduzione = pct_before - pct_after
        print(f"  {decimals:<10} | {pct_before:<14.3f} | {pct_after:<13.3f} | -{riduzione:.3f} punti")

    print("\n  Interpretazione: se la colonna 'DOPO' è vicina a zero (in particolare a")
    print("  decimali=6, il livello 'quasi esatto'), la deduplicazione ha eliminato il")
    print("  leakage strutturale misurato in precedenza. Un residuo piccolo e non-zero è")
    print("  atteso: rappresenta collisioni statistiche casuali, non più ridondanza reale.")


def main():
    df_raw = load_with_day_tracking()

    train_df_before, test_df_before, train_day_before, test_day_before = prepare_train_test_with_day(
        df_raw, apply_dedup=False
    )
    summary_before = run_leakage_analysis(
        train_df_before, test_df_before, train_day_before, test_day_before,
        "PRIMA (pipeline attuale, senza deduplicazione)",
    )

    train_df_after, test_df_after, train_day_after, test_day_after = prepare_train_test_with_day(
        df_raw, apply_dedup=True
    )
    summary_after = run_leakage_analysis(
        train_df_after, test_df_after, train_day_after, test_day_after,
        "DOPO (con remove_near_duplicate_rows prima dello split)",
    )

    print_comparison(summary_before, summary_after)


if __name__ == "__main__":
    main()