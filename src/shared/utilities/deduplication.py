"""
Deduplicazione di righe quasi-duplicate PRIMA dello split train/test.

CONTESTO E GIUSTIFICAZIONE:
analyze_train_test_leakage.py ha misurato che ~25% delle righe di test hanno
una controparte quasi identica in train (stesso arrotondamento a 6 decimali
sulle feature numeriche) — e che una percentuale quasi identica (~25%)
esiste GIÀ dentro il train set da solo, confrontato con se stesso. Questo
significa che la ridondanza non è un artefatto dello split, ma una proprietà
strutturale del dataset CICIDS2018 grezzo (fenomeno documentato in
letteratura, es. Engelen et al. 2021, "Troubleshooting an Intrusion
Detection Dataset" — traffico di sfondo benigno molto ripetitivo produce
molti flussi quasi identici anche a distanza di ore).

Un'analisi complementare (stessa diagnostica, sezione A) ha inoltre mostrato
che l'84% circa dei match train<->test è tra righe dello STESSO giorno di
cattura: uno split per giorno interno eliminerebbe la maggior parte del
leakage misurato, ma non tutto (resta un ~4% cross-giorno irriducibile per
qualunque criterio di split), e ha un problema pratico proprio (un solo
giorno, Thuesday-20-02, è da solo circa metà dell'intero dataset — uno split
per giorno intero non può avvicinarsi a un 80/20 senza forte sbilanciamento
di volume). La deduplicazione attacca la causa (la ridondanza stessa), non
il sintomo (come viene fatto lo split) — per questo è la correzione scelta.

COSA FA QUESTA FUNZIONE, IN CONCRETO:
  1. Costruisce una vista "come la vedrebbe il modello": esclude le colonne
     metadata (IP, porta, timestamp, MAC — la STESSA lista di parole chiave
     di CICIDSPreprocessor) e converte a numerico, SENZA modificare il
     DataFrame originale — la vista serve solo per decidere quali righe
     scartare; le colonne originali restano intatte su quelle che
     sopravvivono.
  2. Arrotonda le feature numeriche a 'decimals' decimali (default 6 — lo
     stesso livello "quasi esatto" usato nella diagnostica: la scelta più
     conservativa, rimuove solo ciò che è praticamente identico, non righe
     solo "abbastanza simili").
  3. Raggruppa le righe per hash della riga arrotondata (pandas.util.
     hash_pandas_object, vettorizzato — stessa tecnica della diagnostica);
     per ogni gruppo con più di una riga tiene UNA sola riga (la prima
     nell'ordine originale) e scarta le altre.
  4. Ritorna il DataFrame ORIGINALE (tutte le colonne, non la vista ridotta)
     filtrato alle sole righe sopravvissute.

COSA NON FA (limiti onesti, da dichiarare in tesi):
  - Non distingue "duplicato per davvero" da "coincidenza statistica tra due
    flussi indipendenti con valori simili": a 6 decimali su feature come
    Flow Duration/IAT la probabilità di collisione casuale è trascurabile,
    ma non è formalmente zero. È un'approssimazione pratica, non una prova
    di identità semantica tra due flussi.
  - Riduce la dimensione effettiva del dataset (~25% in meno, dai numeri
    misurati): il bilancio delle classi dopo la deduplicazione va sempre
    riletto dall'output di questa funzione, perché non è garantito che
    Benign e Attacco vengano ridotti nella stessa proporzione.
  - Va chiamata PRIMA dello split (su df_binarized, sull'intero dataset non
    ancora diviso), non su train/test separatamente: deduplicare i due lati
    in modo indipendente non risolverebbe nulla, perché il problema è
    proprio che la STESSA riga (o quasi) può comparire su entrambi i lati.
"""
import pandas as pd

DEFAULT_METADATA_KEYWORDS = ["timestamp", "flow id", "ip", "port", "mac"]


def remove_near_duplicate_rows(
    df: pd.DataFrame,
    target_column: str = "Label",
    metadata_keywords=None,
    decimals: int = 6,
) -> pd.DataFrame:
    """
    Rimuove le righe quasi-duplicate (stesso arrotondamento a 'decimals'
    decimali sulle feature numeriche, colonne metadata escluse dal
    confronto — stessa lista di parole chiave di CICIDSPreprocessor). Tiene
    UNA riga per gruppo di duplicati, la prima nell'ordine originale del
    DataFrame. Va chiamata PRIMA dello split train/test.

    Ritorna il DataFrame originale (tutte le colonne) filtrato alle sole
    righe sopravvissute. L'indice pandas delle righe tenute NON viene
    resettato: lo split successivo (StratifiedDataSplitter) usa .iloc
    posizionale internamente, quindi funziona correttamente a prescindere
    da quali etichette di indice sopravvivano.
    """
    metadata_keywords = metadata_keywords or DEFAULT_METADATA_KEYWORDS

    feature_cols = [
        col for col in df.columns
        if col != target_column
        and not any(k in str(col).lower() for k in metadata_keywords)
    ]

    numeric_view = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    rounded = numeric_view.round(decimals)
    row_hashes = pd.util.hash_pandas_object(rounded, index=False)

    n_before = len(df)
    keep_mask = ~row_hashes.duplicated(keep="first").values
    df_dedup = df[keep_mask].copy()
    n_after = len(df_dedup)
    n_removed = n_before - n_after

    pct_removed = (n_removed / n_before * 100) if n_before else 0.0
    print(f"[DEDUP] Righe quasi-duplicate rimosse (decimals={decimals}): "
          f"{n_removed} su {n_before} ({pct_removed:.2f}%). "
          f"Righe rimanenti: {n_after}.")

    if target_column in df_dedup.columns and n_after > 0:
        class_pct = (df_dedup[target_column].value_counts(normalize=True) * 100).round(2)
        print(f"[DEDUP] Bilancio classi DOPO la deduplicazione: {class_pct.to_dict()}")

    return df_dedup