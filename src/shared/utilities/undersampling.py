"""
Under-sampling della classe maggioritaria (Benign), applicato SOLO al train
set dopo lo split stratificato -- mai al test set, altrimenti la valutazione
finale non misurerebbe più le prestazioni sulla distribuzione reale dei dati.

CONTESTO: sostituisce la deduplicazione (deduplication.py, rimossa da questo
lavoro) come meccanismo di bilanciamento/riduzione del dataset. A differenza
della deduplicazione -- che toglieva soprattutto righe di classe Attacco
(fino al 75% su alcuni giorni con traffico ripetitivo, es. bruteforce via
Patator) peggiorando lo sbilanciamento e correlata a un possibile calo di
recall (80.98% osservato) -- l'under-sampling agisce esclusivamente sulla
classe MAGGIORITARIA (Benign), lasciando intatta ogni riga di Attacco.

PERCHÉ PRIMA DELLA FEATURE SELECTION: CICIDSFeatureSelector allena una
foresta preliminare per stimare l'importanza OOB di ciascuna feature. Se
quella foresta vede un train set ancora sbilanciato, le stime di importanza
risultano distorte verso i pattern predittivi della classe maggioritaria.
Applicare l'under-sampling PRIMA della feature selection garantisce che
anche quella fase lavori su dati già bilanciati, non solo il tuning finale.
"""
import pandas as pd


def undersample_majority_class(
    train_df: pd.DataFrame,
    target_column: str = "Label",
    majority_class: int = 0,
    minority_class: int = 1,
    ratio: float = 1.0,
    random_state: int = 123,
) -> pd.DataFrame:
    """
    Riduce per campionamento casuale la classe maggioritaria a
    'ratio' volte la dimensione della classe minoritaria, lasciando
    quest'ultima interamente intatta.

    ratio: rapporto desiderato maggioritaria/minoritaria DOPO
    l'under-sampling. ratio=1.0 (default) produce un bilanciamento 1:1,
    lo standard di riferimento più comune in letteratura sull'imbalanced
    learning (lo stesso punto di partenza usato es. nel paper originale di
    SMOTE, Chawla et al. 2002, come baseline di confronto prima di
    esplorare rapporti diversi). Se il train set contiene già una
    proporzione di maggioritaria inferiore a 'ratio', non viene fatto
    nulla (non si sovracampiona per raggiungere il rapporto: l'unico scopo
    di questa funzione è RIDURRE la maggioritaria, mai aumentare la
    minoritaria).

    Ritorna un nuovo DataFrame (maggioritaria sotto-campionata + minoritaria
    intatta), mescolato casualmente (altrimenti tutte le righe minoritarie
    finirebbero in blocco all'inizio o alla fine, il che sarebbe innocuo per
    scikit-learn ma comunque una scelta di igiene dei dati non giustificata).
    """
    if target_column not in train_df.columns:
        raise KeyError(f"Target '{target_column}' non trovato nel DataFrame.")
    if ratio <= 0:
        raise ValueError("ratio deve essere maggiore di 0.")

    majority_df = train_df[train_df[target_column] == majority_class]
    minority_df = train_df[train_df[target_column] == minority_class]

    n_minority = len(minority_df)
    n_majority = len(majority_df)
    target_n_majority = int(round(n_minority * ratio))

    print(f"[UNDERSAMPLING] Classe maggioritaria (label={majority_class}): "
          f"{n_majority} righe. Classe minoritaria (label={minority_class}): "
          f"{n_minority} righe. Rapporto attuale: {n_majority / n_minority:.2f}:1.")

    if target_n_majority >= n_majority:
        print(f"[UNDERSAMPLING] Il rapporto richiesto ({ratio}:1) richiederebbe "
              f"{target_n_majority} righe maggioritarie, >= alle {n_majority} "
              f"disponibili: nessun campionamento necessario, dataset invariato.")
        return train_df.copy()

    majority_sampled = majority_df.sample(n=target_n_majority, random_state=random_state)
    result = pd.concat([majority_sampled, minority_df], ignore_index=False)
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"[UNDERSAMPLING] Classe maggioritaria ridotta a {target_n_majority} righe "
          f"(rapporto finale {ratio}:1). Righe totali nel train set: "
          f"{n_majority} + {n_minority} -> {len(result)}.")

    return result