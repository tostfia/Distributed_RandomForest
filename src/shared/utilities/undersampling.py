"""
Under-sampling della classe maggioritaria (Benign), applicato SOLO al train
set dopo lo split stratificato -- mai al test set, altrimenti la valutazione
finale non misurerebbe più le prestazioni sulla distribuzione reale dei dati.

LIBRERIA: imbalanced-learn (imblearn.under_sampling.RandomUnderSampler),
progetto satellite di scikit-learn (organizzazione scikit-learn-contrib,
non incluso nel pacchetto core sklearn) -- da dichiarare esplicitamente
come dipendenza aggiuntiva in relazione (installazione:
`pip install imbalanced-learn --break-system-packages`).
"""
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

def undersample_majority_class(
    train_df: pd.DataFrame,
    target_column: str = "Label",
    majority_class: int = 0,
    minority_class: int = 1,
    ratio: float = 1.0,
    random_state: int = 123,
) -> pd.DataFrame:
    """
    Riduce per campionamento casuale (imblearn.RandomUnderSampler) la classe
    maggioritaria a 'ratio' volte la dimensione della classe minoritaria,
    lasciando quest'ultima interamente intatta.

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

    NOTA SULLA CONVERSIONE DI PARAMETRO -- verificata sulla documentazione
    ufficiale imbalanced-learn 0.14.2, non data per scontata: il parametro
    nativo 'sampling_strategy' di RandomUnderSampler, se passato come
    float, ha una definizione DIVERSA e nell'ordine INVERSO rispetto al
    nostro 'ratio' (è minoritaria/maggioritaria-dopo, non
    maggioritaria/minoritaria). Per evitare quella conversione ambigua
    (una vecchia issue del progetto stesso segnalava la formula opposta in
    una versione precedente della documentazione, poi corretta), si usa
    qui la forma a DIZIONARIO di 'sampling_strategy'
    ({classe: numero_campioni_desiderato}), che specifica direttamente il
    numero di righe finali della sola classe maggioritaria senza alcuna
    conversione implicita da sbagliare.

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

    # Verifica che ogni riga sia stata contata in una delle due classi: se
    # target_column contenesse un valore diverso da majority_class/
    # minority_class (es. un'etichetta residua non binarizzata correttamente
    # a monte), quelle righe sparirebbero silenziosamente dal risultato
    # senza alcun avviso. Con binarize_target() questo non dovrebbe mai
    # accadere, ma il controllo costa nulla ed evita un errore silenzioso.
    n_unaccounted = len(train_df) - len(majority_df) - len(minority_df)
    if n_unaccounted > 0:
        raise ValueError(
            f"{n_unaccounted} righe con '{target_column}' diverso sia da "
            f"majority_class={majority_class} sia da minority_class={minority_class}: "
            f"verificare che il target sia stato binarizzato correttamente a monte "
            f"prima di chiamare questa funzione."
        )

    n_minority = len(minority_df)
    n_majority = len(majority_df)

    # Guardia esplicita PRIMA di qualunque divisione per n_minority (il
    # rapporto stampato subito sotto lo farebbe implicitamente): con
    # n_minority=0 il messaggio d'errore nativo di Python (ZeroDivisionError
    # generico) non spiegherebbe la causa reale -- questo controllo dà un
    # errore leggibile invece di un crash criptico.
    if n_minority == 0:
        raise ValueError(
            f"Nessuna riga di classe minoritaria (label={minority_class}) nel train set: "
            f"impossibile calcolare un rapporto di bilanciamento. Verificare la pipeline "
            f"a monte (split, campionamento) prima di questo passo."
        )

    # Avviso (non blocca l'esecuzione) se le etichette majority_class/
    # minority_class non corrispondono a quale classe è REALMENTE più
    # numerosa nei dati: un'inversione qui produrrebbe un risultato valido
    # ma probabilmente non quello inteso dal chiamante.
    if n_majority < n_minority:
        print(f"[UNDERSAMPLING] [ATTENZIONE] majority_class={majority_class} ha meno righe "
              f"({n_majority}) di minority_class={minority_class} ({n_minority}): verificare "
              f"che le due etichette non siano state invertite per errore.")

    target_n_majority = int(round(n_minority * ratio))

    print(f"[UNDERSAMPLING] Classe maggioritaria (label={majority_class}): "
          f"{n_majority} righe. Classe minoritaria (label={minority_class}): "
          f"{n_minority} righe. Rapporto attuale: {n_majority / n_minority:.2f}:1.")

    if target_n_majority >= n_majority:
        print(f"[UNDERSAMPLING] Il rapporto richiesto ({ratio}:1) richiederebbe "
              f"{target_n_majority} righe maggioritarie, >= alle {n_majority} "
              f"disponibili: nessun campionamento necessario, dataset invariato.")
        return train_df.copy()

    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    rus = RandomUnderSampler(
        sampling_strategy={majority_class: target_n_majority},
        random_state=random_state,
        replacement=False,
    )
    X_res, y_res = rus.fit_resample(X, y)

    result = X_res.copy()
    result[target_column] = y_res
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"[UNDERSAMPLING] Classe maggioritaria ridotta a {target_n_majority} righe "
          f"(rapporto finale {ratio}:1, via imblearn.RandomUnderSampler). Righe totali "
          f"nel train set: {n_majority} + {n_minority} -> {len(result)}.")

    return result