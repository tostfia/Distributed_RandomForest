import os
import json
import time
import numpy as np
import pickle
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # i tuoi print restano l'unico output

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score, precision_recall_curve, roc_curve, f1_score, confusion_matrix,
)
from src.shared.utilities.undersampling import undersample_majority_class
from src.shared.config import SystemConfig

# Import delle utility condivise e del loader con campionamento probabilistico
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
from src.shared.utilities.datasplitter import StratifiedDataSplitter
from src.shared.utilities.featureselection import CICIDSFeatureSelector

# ---------------------------------------------------------------------------
# IPERPARAMETRI DEL TASK SINTETICO DI REGRESSIONE.
#
# PRIMA: una configurazione dichiarata a priori (SYNTHETIC_REGRESSOR_REFERENCE_HP),
# poi (in una revisione successiva) un tuning OOB dedicato via Optuna. Il
# tuning è stato rimosso di nuovo: il dataset sintetico qui serve SOLO da
# stress-test di scalabilità (confronto baseline single-node vs cluster
# distribuito), non a trovare il modello con l'R² più alto -- non ha senso
# spendere 60 trial di ricerca per un obiettivo che non è "il modello
# migliore possibile".
#
# ORA: i default di scikit-learn per RandomForestRegressor, con UNA sola
# eccezione dichiarata (max_features, sotto) più n_estimators (mai un
# default, va sempre misurato -- vedi sotto REGRESSOR_DEFAULT_HP).
#
# NESSUNA feature selection (come prima): su Friedman #1 la separazione
# segnale/rumore è nota per costruzione dal generatore (5 feature
# informative fisse, il resto rumore puro -- vedi SyntheticDataLoader).
#
# ISOLAMENTO DELL'EFFETTO DI SCALABILITA': stessa idea di sempre -- gli
# iperparametri restano IDENTICI in ogni esperimento del cluster (1, 3, 5, 7
# worker), letti dalla copia per-task ("config_synthetic_regressor.json").
# ---------------------------------------------------------------------------
REGRESSOR_DEFAULT_HP = {
    # Tutti i valori sotto sono i DEFAULT ufficiali di
    # sklearn.ensemble.RandomForestRegressor (scikit-learn 1.6.1) --
    # nessuno scelto da noi, tranne max_features (vedi commento dedicato).
    "max_depth": None,
    "min_samples_split": 2,
    "criterion": "squared_error",
    "bootstrap": True,
    "max_samples": None,  # None = bootstrap sample_size = n_samples (default sklearn)
    # ECCEZIONE DICHIARATA: il default letterale di sklearn per la
    # regressione è max_features=1.0 (nessun sottoinsieme di feature ad
    # ogni split -- equivalente a "bagged trees", nessuna randomizzazione
    # sulle feature). La stessa User Guide di scikit-learn
    # (Ensemble methods, sez. 1.11.2.3) lo definisce sì "un buon default
    # empirico", ma nella frase successiva indica esplicitamente
    # un'alternativa standard in letteratura: "more randomness can be
    # achieved by setting smaller values (e.g. 0.3 is a typical default in
    # the literature)" -- che coincide con la raccomandazione classica di
    # Breiman per la regressione (m ≈ p/3). Non è quindi un allontanamento
    # dai default, ma la scelta della seconda alternativa già documentata
    # dalla stessa fonte. Necessario anche per motivi pratici: con
    # max_features=1.0 un solo fit di produzione (n_samples~1.000.000,
    # n_estimators~100) è stato misurato empiricamente in ordine di
    # 35-40 minuti; con max_features=1/3 scende a ~16 minuti (ancora un
    # carico di lavoro sostanzioso, in linea con lo scopo di stress-test).
    "max_features": 1 / 3,
}


# ---------------------------------------------------------------------------
# RICERCA IPERPARAMETRICA BASATA SU OOB (invece di k-fold Cross-Validation)
#
# GIUSTIFICAZIONE TEORICA (Breiman 2001, Sec. 3.1 — "Using out-of-bag
# estimates to monitor error, strength, and correlation"):
#   "the out-of-bag estimate is as accurate as using a test set of the same
#   size as the training set. Therefore, using the out-of-bag error estimate
#   removes the need for a set aside test set."
#   "unlike cross-validation, where bias is present but its extent unknown,
#   the out-of-bag estimates are unbiased."
#
# In pratica: ogni albero della foresta è addestrato su un bootstrap sample,
# lasciando fuori (out-of-bag) circa un terzo dei dati per quell'albero — uno
# split train/validation "gratuito" e diverso per ogni albero, GIA' incluso
# nel fit di un RandomForestClassifier con bootstrap=True. Non serve rifare
# k fit separati come nella k-fold CV: un solo fit per combinazione di
# iperparametri basta per ottenere una stima non distorta dell'errore di
# generalizzazione (a differenza della CV, il cui bias è presente ma non
# quantificabile secondo il paper).
#
# Conseguenza pratica per il budget di tempo: a parità di fit totali
# eseguibili su CPU, la ricerca OOB esplora ~5x più combinazioni della
# RandomizedSearchCV con cv=5 (1 fit invece di 5 per combinazione).
#
# VINCOLO: la stima OOB richiede bootstrap=True (senza campionamento
# bootstrap non esistono campioni "out-of-bag" da definire). Per questo la
# griglia di ricerca qui sotto include solo bootstrap=True — è una
# restrizione coerente con l'algoritmo originale di Breiman (Definition 1.1:
# ogni foresta di Breiman usa il bootstrap, "the random vector Θ... resulting
# in bagging" nella sua stessa descrizione), non solo una scorciatoia
# computazionale.
# ---------------------------------------------------------------------------

def _oob_classification_metrics(rf, y_train):
    """
    Calcola le metriche OOB (accuracy, precision, recall, F1) da un
    RandomForestClassifier con oob_score=True, escludendo correttamente le
    righe prive di copertura OOB (mai out-of-bag per nessun albero).

    ATTENZIONE ALLA VERSIONE DI SCIKIT-LEARN -- verificato empiricamente su
    questo progetto (scikit-learn 1.6.1, test controllato con copertura OOB
    incompleta):
        Righe con somma zero:  11
        Righe con NaN:          0
    Su questa versione, le righe prive di copertura OOB vengono riempite con
    [0., 0.] (tutte le classi a zero), NON con NaN -- a differenza di quanto
    riportato dalla documentazione ufficiale più recente di scikit-learn
    (>=1.9.x circa), che descrive un riempimento a NaN. Un secondo test ha
    inoltre confermato che l'attributo nativo rf.oob_score_ EREDITA questo
    bias su questa versione (0.38000, identico all'accuracy "naive" calcolata
    includendo le righe non coperte come predizioni valide di classe 0; NON
    0.38462, l'accuracy corretta escludendole) -- quindi la correzione
    manuale qui sotto non è un accorgimento superfluo, è necessaria.

    Il controllo copre ENTRAMBI i comportamenti (somma zero E NaN), non solo
    quello osservato sulla versione attualmente installata: resta corretto
    anche se l'ambiente di esecuzione dovesse cambiare versione di
    scikit-learn in futuro, senza dover ridiagnosticare da capo.
    """
    oob_decision = rf.oob_decision_function_

    zero_filled = oob_decision.sum(axis=1) == 0
    nan_filled = np.isnan(oob_decision).any(axis=1)
    valid_mask = ~(zero_filled | nan_filled)

    n_missing_oob = int((~valid_mask).sum())

    if not valid_mask.any():
        return {"oob_accuracy": float("nan"), "oob_precision": float("nan"),
                "oob_recall": float("nan"), "oob_f1": float("nan"),
                "n_missing_oob": n_missing_oob}

    oob_pred = np.argmax(oob_decision[valid_mask], axis=1)
    y_valid = np.asarray(y_train)[valid_mask]

    return {
        "oob_accuracy": float(np.mean(oob_pred == y_valid)),
        "oob_precision": precision_score(y_valid, oob_pred, zero_division=0),
        "oob_recall": recall_score(y_valid, oob_pred, zero_division=0),
        "oob_f1": f1_score(y_valid, oob_pred, zero_division=0),
        "n_missing_oob": n_missing_oob,
    }

def optuna_oob_hyperparameter_search(train_df, target_col, n_trials, random_state,
                                      importance_threshold, multicollinearity_distance_threshold,
                                      refit_metric="f1", n_jobs=None):
    """
    Sostituisce ParameterSampler + loop con una ricerca Optuna (TPE) sullo
    stesso spazio di iperparametri, stessa giustificazione teorica (Breiman
    2001 Sec. 3.1): un solo fit per trial, oob_score=True, nessuna k-fold CV.
    TPE campiona in modo informato dai trial precedenti invece che
    uniformemente, quindi a parità di budget converge su combinazioni
    migliori del campionamento puramente casuale.

    FEATURE SELECTION SPOSTATA DENTRO LA RICERCA -- allineamento al pattern
    ufficiale scikit-learn (SelectFromModel dentro una Pipeline, rifittato
    per ogni combinazione esplorata da GridSearchCV/RandomizedSearchCV,
    invece che fissato una volta prima del tuning): la selezione delle
    feature ora si muove insieme alla ricerca degli iperparametri, non è più
    calcolata una sola volta con una configurazione fissa (max_features='sqrt')
    scollegata da quella che il tuning sceglierà poi come ottima.

    A DIFFERENZA del pattern SelectFromModel di sklearn (importanza Gini,
    nessuna riduzione di ridondanza), qui si mantiene il metodo proprio di
    questo lavoro (OOB permutation importance + clustering gerarchico per
    multicollinearità, vedi CICIDSFeatureSelector) -- si segue il PRINCIPIO
    strutturale raccomandato da sklearn (selezione e tuning uniti), non lo
    strumento specifico, per non regredire a un criterio di selezione meno
    rigoroso di quello già scelto e giustificato altrove in questo lavoro.

    COSTO CONTROLLATO CON UNA CACHE: rifare la permutation importance ad ogni
    singolo trial (60 volte) moltiplicherebbe il tempo di calcolo di un
    fattore ~60x (3-9 minuti a passata), non sostenibile nei tempi
    disponibili. Solo max_features cambia concretamente QUALI feature
    risultano importanti (sqrt vs log2 cambiano le feature candidate ad ogni
    split); gli altri iperparametri (n_estimators, min_samples_split,
    criterion, class_weight, max_samples) influenzano quanto bene il modello
    sfrutta le feature disponibili, non quali feature emergono come più
    informative. La feature selection viene quindi calcolata al più UNA
    VOLTA PER VALORE DI max_features (2 volte in totale, non 60), e riusata
    da tutti i trial che campionano lo stesso max_features.
    """
    if n_jobs is None:
        cpu_count = os.cpu_count() or 2
        n_jobs = max(1, cpu_count - 1)

    results = []
    # cache: max_features -> (CICIDSFeatureSelector fittato, train_df già ridotto)
    feature_selection_cache = {}

    def get_or_compute_feature_selection(max_features):
        if max_features not in feature_selection_cache:
            print(f"\n[Feature Selection] Nessuna selezione in cache per max_features="
                  f"'{max_features}' -- calcolo ora (OOB permutation importance + "
                  f"riduzione multicollinearità, foresta preliminare con questo stesso "
                  f"max_features)...")
            fs = CICIDSFeatureSelector(
                target_column=target_col,
                importance_threshold=importance_threshold,
                rf_random_state=random_state,
                rf_max_features=max_features,
                reduce_multicollinearity=True,
                multicollinearity_distance_threshold=multicollinearity_distance_threshold,
                # Nome distinto per max_features: senza questo, la seconda
                # chiamata a questa cache (per l'altro valore di
                # max_features) sovrascriverebbe silenziosamente il
                # dendrogramma della prima (bug corretto in
                # CICIDSFeatureSelector).
                dendrogram_plot_path=f"feature_correlation_dendrogram_{max_features}.png",
            )
            train_selected = fs.fit_transform(train_df)
            feature_selection_cache[max_features] = (fs, train_selected)
            print(f"[Feature Selection] Cache popolata per max_features='{max_features}': "
                  f"{train_selected.shape[1] - 1} feature selezionate.\n")
        return feature_selection_cache[max_features]

    def objective(trial):
        # ---------------------------------------------------------------
        # GIUSTIFICAZIONE DELLA GRIGLIA -- non tutti i valori hanno lo
        # stesso status epistemico, va dichiarato esplicitamente quali sono
        # motivati e quali sono scelte pratiche non derivate dai dati:
        #
        #   ESAUSTIVI/TEORICAMENTE MOTIVATI (non un sottoinsieme arbitrario):
        #   - max_features ['sqrt','log2']: le due convenzioni standard per
        #     classificazione (Breiman 2001, 'sqrt' vicina a log2(M)+1).
        #   - criterion ['gini','entropy']: uniche due opzioni che
        #     scikit-learn offre per la classificazione.
        #   - class_weight: NON esplorato (fissato a None). Il train set
        #     arriva già bilanciato 1:1 dall'undersampling a monte
        #     (undersample_majority_class): su un train set già bilanciato
        #     class_weight='balanced' non avrebbe impatto concreto
        #     (pesi quasi identici a None), quindi non è una dimensione di
        #     ricerca utile qui -- a differenza del dataset sintetico di
        #     classificazione (FASE 4, eredita gli iperparametri dal
        #     reale), che resta sbilanciato e per cui class_weight sarebbe
        #     rilevante se si facesse un tuning dedicato.
        #   - bootstrap [True]: vincolo definitorio, la stima OOB richiede
        #     bootstrap=True (Breiman, Definition 1.1).
        #   - max_samples, esteso a 0.3: aggiunto apposta come leva diretta
        #     per abbassare rho_bar (correlazione tra alberi, vedi
        #     analyze_tree_correlation.py), non un valore a caso.
        #
        #   max_depth [10,25,None] e min_samples_split [2,5,10] -- VERIFICATO
        #   sulla User Guide ufficiale scikit-learn (sezione Ensemble):
        #   "Good results are often achieved when setting max_depth=None in
        #   combination with min_samples_split=2 (i.e., when fully
        #   developing the trees)" -- è anche la combinazione di default di
        #   RandomForestClassifier. La griglia INCLUDE deliberatamente
        #   questo default raccomandato (None, 2) come caso di riferimento,
        #   e lo affianca a valori più regolarizzati (10/25 per max_depth,
        #   5/10 per min_samples_split) per verificare EMPIRICAMENTE se
        #   allontanarsi dal default aiuta su un dataset con multicollinearità
        #   nota (rho_bar elevato, Sezione baseline-tuning) -- "often" nella
        #   citazione non è "always": non è garantito che il default generico
        #   sia ottimale anche per questo dataset specifico, da qui la
        #   verifica invece di accettarlo per assunzione.
        #
        #   n_estimators [10..200]: limite superiore allineato alla curva
        #   fine-grained di analyze_classification_n_estimators.py (prima
        #   fermo a 100, incoerenza corretta) -- griglia esplorativa a
        #   copertura crescente, non derivata da un calcolo specifico.
        # ---------------------------------------------------------------
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [10, 20, 30, 40, 60, 80, 100, 150, 200]),
            "max_depth": trial.suggest_categorical("max_depth", [None]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [2]),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            # class_weight NON esplorato: il train set arriva già bilanciato
            # 1:1 dall'undersampling (undersample_majority_class, eseguito
            # a monte in run_baseline()). Su un train set già bilanciato
            # class_weight='balanced' produce pesi quasi identici a None
            # (le due classi hanno frequenza quasi uguale), quindi era una
            # dimensione di ricerca che non poteva mai avere un impatto
            # concreto sui risultati -- rimossa per non allargare
            # inutilmente lo spazio degli iperparametri (60 trial esplorati
            # su una griglia più piccola e più significativa).
            "class_weight": None,
            "bootstrap": True,
            "max_samples": trial.suggest_categorical("max_samples", [0.3, 0.5, 0.7, 0.8, 1.0]),
        }

        fs, train_selected = get_or_compute_feature_selection(params["max_features"])
        X_trial = train_selected.drop(columns=[target_col])
        y_trial = train_selected[target_col]

        print(f"[Optuna OOB] trial {trial.number + 1}/{n_trials} in corso: {params} "
              f"({X_trial.shape[1]} feature) ...", flush=True)

        rf = RandomForestClassifier(**params, random_state=random_state,
                                     oob_score=True, n_jobs=n_jobs)
        start = time.perf_counter()
        rf.fit(X_trial, y_trial)
        fit_time = time.perf_counter() - start

        metrics = _oob_classification_metrics(rf, y_trial)
        print(f"[Optuna OOB] trial {trial.number + 1}/{n_trials} completato in {fit_time:.1f}s — "
              f"OOB F1={metrics['oob_f1']:.4f}, OOB Acc={metrics['oob_accuracy']:.4f}", flush=True)
        if metrics["n_missing_oob"] > 0:
            pct = metrics["n_missing_oob"] / len(y_trial) * 100
            print(f"   [OOB] {metrics['n_missing_oob']} righe ({pct:.2f}%) senza copertura OOB, escluse.")

        results.append({"params": params, "fit_time": fit_time, "n_features": X_trial.shape[1],
                         **metrics,
                         "oob_accuracy": metrics["oob_accuracy"], "oob_precision": metrics["oob_precision"],
                         "oob_recall": metrics["oob_recall"], "oob_f1": metrics["oob_f1"]})

        return metrics[f"oob_{refit_metric}"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    # Verifica EMPIRICA (non solo assunzione teorica) se max_features ha
    # davvero cambiato il set di feature selezionato, invece di darlo per
    # scontato -- riportabile in relazione in entrambi i casi (conferma o
    # smentisce l'ipotesi che max_features influenzi la selezione).
    if len(feature_selection_cache) > 1:
        unique_sets = {
            tuple(sorted(c.drop(columns=[target_col]).columns))
            for _, c in feature_selection_cache.values()
        }
        print(f"\n[Feature Selection] Verifica: {len(feature_selection_cache)} valori di "
              f"max_features esplorati, {len(unique_sets)} set di feature DISTINTI prodotti.")
        if len(unique_sets) == 1:
            print("  [NOTA] Stesso identico set di feature per tutti i valori di max_features: "
                  "in questo caso specifico max_features non ha inciso sulla selezione -- "
                  "verifica empirica, riportabile in relazione.")
        else:
            print("  [NOTA] Set di feature diversi tra i valori di max_features esplorati: "
                  "la selezione È risultata sensibile al valore usato per la foresta "
                  "preliminare, a conferma del meccanismo ipotizzato.")

    results.sort(key=lambda r: r[f"oob_{refit_metric}"], reverse=True)
    best_params = dict(results[0]["params"])
    best_fs, best_train_selected = feature_selection_cache[best_params["max_features"]]
    return best_params, results, best_fs, best_train_selected

# ---------------------------------------------------------------------------
# n_estimators NON viene più raffinato né scelto interattivamente qui.
#
# PRIMA: questo file conteneva una seconda ricerca Optuna
# (optuna_refine_n_estimators, senza warm_start) seguita da una scelta
# interattiva a runtime (scegli_n_estimators, con tolleranza di F1) che
# poteva RISCRIVERE l'n_estimators trovato da optuna_oob_hyperparameter_search
# -- tre punti diversi, con tre metodi diversi (Optuna categorico, Optuna
# GridSampler senza warm_start, soglia di tolleranza manuale) che decidevano
# lo stesso numero.
#
# ORA: n_estimators è semplicemente quello scelto da
# optuna_oob_hyperparameter_search insieme a tutti gli altri iperparametri
# (stessa combinazione vincente, stessa metrica OOB, nessun ripensamento
# successivo). La giustificazione empirica del valore -- la curva OOB con
# warm_start, fedele all'esempio ufficiale scikit-learn "OOB Errors for
# Random Forests" -- è stata spostata per intero in un modulo indipendente,
# 'analyze_n_estimators.py', che copre sia il task di classificazione
# (dataset reale) sia quello di regressione (dataset sintetico): va eseguito
# SEPARATAMENTE, a valle di un run di questo script, per produrre il
# grafico/tabella da riportare in relazione. Non scrive né modifica alcun
# manifesto: è puramente diagnostico, mai in the hot path della baseline.
# ---------------------------------------------------------------------------


def run_baseline():
    # --- CONFIGURAZIONE STILISTICA REPORT ---
    LUNGHEZZA_LINEA = 80
    DOPPIA_LINEA = "═" * LUNGHEZZA_LINEA
    LINEA_SINGOLA = "─" * LUNGHEZZA_LINEA

    print("=====================================================")
    print("   AVVIO FASE TUNING & BASELINE SPECULARE AL CLUSTER ")
    print("=====================================================\n")

    # Variabili di configurazione
    RANDOM_SEED = 123
    TEST_SIZE = 0.2
    # Campionamento RIBILANCIATO per giorno di cattura, al posto di una
    # sample_fraction uniforme (che favorirebbe sistematicamente i giorni più
    # grandi — es. Thuesday-20-02-2018, da solo quasi la metà del dataset
    # totale). Vincolo: deve restare sotto il file più piccolo tra i 10
    # (Thursday-01-03-2018, 331.125 righe), altrimenti quel giorno smette di
    # contribuire alla pari con gli altri. 100.000 è stato scelto per
    # restare con margine sotto quella soglia, per aumentare la varietà del
    # train set e verificare se questo sposta in avanti il plateau di
    # n_estimators osservato nella diagnostica OOB (vedi
    # analyze_classification_n_estimators.py). Nessuna sample_fraction
    # uniforme definita qui: RawCSVDataLoader la richiede solo se
    # target_rows_per_day è None (non il caso qui), e mantenerla come
    # variabile morta avrebbe solo suggerito un ruolo che non ha.
    TARGET_ROWS_PER_DAY = 100_000
    # Feature selection: OOB permutation importance, fedele a Breiman (2001)
    # Sec. 10 — criterio primario, vedi CICIDSFeatureSelector.
    IMPORTANCE_THRESHOLD = 0.0
    # Secondo criterio (dopo quello sopra): riduzione della multicollinearità
    # via clustering gerarchico — vedi reduce_multicollinearity() in
    # CICIDSFeatureSelector per la motivazione completa e il significato
    # della soglia (distanza 1-|correlazione di Spearman|; 0.2 ~ raggruppa
    # feature con |correlazione| >= 0.8 -- soglia più conservativa del
    # default 0.3 usato nei run precedenti, quindi cluster più piccoli e
    # più feature superstiti a parità di dataset).
    MULTICOLLINEARITY_DISTANCE_THRESHOLD = 0.2
    # Feature note in letteratura sul CICIDS2018 come possibili "socket
    # surrogate"/artefatti della configurazione fissa degli script di
    # attacco (es. gli attack tool usati per generare il dataset spesso
    # forzano un protocollo/pattern fisso, indipendente dalla vera natura
    # dell'attacco) -- non rimosse a priori (nessuna prova diretta che SIA
    # leakage per questo dataset specifico), ma segnalate esplicitamente
    # in relazione se sopravvivono alla feature selection, così la loro
    # presenza/assenza tra le feature vincenti è sempre visibile senza
    # dover aprire manualmente 'config_real.json' -> 'feature_selezionate'.
    SUSPECT_LEAKAGE_FEATURES = ["Protocol"]
    # Quante feature (in ordine di importanza OOB decrescente) stampare nel
    # report -- puramente diagnostico/descrittivo, non usato per decidere
    # nulla nella pipeline (la feature selection ha già fatto la sua scelta
    # separatamente, vedi CICIDSFeatureSelector).
    TOP_N_FEATURE_IMPORTANCE = 10
    # Under-sampling della classe maggioritaria (Benign), solo sul train set
    # -- vedi undersampling.py per la motivazione completa. ratio=1.0
    # produce un bilanciamento 1:1 (maggioritaria ridotta alla stessa
    # dimensione della minoritaria), lo standard di riferimento più comune
    # in letteratura come punto di partenza prima di esplorare rapporti
    # diversi.
    UNDERSAMPLING_RATIO = 1.0
    # Validation set per la calibrazione della soglia di decisione finale
    # (vedi FASE 4): ritagliato dal train PRIMA dell'undersampling, quindi
    # con la vera distribuzione sbilanciata (stessa proporzione Benign/
    # Attacco del test set) -- MAI toccato da undersampling/training.
    # Nasce per correggere un problema concreto osservato: scegliere la
    # soglia via ROC/Youden sulle probabilità OOB di un modello addestrato
    # su train RIBILANCIATO eredita comunque la distorsione del
    # ribilanciamento (le probabilità OOB sono anch'esse calcolate sullo
    # stesso train ribilanciato). Un validation set separato, mai
    # ribilanciato, non ha questo problema -- e resta comunque distinto dal
    # test set, quindi nessun leakage nella valutazione finale.
    VALIDATION_SIZE_FOR_THRESHOLD = 0.15
    # Soglia alternativa scelta con un vincolo di business invece che con
    # F1-max: "massimizza la recall SUBORDINATAMENTE a un FPR sul
    # validation set <= TARGET_FPR_CONSTRAINT" (criterio di Neyman-Pearson,
    # standard nei sistemi IDS reali dove un FPR alto satura il SOC di
    # falsi allarmi -- vedi discussione). Calcolato in aggiunta alla soglia
    # F1-max, non al suo posto: il report mostra entrambi gli operating
    # point, così la scelta finale resta esplicita e motivata, non nascosta
    # in una costante.
    TARGET_FPR_CONSTRAINT = 0.01

    target_col = "Label"
    BOOT_CONFIG_PATH = os.path.join("./.local_storage", "config.json")
    OUTPUT_DIR = "./outputs_baseline"
    REAL_CONFIG_PATH = os.path.join(OUTPUT_DIR, "config_real.json")
    SYNTHETIC_CONFIG_PATH = os.path.join(OUTPUT_DIR, "config_synthetic.json")
    # NESSUN tuning per il regressore (vedi REGRESSOR_DEFAULT_HP più sotto):
    # questo path serve solo a rileggere n_estimators da un run precedente
    # (l'unico iperparametro mai lasciato a un default, va sempre misurato
    # con analyze_n_estimators.py) dalla copia per-task già scritta per OGNI
    # run sintetico -- nessun file aggiuntivo introdotto.
    REGRESSOR_TUNING_CONFIG_PATH = os.path.join(OUTPUT_DIR, "config_synthetic_regressor.json")
    dataset_type = "real"
    user_tree_type = "classifier"

    sys_cfg = SystemConfig()
    print(f" • Ambiente infrastrutturale rilevato: {sys_cfg.env.upper()}")

    # ---------------------------------------------------------
    # FASE 1: ETL CON CAMPIONAMENTO PROBABILISTICO 
    # ---------------------------------------------------------
    print(">>> FASE 1: Estrazione e Preprocessing Dati")
    
    if os.path.exists(BOOT_CONFIG_PATH):
        with open(BOOT_CONFIG_PATH, "r") as f:
            try:
                raw_state = json.load(f)
                if not isinstance(raw_state, dict):
                    raise ValueError("Il contenuto del file di stato locale non è un oggetto JSON valido.")

                if "baseline_boot" in raw_state:
                    # Nuovo formato strutturato: {"baseline_boot": {...}, "last_training_request": {...}}
                    boot_cfg = raw_state["baseline_boot"]
                elif "dataset_type" in raw_state and "hyperparameters" not in raw_state:
                    # Retrocompatibilità: vecchio formato piatto scritto direttamente come boot config.
                    boot_cfg = raw_state
                    print(f" [INFO] '{BOOT_CONFIG_PATH}' è nel formato precedente (piatto). Letto comunque per retrocompatibilità.")
                else:
                    # Il file esiste ma contiene solo (o principalmente) una last_training_request:
                    # non c'è una boot config valida, si scala sui default.
                    boot_cfg = {}
                    print(f" [INFO] '{BOOT_CONFIG_PATH}' non contiene una sezione 'baseline_boot' valida. Uso i default.")

                dataset_type = boot_cfg.get("dataset_type", "real")
                user_tree_type = boot_cfg.get("tree_type", "classifier")
                if boot_cfg:
                    print(f" [INFO] Configurazione di boot letta con successo da '{BOOT_CONFIG_PATH}'")
            except Exception as e:
                print(f" [ATTENZIONE] Errore nel parsing di {BOOT_CONFIG_PATH}: {e}")
                pass
    else:
        print(f" [INFO] Nessun file di boot trovato in '{BOOT_CONFIG_PATH}'. Scalo sul dataset reale di default.")

    # ---------------------------------------------------------
    # SCELTA 1: eseguire il tuning OOB (FASE 2, ~50 fit, costoso) o riusare
    # gli iperparametri già presenti in un config_real.json esistente
    # (salta la ricerca, va dritto alla FASE 4 con quegli iperparametri).
    # Utile per iterare rapidamente su altre parti della pipeline (es.
    # preprocessing, feature selection, dimensione campione) senza dover
    # rifare mezz'ora di ricerca ogni volta che non serve.
    #
    # SOLO PER IL DATASET REALE: è l'unico caso in cui esiste ancora un
    # tuning da poter saltare. Il classificatore sintetico eredita sempre
    # dal reale (nessuna scelta da fare qui), il regressore sintetico non
    # tuna più nulla (vedi REGRESSOR_DEFAULT_HP) -- chiedere questo prompt
    # anche per quei due casi sarebbe fuorviante (nessuna delle due opzioni
    # avrebbe effetto).
    # ---------------------------------------------------------
    if dataset_type == "real":
        print("\nEseguire il tuning iperparametrico (FASE 2) o riusare gli iperparametri "
              "già presenti in un manifesto esistente?")
        print("  [1] Esegui il tuning (default) ")
        print("  [2] Salta il tuning — riusa gli iperparametri già presenti in "
              "'outputs_baseline/config_real.json'")
        scelta_tuning = input("  Scelta [Default: 1]: ").strip() or "1"
        SKIP_TUNING = (scelta_tuning == "2")
    else:
        SKIP_TUNING = False  # non usata nei rami sintetici, valore neutro

    # Default: nessun validation set (usato solo dal ramo dataset reale, per
    # la calibrazione della soglia -- vedi VALIDATION_SIZE_FOR_THRESHOLD).
    validation_df = None

    if dataset_type == "synthetic":
        print(">>> FASE 1: Generazione e Preprocessing Dataset Sintetico")
        best_hp_reale = None
        if user_tree_type == "classifier":
            if not os.path.exists(REAL_CONFIG_PATH):
                raise FileNotFoundError(
                    f"Per eseguire la baseline sintetica di CLASSIFICAZIONE serve aver già "
                    f"eseguito il tuning sul dataset reale: '{REAL_CONFIG_PATH}' non trovato. "
                    f"Eseguire prima la baseline sul dataset reale (opzione 1)."
                )
            with open(REAL_CONFIG_PATH, "r") as f:
                real_config = json.load(f)
            best_hp_reale = real_config["hyperparameters"]
            print(f" [INFO] Iperparametri di riferimento caricati dal tuning sul reale: '{REAL_CONFIG_PATH}'")
        else:
            print(" [INFO] Task REGRESSOR: nessun tuning (né sul reale né sul sintetico stesso) "
                  "-- iperparametri di default sklearn, vedi REGRESSOR_DEFAULT_HP.")

        if os.path.exists(SYNTHETIC_CONFIG_PATH):
            with open(SYNTHETIC_CONFIG_PATH, "r") as f:
                try:
                    tmp_cfg = json.load(f)
                    print(f" [INFO] Configurazione sintetica letta con successo da '{SYNTHETIC_CONFIG_PATH}'")
                except Exception as e:
                    tmp_cfg = {}
                    print(f" [ATTENZIONE] Errore nel parsing di {SYNTHETIC_CONFIG_PATH}: {e}")
        else:
            tmp_cfg = {}
        
        # Stessa convenzione di CentralizedOrchestrator._prepare_data
        # ("Target" per la regressione, "Label" per la classificazione). Prima
        # era fissa a "Target" anche in classificazione, quindi baseline e
        # cluster producevano CSV con la colonna target chiamata diversamente.
        target_col = "Target" if user_tree_type == "regressor" else "Label"

        # Default DIVERSI per task: la regressione (Friedman #1) usa una
        # ricetta diversa dalla classificazione, non ha senso condividere gli
        # stessi fallback. Se tmp_cfg contiene già un valore (run precedente
        # dello STESSO task), quello ha sempre precedenza.
        if user_tree_type == "regressor":
            # n_samples >= 1.000.000: dimensione scelta per dare al cluster
            # abbastanza lavoro da rendere interessante il confronto di
            # scalabilità con la baseline single-node (unico scopo di questo
            # dataset -- non serve massimizzare l'accuratezza del modello).
            n_samples = tmp_cfg.get("n_samples", 1_000_000)
            # n_features=50 con Friedman #1: 5 informative + 5 redundant + 40 noise, per un SNR ≈ 10:1
            n_features = tmp_cfg.get("n_features", 50)
        else:
            n_features = tmp_cfg.get("n_features", 30)
            # Default allineato a SyntheticDataLoader (n_samples=300000).
            # Prima qui il default era 500000: in assenza di un manifesto la
            # baseline generava un dataset 1.67x più grande di quello del
            # cluster, e i tempi di addestramento non erano confrontabili.
            n_samples = tmp_cfg.get("n_samples", 300000)
        # noise=0.5: calibrato sulla deviazione standard EMPIRICA del target
        # "pulito" di Friedman #1 (misurata: std≈4.87, costante al variare di
        # n_features perché dipende solo dalle 5 feature informative fisse),
        # per un SNR ≈ 10:1 (rumore ≈ 10% della variabilità naturale del
        # target) -- livello moderato, coerente con la pratica comune per
        # dataset sintetici "non banali ma non dominati dal rumore".
        noise = tmp_cfg.get("noise", 2.5)
        n_informative = tmp_cfg.get("n_informative", int(n_features * 0.35))
        n_redundant = tmp_cfg.get("n_redundant", 5)
        n_clusters_per_class = tmp_cfg.get("n_clusters_per_class", 2)
        flip_y = tmp_cfg.get("flip_y", 0.01)
        weight = tmp_cfg.get("weight", [0.9, 0.1])

        # Registriamo i parametri EFFETTIVAMENTE usati per generare il dataset,
        # non quelli riletti dal file di input. Prima il dizionario veniva
        # costruito filtrando 'tmp_cfg': al primo run il manifesto non esiste,
        # tmp_cfg è vuoto e quindi n_samples/n_features non venivano MAI
        # persistiti. Il manifesto restava privo della ricetta, SyntheticDataLoader
        # continuava a usare i propri default e le due parti non convergevano
        # mai su una configurazione comune.
        dataset_gen_params = {
            "n_samples": n_samples,
            "n_features": n_features,
            "target_column": target_col,
            "random_seed": RANDOM_SEED,
        }
        if user_tree_type == "regressor":
            dataset_gen_params["noise"] = noise
        else:
            dataset_gen_params["n_informative"] = n_informative
            dataset_gen_params["n_redundant"] = n_redundant
            dataset_gen_params["n_clusters_per_class"] = n_clusters_per_class
            dataset_gen_params["flip_y"] = flip_y
            dataset_gen_params["weight"] = weight

        task_str = "regression" if user_tree_type == "regressor" else "classification"
        print(f" • Tipo Dataset: Sintetico (Stress Test Task - {user_tree_type.upper()})")

        loader_kwargs = dict(
            task=task_str,
            n_samples=n_samples,
            n_features=n_features,
            random_seed=RANDOM_SEED,
            target_column=target_col,
        )
        if user_tree_type == "regressor":
            loader_kwargs["noise"] = noise
        else:
            loader_kwargs["n_informative"] = n_informative
            loader_kwargs["n_redundant"] = n_redundant
            loader_kwargs["n_clusters_per_class"] = n_clusters_per_class
            loader_kwargs["flip_y"] = flip_y
            loader_kwargs["weight"] = weight

        loader = SyntheticDataLoader(**loader_kwargs)
        # Generazione e split cronometrati come I/O + ETL. Prima erano azzerati
        # (io_time = etl_time = 0.0): la baseline sintetica dichiarava zero costo
        # di preparazione dati mentre il lato distribuito conteggia l'intero
        # _prepare_data (30-40s su AWS per via di S3), quindi un confronto sui
        # tempi TOTALI risultava sistematicamente sfavorevole al cluster.
        io_start_time = time.perf_counter()
        df_clean = loader.load()
        io_time = time.perf_counter() - io_start_time
        print(f"[OK] Generazione dataset sintetico completata in {io_time:.4f} secondi.")

        preprocess_start_time = time.perf_counter()
        if user_tree_type == "regressor":
            train_df, test_df = train_test_split(df_clean, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        else:
            # Split STRATIFICATO come in _prepare_data: sul sintetico di
            # classificazione le classi sono sbilanciate (weights [0.9, 0.1]),
            # quindi uno split non stratificato darebbe alla baseline una
            # ripartizione train/test diversa da quella vista dal cluster.
            train_df, test_df = StratifiedDataSplitter(
                target_column=target_col, test_size=TEST_SIZE, random_state=RANDOM_SEED
            ).split(df_clean)
        etl_time = time.perf_counter() - preprocess_start_time
    else:
        data_folder = os.environ.get("DATASET_LOCAL_PATH", "./dataset_cache")
        if not os.path.exists(data_folder):
            raise FileNotFoundError(
                f"Cartella del dataset reale non trovata: '{data_folder}'. "
                f"Posiziona lì i CSV del CICIDS, oppure indica un percorso diverso "
                f"con la variabile d'ambiente DATASET_LOCAL_PATH. "
                f"(Nessun fallback automatico: leggere CSV da una cartella non "
                f"prevista produrrebbe una baseline sbagliata senza segnalarlo.)"
            )

        print(f" • Cartella sorgente identificata per dati reali: '{data_folder}'")
        print(f" • Tipo Dataset: Reale (Campionamento RIBILANCIATO per giorno, "
              f"target ~{TARGET_ROWS_PER_DAY} righe/giorno, Seed: {RANDOM_SEED})")

        io_start_time = time.perf_counter()
        loader = RawCSVDataLoader(
            data_url=data_folder,
            # sample_fraction non passata: usa il default 1.0 di
            # RawCSVDataLoader, comunque ignorato perché target_rows_per_day
            # non è None (ha sempre priorità -- vedi RawCSVDataLoader.load()).
            dataset_seed=RANDOM_SEED,
            target_rows_per_day=TARGET_ROWS_PER_DAY,
        )
        df_raw = loader.load()
        io_time = time.perf_counter() - io_start_time
        print(f"[OK] Caricamento dati (I/O) completato in {io_time:.4f} secondi.")

        preprocess_start_time = time.perf_counter()
        preprocessor = CICIDSPreprocessor(target_column=target_col)
        splitter = StratifiedDataSplitter(target_column=target_col, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
        print(" • Binarizzazione sul dato intero...")
        df_binarized = preprocessor.binarize_target(df_raw)

        print(" • Esecuzione Split Stratificato...")
        train_df, test_df = splitter.split(df_binarized)

        print(" • Preprocessing indipendente sul Train Set...")
        train_df = preprocessor.process(train_df)
        
        print(" • Preprocessing indipendente sul Test Set...")
        test_df = preprocessor.process(test_df)

        # Validation set per la calibrazione della soglia di decisione
        # (vedi FASE 4 e VALIDATION_SIZE_FOR_THRESHOLD): ritagliato QUI,
        # PRIMA dell'undersampling, così mantiene la vera distribuzione
        # sbilanciata (stessa proporzione Benign/Attacco del test set) --
        # non toccato da undersampling né usato per il training.
        print(f" • Split di un validation set ({VALIDATION_SIZE_FOR_THRESHOLD*100:.0f}% del train, "
              f"distribuzione originale, per la calibrazione della soglia)...")
        validation_splitter = StratifiedDataSplitter(
            target_column=target_col, test_size=VALIDATION_SIZE_FOR_THRESHOLD, random_state=RANDOM_SEED
        )
        train_df, validation_df = validation_splitter.split(train_df)

        print(" • Under-sampling della classe maggioritaria (solo train set)...")
        train_df = undersample_majority_class(
            train_df, target_column=target_col,
            majority_class=0, minority_class=1,
            ratio=UNDERSAMPLING_RATIO, random_state=RANDOM_SEED,
        )
        # La feature selection NON viene più applicata qui, con una
        # configurazione fissa scollegata dal tuning: è stata spostata
        # dentro optuna_oob_hyperparameter_search, dove si muove insieme
        # alla ricerca degli iperparametri (vedi il docstring di quella
        # funzione per la motivazione completa, allineata al pattern
        # ufficiale scikit-learn SelectFromModel-dentro-Pipeline).
        # train_df/test_df restano qui il dataset COMPLETO (69 feature),
        # non ancora ridotto -- la riduzione avviene più avanti, per il
        # valore di max_features vincente del tuning.
        etl_time = time.perf_counter() - preprocess_start_time

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    # Default: nessun validation set per la calibrazione della soglia
    # (dataset sintetico, o regressore -- non applicabile). Sovrascritto più
    # avanti SOLO nel ramo dataset reale, dopo la feature selection.
    X_val, y_val = None, None
    # Default: nessun punteggio di importanza (permutation OOB) -- la
    # feature selection sul sintetico non lo calcola (segnale/rumore già
    # noto a priori dal generatore, vedi SyntheticDataLoader). Sovrascritto
    # più avanti SOLO nel ramo dataset reale.
    feature_importance_scores = None

    if dataset_type == "synthetic":
        dizionario_feature = {"eliminate": [], "salvate": list(X_train.columns)}
        print(f" • Volume Train: {X_train.shape} | Volume Test: {X_test.shape}")
    else:
        print(f" • Volume Train (PRIMA della feature selection, 69 feature): {X_train.shape} | "
              f"Volume Test: {X_test.shape}")
    print("-" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config_path_final = os.path.join(OUTPUT_DIR, "config_real.json")
    pickle_path_final = os.path.join(
        OUTPUT_DIR,
        f"baseline_random_forest_{user_tree_type}.pkl" if dataset_type == "synthetic" else "baseline_random_forest_completa.pkl"
    )

    if dataset_type == "real":
    # ---------------------------------------------------------
    # FASE 2: TUNING BASATO SU STIMA OOB (Breiman 2001, Sec. 3.1)
    # ---------------------------------------------------------
        if SKIP_TUNING:
            print("\n>>> FASE 2: SALTATA su richiesta — riuso iperparametri da "
                  f"'{REAL_CONFIG_PATH}'...")
            if not os.path.exists(REAL_CONFIG_PATH):
                raise FileNotFoundError(
                    f"Hai scelto di saltare il tuning, ma '{REAL_CONFIG_PATH}' non esiste "
                    f"ancora: serve un manifesto precedente da cui leggere gli iperparametri. "
                    f"Esegui prima il tuning almeno una volta (Scelta 1)."
                )
            with open(REAL_CONFIG_PATH, "r") as f:
                existing_config = json.load(f)
            best_params = dict(existing_config["hyperparameters"])
            print(f"[OK] Iperparametri riusati da '{REAL_CONFIG_PATH}': {best_params}")
            oob_search_results = None
            tempo_tuning = 0.0
            tempo_medio_fit_tuning = 0.0

            # La feature selection non è più eseguita a monte del tuning (vedi
            # optuna_oob_hyperparameter_search per la motivazione): con il
            # tuning saltato, va comunque eseguita qui una volta, usando il
            # max_features già presente nel manifesto riusato, per restare
            # coerenti con l'iperparametro che quella configurazione dichiara.
            print(" • Applicazione Feature Selection (max_features dal manifesto riusato: "
                  f"'{best_params.get('max_features', 'sqrt')}')...")
            fs = CICIDSFeatureSelector(
                target_column=target_col,
                importance_threshold=IMPORTANCE_THRESHOLD,
                rf_random_state=RANDOM_SEED,
                rf_max_features=best_params.get("max_features", "sqrt"),
                reduce_multicollinearity=True,
                multicollinearity_distance_threshold=MULTICOLLINEARITY_DISTANCE_THRESHOLD,
                dendrogram_plot_path=(
                    f"feature_correlation_dendrogram_{best_params.get('max_features', 'sqrt')}"
                    f"_skiptuning.png"
                ),
            )
            train_df = fs.fit_transform(train_df)
            test_df = fs.transform(test_df)
            if validation_df is not None:
                validation_df = fs.transform(validation_df)
            dizionario_feature = fs.feature_summary_
            feature_importance_scores = fs.importance_scores_
        else:
            print("\n>>> FASE 2: Esplorazione Spazio Iperparametri (Tuning via stima OOB, "
                  "feature selection integrata per max_features — vedi docstring)...")
            N_ITER_TUNING = 60
            start_tuning = time.perf_counter()
            best_params, oob_search_results, best_fs, best_train_selected = optuna_oob_hyperparameter_search(
                train_df, target_col,
                n_trials=N_ITER_TUNING,
                random_state=RANDOM_SEED,
                importance_threshold=IMPORTANCE_THRESHOLD,
                multicollinearity_distance_threshold=MULTICOLLINEARITY_DISTANCE_THRESHOLD,
                refit_metric="f1",
            )
            tempo_tuning = time.perf_counter() - start_tuning
            print(f"[OK] Tuning completato ({len(oob_search_results)} combinazioni esplorate via OOB). "
                  f"Iperparametri ottimali: {best_params}")
            tempo_medio_fit_tuning = float(np.mean([r["fit_time"] for r in oob_search_results]))

            # Applica al train/test set il set di feature calcolato per il
            # max_features della combinazione vincente (dalla cache interna
            # a optuna_oob_hyperparameter_search).
            train_df = best_train_selected
            test_df = best_fs.transform(test_df)
            if validation_df is not None:
                validation_df = best_fs.transform(validation_df)
            dizionario_feature = best_fs.feature_summary_
            feature_importance_scores = best_fs.importance_scores_

        # Ricostruisce X_train/y_train/X_test/y_test (e X_val/y_val, se
        # presente) dal train/test/validation ORA ridotti alle feature
        # selezionate (nel ramo SKIP_TUNING appena calcolate sopra; nel ramo
        # di tuning, quelle del max_features vincente) -- necessario perché
        # la costruzione precedente usava ancora il dataset completo a 69
        # feature, dato che la feature selection è stata spostata dopo
        # quel punto.
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        if validation_df is not None:
            X_val = validation_df.drop(columns=[target_col])
            y_val = validation_df[target_col]
        else:
            X_val, y_val = None, None
        print(f" • Volume Train (DOPO la feature selection): {X_train.shape} | "
              f"Volume Test: {X_test.shape}")

        # n_estimators: nessun raffinamento/override qui -- resta esattamente
        # il valore scelto da optuna_oob_hyperparameter_search insieme agli
        # altri iperparametri (vedi commento sopra run_baseline()). La
        # giustificazione empirica (curva OOB warm_start) va prodotta A PARTE
        # con 'python -m src.baseline.analyze_n_estimators --task classifier'.
        motivazione_n_estimators = (
            f"n_estimators={best_params['n_estimators']} scelto dalla ricerca OOB congiunta "
            f"(optuna_oob_hyperparameter_search, stessa combinazione vincente di tutti gli "
            f"altri iperparametri, OOB F1 come refit_metric). Giustificazione empirica della "
            f"scelta -- curva OOB con warm_start -- prodotta separatamente da "
            f"analyze_n_estimators.py (non eseguita automaticamente qui)."
        )

        # Configurazione e creazione cartella di output dedicata
        OUTPUT_DIR = "./outputs_baseline"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        config_path_final = os.path.join(OUTPUT_DIR, "config_real.json")
        pickle_path_final = os.path.join(OUTPUT_DIR, "baseline_random_forest_completa.pkl")

        # ---------------------------------------------------------
        # FASE 3: SCRITTURA MANIFESTO CONFIG_REAL.JSON
        # ---------------------------------------------------------
        best_booststrap = bool(best_params.get("bootstrap", True))
        config_data = {
            "mode": "distributed",
            "dataset_type": dataset_type,
            "dataset_path": data_folder if dataset_type == "real" else "synthetic",
            "feature_selection_method": dizionario_feature.get("selection_method", "permutation_importance"),
            "importance_threshold": IMPORTANCE_THRESHOLD,
            "feature_eliminata" : dizionario_feature["eliminate"],
            "feature_selezionate" : dizionario_feature["salvate"],
            "n_estimators_note": motivazione_n_estimators,
            "hyperparameters": {
                "n_estimators": int(best_params.get("n_estimators", 10)),
                "max_depth": best_params.get("max_depth") ,
                "min_samples_split": int(best_params.get("min_samples_split", 2)),
                "max_features": best_params.get("max_features", "sqrt"),
                "criterion": best_params.get("criterion", "gini"),
                "class_weight": best_params.get("class_weight", None),
                "bootstrap": best_booststrap,
                "max_samples": float(best_params.get("max_samples", 1.0)) if best_booststrap else 1.0,
                "tree_type": user_tree_type,
                "target_column": target_col,
                "random_state": int(RANDOM_SEED)
            }
        }

        with open(config_path_final, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print(f"[OK] Manifesto 'config_real.json' salvato correttamente in: '{config_path_final}'")

    elif user_tree_type == "classifier":
        print("\n>>> FASE 2: SALTATA — riuso iperparametri ottenuti dal tuning sul dataset reale")
        tempo_tuning = 0.0
        tempo_medio_fit_tuning = 0.0
        oob_search_results = None

        # Stesso algoritmo/task del tuning reale (classificazione binaria):
        # ereditare gli iperparametri è metodologicamente corretto.
        best_bootstrap = bool(best_hp_reale.get("bootstrap", True))
        hp_sintetici = {
            "n_estimators": int(best_hp_reale.get("n_estimators", 10)),
            "max_depth": best_hp_reale.get("max_depth"),
            "min_samples_split": int(best_hp_reale.get("min_samples_split", 2)),
            "max_features": best_hp_reale.get("max_features", "sqrt"),
            "criterion": best_hp_reale.get("criterion", "gini"),
            # Anche il dataset sintetico di classificazione è sbilanciato
            # (weights [0.9, 0.1] in SyntheticDataLoader): class_weight va
            # ereditato esattamente come gli altri iperparametri, altrimenti
            # l'eredità dichiarata "corretta perché stesso task" sarebbe solo
            # parziale.
            "class_weight": best_hp_reale.get("class_weight", None),
            "bootstrap": best_bootstrap,
            "max_samples": float(best_hp_reale.get("max_samples", 1.0)) if best_bootstrap else 1.0,
        }

    else:
        # ---------------------------------------------------------------
        # Task REGRESSOR: NESSUN tuning. Iperparametri di default sklearn
        # (REGRESSOR_DEFAULT_HP, a inizio file) -- il dataset sintetico serve
        # solo da stress-test di scalabilità, non a massimizzare l'R².
        #
        # n_estimators è l'unica eccezione: non è mai un "default" da usare
        # alla cieca, va sempre determinato via curva OOB warm_start (Breiman
        # 2001, Sec. 3.1) -- MA questo script non la calcola: la curva va
        # prodotta a parte con 'analyze_n_estimators.py --task regressor'
        # (lettura visiva del grafico, nessun knee-detection automatico --
        # vedi quel file per il motivo). Qui ci si limita a offrire la
        # possibilità di inserire il valore così deciso, invece di dover
        # editare a mano il manifesto.
        # ---------------------------------------------------------------
        print("\n>>> FASE 2 (regressore sintetico): NESSUN TUNING — iperparametri di "
              "default sklearn (vedi REGRESSOR_DEFAULT_HP)...")
        if os.path.exists(REGRESSOR_TUNING_CONFIG_PATH):
            with open(REGRESSOR_TUNING_CONFIG_PATH, "r") as f:
                existing_reg_config = json.load(f)
            n_estimators_default = int(existing_reg_config["hyperparameters"].get("n_estimators", 100))
            fonte_default = f"copia per-task precedente ('{REGRESSOR_TUNING_CONFIG_PATH}')"
        else:
            n_estimators_default = 100  # default sklearn stesso, solo per il primo bootstrap
            fonte_default = "default sklearn (nessun run precedente)"

        print(f"\n  n_estimators disponibile: {n_estimators_default} (fonte: {fonte_default}).")
        print("  [1] Usa questo valore (default)")
        print("  [2] Inserisci un valore custom (es. deciso guardando la curva OOB prodotta a "
              "parte con 'python -m src.baseline.analyze_n_estimators --task regressor')")
        scelta_n_est = input("  Scelta [Default: 1]: ").strip() or "1"

        if scelta_n_est == "2":
            scelta_valore = input("  n_estimators: ").strip()
            if scelta_valore:
                n_estimators_reg = int(scelta_valore)
                fonte_n_estimators = "valore custom inserito manualmente"
            else:
                n_estimators_reg = n_estimators_default
                fonte_n_estimators = fonte_default
        else:
            n_estimators_reg = n_estimators_default
            fonte_n_estimators = fonte_default

        print(f"[OK] n_estimators={n_estimators_reg} (fonte: {fonte_n_estimators}).")

        hp_sintetici = {**REGRESSOR_DEFAULT_HP, "n_estimators": n_estimators_reg}
        tempo_tuning = 0.0
        tempo_medio_fit_tuning = 0.0
        oob_search_results = None

        config_data = {
            "mode": "distributed",
            "dataset_type": "synthetic",
            "dataset_path": "synthetic",
            **dataset_gen_params,
            "feature_eliminata": dizionario_feature["eliminate"],
            "feature_selezionate": dizionario_feature["salvate"],
            "hyperparameters": {
                **hp_sintetici,
                "tree_type": user_tree_type,
                "target_column": target_col,
                "random_state": int(RANDOM_SEED)
            }
        }

        # 'config_synthetic.json' è il manifesto ATTIVO: è quello letto sia da
        # SyntheticDataLoader (ricetta del dataset) sia dal client
        # (load_hyperparameters_from_config). Essendo unico, un run di
        # classificazione sovrascrive quello di regressione e viceversa — ma la
        # traccia richiede ENTRAMBI i task. Ne salviamo quindi anche una copia
        # per-task, così i due esperimenti restano documentati e ricostruibili
        # (il .pkl del modello era già differenziato per tree_type, il manifesto no).
        config_path_synthetic = os.path.join(OUTPUT_DIR, "config_synthetic.json")
        with open(config_path_synthetic, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print(f"[OK] Manifesto sintetico ATTIVO ({user_tree_type}) salvato in: '{config_path_synthetic}'")

        config_path_synthetic_task = os.path.join(OUTPUT_DIR, f"config_synthetic_{user_tree_type}.json")
        with open(config_path_synthetic_task, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print(f"[OK] Copia per-task archiviata in: '{config_path_synthetic_task}'")
        print(f"[NOTA] Il cluster legge SEMPRE '{config_path_synthetic}': per passare all'altro "
              f"task sintetico va rilanciata la baseline, non basta avere la copia per-task.")

    # ---------------------------------------------------------
    # FASE 4: ADDESTRAMENTO FINALE & INFERENZA LOCALE
    # ---------------------------------------------------------
    print("\n>>> FASE 4: Addestramento Finale Monolitico per estrazione T_seq...")
    hp = config_data["hyperparameters"]
    rf_kwargs = dict(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        min_samples_split=hp["min_samples_split"],
        max_features=hp.get("max_features", "sqrt" if user_tree_type == "classifier" else 1 / 3),
        bootstrap=hp["bootstrap"],
        n_jobs=1,
        random_state=RANDOM_SEED
    )
    
    if hp["bootstrap"]:
        rf_kwargs["max_samples"] = hp["max_samples"]

    if hp.get("criterion"):
        rf_kwargs["criterion"] = hp["criterion"]

    if user_tree_type == "classifier":
        rf_kwargs["class_weight"] = hp.get("class_weight")
        tree_clf = RandomForestClassifier(**rf_kwargs)
    else:
        tree_clf = RandomForestRegressor(**rf_kwargs)
    
    start_train_finale = time.perf_counter()
    tree_clf.fit(X_train, y_train)
    t_seq = time.perf_counter() - start_train_finale
    print(f"[OK] Fitting completato. T_seq ottenuto: {t_seq:.4f} secondi.")

    # ---------------------------------------------------------
    # SECONDA BASELINE: stessa macchina, TUTTI i core (n_jobs=-1)
    #
    # Perché serve: T_seq è monocore (n_jobs=1), mentre ogni worker del cluster
    # addestra i propri alberi con un Pool multiprocesso (BaseWorker:
    # allocated_cores = cpu_count-1 su AWS). Se in relazione si presenta
    # T_seq / T_distribuito come "speedup della distribuzione", quel numero
    # include anche il guadagno del semplice multicore locale, che con
    # l'architettura distribuita non c'entra nulla.
    #
    # Con entrambi i riferimenti l'analisi diventa onesta e più ricca:
    #  • T_seq          -> speedup ed efficienza confrontabili con la teoria;
    #  • T_1node_par    -> "quanto guadagno DAVVERO distribuendo, rispetto a
    #                       usare al meglio una macchina sola?"
    # ---------------------------------------------------------
    cpu_disponibili = os.cpu_count() or 1
    print(f"\n>>> FASE 4b: Baseline su singola macchina MULTICORE ({cpu_disponibili} core logici, n_jobs=-1)...")
    rf_kwargs_par = dict(rf_kwargs)
    rf_kwargs_par["n_jobs"] = -1
    if user_tree_type == "classifier":
        tree_clf_par = RandomForestClassifier(**rf_kwargs_par)
    else:
        tree_clf_par = RandomForestRegressor(**rf_kwargs_par)

    start_train_par = time.perf_counter()
    tree_clf_par.fit(X_train, y_train)
    t_1node_parallel = time.perf_counter() - start_train_par
    speedup_multicore = (t_seq / t_1node_parallel) if t_1node_parallel > 0 else 1.0
    print(f"[OK] Fitting multicore completato: {t_1node_parallel:.4f} s "
          f"(speedup del solo multicore locale: {speedup_multicore:.2f}x)")
    # Il modello usato per le metriche resta quello sequenziale: n_jobs cambia
    # solo COME viene calcolato il fit, non il risultato (stesso random_state),
    # quindi tree_clf_par serve unicamente da riferimento temporale.

    print("\n[LOCAL] Calcolo delle predizioni e latenza sul Test Set indipendente...")
    start_inferenza = time.perf_counter()

    if user_tree_type == "classifier":
        local_proba = tree_clf.predict_proba(X_test)[:, 1]
        local_preds_default = tree_clf.predict(X_test)  # soglia implicita 0.50, tenuta per confronto

        if X_val is not None:
            # --- Soglia di decisione scelta sul VALIDATION SET, non sul
            # test set --- selezionare la soglia guardando il test set
            # sarebbe leakage (si taratura un iperparametro, la soglia,
            # sullo stesso set su cui si riportano poi le metriche finali).
            # Il validation set (vedi VALIDATION_SIZE_FOR_THRESHOLD) è
            # ritagliato dal train PRIMA dell'undersampling: mantiene la
            # vera distribuzione sbilanciata, quindi la soglia scelta qui
            # non eredita la distorsione di un train ribilanciato (a
            # differenza di una soglia scelta sulle probabilità OOB di un
            # modello addestrato su train ribilanciato, che sconterebbe la
            # stessa distorsione).
            val_proba = tree_clf.predict_proba(X_val)[:, 1]
            precisions, recalls, pr_thresholds = precision_recall_curve(y_val, val_proba)
            # precision_recall_curve restituisce un punto finale (recall=0,
            # precision=1) senza soglia corrispondente: scartato dal calcolo
            # dell'F1 qui sotto (precisions[:-1]/recalls[:-1]), altrimenti
            # pr_thresholds e i due array precision/recall avrebbero
            # lunghezze diverse.
            f1_scores = np.where(
                (precisions[:-1] + recalls[:-1]) > 0,
                2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12),
                0.0,
            )
            best_idx = int(np.argmax(f1_scores))
            decision_threshold = float(pr_thresholds[best_idx])
            print(f"   [SOGLIA] Soglia scelta sul validation set (massimizzazione F1): "
                  f"{decision_threshold:.4f} (Precision={precisions[best_idx]:.4f}, "
                  f"Recall={recalls[best_idx]:.4f}, F1={f1_scores[best_idx]:.4f} sul validation; "
                  f"default implicito di .predict() = 0.50).")
            local_preds = (local_proba >= decision_threshold).astype(int)

            # --- Soglia alternativa: vincolo di business FPR <= TARGET_FPR_CONSTRAINT ---
            # Calcolata sullo STESSO validation set (mai sul test), criterio
            # di Neyman-Pearson: tra le soglie che rispettano il vincolo,
            # quella con la recall (TPR) più alta -- non un secondo modello,
            # solo un secondo punto operativo sulla stessa ROC. Tenuta
            # SEPARATA dalla soglia F1-max scelta sopra (quella resta la
            # soglia "ufficiale" del modello, usata per local_preds): questo
            # è un confronto aggiuntivo nel report, non una sostituzione.
            fpr_val, tpr_val, roc_thresholds = roc_curve(y_val, val_proba)
            acceptable = np.where(fpr_val <= TARGET_FPR_CONSTRAINT)[0]
            if len(acceptable) > 0:
                best_constrained_idx = int(acceptable[np.argmax(tpr_val[acceptable])])
                threshold_low_fpr = float(roc_thresholds[best_constrained_idx])
                fpr_at_low = float(fpr_val[best_constrained_idx])
                tpr_at_low = float(tpr_val[best_constrained_idx])
                print(f"   [SOGLIA] Soglia alternativa vincolata (FPR <= {TARGET_FPR_CONSTRAINT*100:.1f}% "
                      f"sul validation): {threshold_low_fpr:.4f} (FPR={fpr_at_low:.4f}, "
                      f"Recall={tpr_at_low:.4f} sul validation).")
                local_preds_low_fpr = (local_proba >= threshold_low_fpr).astype(int)
            else:
                # Nessuna soglia nella ROC del validation rispetta il vincolo
                # (può succedere con un vincolo molto stringente su un
                # validation piccolo/rumoroso): nessuna soglia alternativa
                # calcolabile, non un errore.
                threshold_low_fpr = None
                local_preds_low_fpr = None
                print(f"   [SOGLIA] Nessuna soglia nel validation rispetta il vincolo "
                      f"FPR <= {TARGET_FPR_CONSTRAINT*100:.1f}%: soglia alternativa non disponibile.")
        else:
            # Nessun validation set disponibile (dataset sintetico): soglia
            # di default, comportamento invariato rispetto a prima.
            decision_threshold = 0.5
            local_preds = local_preds_default
            threshold_low_fpr = None
            local_preds_low_fpr = None
    else:
        local_proba = None
        local_preds_default = None
        decision_threshold = None
        threshold_low_fpr = None
        local_preds_low_fpr = None
        local_preds = tree_clf.predict(X_test)

    tempo_inferenza_totale = time.perf_counter() - start_inferenza

    # ---------------------------------------------------------
    # Valutazione, diramata per tipo di task (user_tree_type)
    # ---------------------------------------------------------
    if user_tree_type == "classifier":
        test_accuracy = np.mean(local_preds == y_test)
        test_precision = precision_score(y_test, local_preds, zero_division=0)
        test_recall = recall_score(y_test, local_preds, zero_division=0)
        test_f1 = f1_score(y_test, local_preds, zero_division=0)
        test_roc_auc = roc_auc_score(y_test, local_proba) if local_proba is not None else 0.0
        cm = confusion_matrix(y_test, local_preds)
        tn, fp, fn, tp = cm.ravel()
        test_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        # Metriche a soglia di default (0.50), tenute SOLO per confronto in
        # relazione -- non sono quelle "ufficiali" del modello, che restano
        # quelle a decision_threshold (vedi sopra).
        test_accuracy_default = np.mean(local_preds_default == y_test)
        test_precision_default = precision_score(y_test, local_preds_default, zero_division=0)
        test_recall_default = recall_score(y_test, local_preds_default, zero_division=0)
        test_f1_default = f1_score(y_test, local_preds_default, zero_division=0)
        cm_default = confusion_matrix(y_test, local_preds_default)
        tn_d, fp_d, _, _ = cm_default.ravel()
        test_fpr_default = float(fp_d / (fp_d + tn_d)) if (fp_d + tn_d) > 0 else 0.0

        # Secondo operating point: soglia vincolata FPR<=TARGET_FPR_CONSTRAINT
        # sul validation, VALUTATA QUI sul test set (mai usata per scegliere
        # la soglia, solo per riportarne l'effetto reale) -- vedi discussione
        # su criterio di Neyman-Pearson sopra.
        if local_preds_low_fpr is not None:
            test_accuracy_low_fpr = np.mean(local_preds_low_fpr == y_test)
            test_precision_low_fpr = precision_score(y_test, local_preds_low_fpr, zero_division=0)
            test_recall_low_fpr = recall_score(y_test, local_preds_low_fpr, zero_division=0)
            test_f1_low_fpr = f1_score(y_test, local_preds_low_fpr, zero_division=0)
            cm_low_fpr = confusion_matrix(y_test, local_preds_low_fpr)
            tn_l, fp_l, _, _ = cm_low_fpr.ravel()
            test_fpr_low_fpr = float(fp_l / (fp_l + tn_l)) if (fp_l + tn_l) > 0 else 0.0
            soglia_vincolata_fpr = {
                "soglia": threshold_low_fpr,
                "accuracy": test_accuracy_low_fpr,
                "precision": test_precision_low_fpr,
                "recall": test_recall_low_fpr,
                "f1": test_f1_low_fpr,
                "fpr": test_fpr_low_fpr,
                "matrice_confusione": cm_low_fpr.tolist(),
            }
        else:
            soglia_vincolata_fpr = None

        metriche_test = {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "roc_auc": test_roc_auc,
            "fpr": test_fpr,
            "decision_threshold": decision_threshold,
            "confronto_soglia_default_0_50": {
                "accuracy": test_accuracy_default,
                "precision": test_precision_default,
                "recall": test_recall_default,
                "f1": test_f1_default,
                "fpr": test_fpr_default,
            },
            "soglia_vincolata_fpr": soglia_vincolata_fpr,
        }
    else:
        test_mse = mean_squared_error(y_test, local_preds)
        test_rmse = float(np.sqrt(test_mse))
        test_mae = mean_absolute_error(y_test, local_preds)
        test_r2 = r2_score(y_test, local_preds)

        metriche_test = {
            "mse": test_mse,
            "rmse": test_rmse,
            "mae": test_mae,
            "r2": test_r2
        }

    metadata_pipeline = {
        "modello_addestrato": tree_clf,
        "features_mappate": list(X_train.columns),
        # Permutation importance OOB (feature -> percent increase error rate),
        # calcolata dalla feature selection -- None per il sintetico (segnale/
        # rumore noto a priori, nessuna importance calcolata). Persistita qui
        # per poterla riusare in diagnostica senza dover rifare il fit.
        "feature_importance_scores": (
            feature_importance_scores.to_dict() if feature_importance_scores is not None else None
        ),
        # Scomposizione completa dei tempi: senza io_time/etl_time non è
        # possibile confrontare correttamente con il cluster, il cui tempo
        # totale include sempre la preparazione dati (vedi
        # CentralizedOrchestrator.last_etl_seconds).
        "baseline_tempi_locali": {
            "tempo_totale_tuning": tempo_tuning,
            "tempo_medio_fit_tuning": tempo_medio_fit_tuning,
            "io_time": io_time,
            "etl_time": etl_time,
            "t_seq": t_seq,
            "t_1node_parallel": t_1node_parallel,
            "speedup_multicore_locale": speedup_multicore,
            "cpu_count": cpu_disponibili,
            "tempo_inferenza_totale": tempo_inferenza_totale
        },
        # Dimensioni effettive: servono a verificare a colpo d'occhio che
        # baseline e cluster abbiano lavorato sullo stesso volume di dati.
        "dataset_shape": {
            "train": list(X_train.shape),
            "test": list(X_test.shape),
            "validation": list(X_val.shape) if X_val is not None else None,
            "dataset_type": dataset_type,
            "tree_type": user_tree_type,
        },
        "iperparametri_usati": dict(hp),
        "metriche_test": metriche_test
    }

    with open(pickle_path_final, "wb") as f:
        pickle.dump(metadata_pipeline, f)
    print(f"[OK] Pipeline locale (file .pkl) salvata in: '{pickle_path_final}'")

    # ---------------------------------------------------------
    # FASE 5: OUTPUT REPORT COMPLETO Condizionato sul Task
    # ---------------------------------------------------------
    print("\n" + DOPPIA_LINEA)
    print(f"{'REPORT ESTESO DI VALIDAZIONE E BENCHMARK':^{LUNGHEZZA_LINEA}}")
    print(DOPPIA_LINEA)

    if dataset_type == "real" and oob_search_results is not None:
        top_n = min(15, len(oob_search_results))
        print(f"\n1. RICERCA IPERPARAMETRICA VIA STIMA OOB (Breiman 2001, Sec. 3.1)")
        print(LINEA_SINGOLA)
        print(f"  {len(oob_search_results)} combinazioni esplorate, un solo fit ciascuna "
              f"(nessuna k-fold Cross-Validation — vedi oob_hyperparameter_search).")
        print(f"  Le {top_n} migliori per OOB F1-Score:")
        print(LINEA_SINGOLA)
        print(f"  {'Rank':<5} | {'n_est':<6} | {'depth':<6} | {'OOB Acc':<9} | {'OOB Prec':<9} | "
              f"{'OOB Rec':<9} | {'OOB F1':<9}")
        print(LINEA_SINGOLA)
        for rank, r in enumerate(oob_search_results[:top_n], start=1):
            p = r["params"]
            depth_str = str(p.get("max_depth")) if p.get("max_depth") is not None else "None"
            print(f"  {rank:<5} | {p.get('n_estimators'):<6} | {depth_str:<6} | "
                  f"{r['oob_accuracy']*100:8.2f}% | {r['oob_precision']*100:8.2f}% | "
                  f"{r['oob_recall']*100:8.2f}% | {r['oob_f1']*100:8.2f}%")
        print(LINEA_SINGOLA)

        # NOTA METODOLOGICA: qui la media/dev.std è calcolata sulle DIVERSE
        # COMBINAZIONI di iperparametri esplorate, NON sui fold di un unico
        # modello (non esistono più fold). Non è quindi un indicatore di
        # varianza del modello vincitore — è un indicatore di quanto le
        # prestazioni OOB variano nello spazio degli iperparametri esplorato.
        # Le due cose vanno tenute distinte in relazione: la prima versione
        # (CV) misurava la stabilità del modello scelto su split diversi; qui
        # si misura la sensibilità della foresta alla scelta di iperparametri.
        print(f"\n2. SPREAD DELLE PRESTAZIONI OOB SULLE {len(oob_search_results)} COMBINAZIONI ESPLORATE")
        print("   (dispersione nello spazio degli iperparametri, NON varianza tra fold)")
        print(LINEA_SINGOLA)
        metriche_oob = [
            ("OOB ACCURACY", [r["oob_accuracy"] for r in oob_search_results]),
            ("OOB PRECISION", [r["oob_precision"] for r in oob_search_results]),
            ("OOB RECALL", [r["oob_recall"] for r in oob_search_results]),
            ("OOB F1-SCORE", [r["oob_f1"] for r in oob_search_results]),
        ]
        for nome, array in metriche_oob:
            media = np.mean(array) * 100
            dev_std = np.std(array) * 100
            print(f"  ▸ {nome:<25} : {media:6.2f}%  (± {dev_std:.2f}%)")

        print(f"\n2b. SCELTA FINALE DI n_estimators")
        print(LINEA_SINGOLA)
        print(f"  ▸ {motivazione_n_estimators}")
    else:
        if dataset_type == "real":
            print(f"\n1-2. TUNING SALTATO — iperparametri riusati da manifesto esistente "
                  f"('{REAL_CONFIG_PATH}')")
        elif user_tree_type == "classifier":
            print(f"\n1-2. TUNING SALTATO — iperparametri ereditati dal tuning sul dataset reale "
                  f"(stesso task, stessa distribuzione binaria sbilanciata)")
        else:
            print(f"\n1-2. REGRESSORE SINTETICO — NESSUN TUNING, iperparametri di default sklearn "
                  f"tranne n_estimators (fonte: {fonte_n_estimators})")
        print(LINEA_SINGOLA)
        print(f"  ▸ Iperparametri applicati: {hp}")

    print(f"\n3. PERFORMANCE REALI SUL TEST SET ({user_tree_type.upper()})")
    print(LINEA_SINGOLA)
    if user_tree_type == "classifier":
        soglia_desc = (
            "scelta su validation set separato, massimizzazione F1 -- vedi sezione 3b"
            if X_val is not None else
            "default 0.50 (nessun validation set per questo dataset)"
        )
        print(f"  ▸ Soglia di decisione: {metriche_test['decision_threshold']:.4f} ({soglia_desc})")
        print(f"  ▸ ACCURACY SUL TEST SET   : {metriche_test['accuracy'] * 100:6.2f}%")
        print(f"  ▸ PRECISION SUL TEST SET  : {metriche_test['precision'] * 100:6.2f}%")
        print(f"  ▸ RECALL SUL TEST SET     : {metriche_test['recall'] * 100:6.2f}%")
        print(f"  ▸ F1-SCORE SUL TEST SET   : {metriche_test['f1'] * 100:6.2f}%")
        print(f"  ▸ FPR SUL TEST SET        : {metriche_test['fpr'] * 100:6.2f}%")
        print(f"  ▸ ROC-AUC SUL TEST SET    : {metriche_test['roc_auc']:6.4f}")
        print("\n  Matrice di Confusione sul Test Set:")
        for riga in cm:
            print(" " * 6 + " ".join(f"[{val:4d}]" for val in riga))

        d = metriche_test["confronto_soglia_default_0_50"]
        print(f"\n3b. CONFRONTO CON LA SOGLIA DI DEFAULT (0.50, quella implicita di .predict())")
        print(LINEA_SINGOLA)
        print(f"  {'Metrica':<12} | {'Soglia scelta':>14} | {'Soglia 0.50':>12} | {'Delta':>8}")
        for nome, chiave in [("Accuracy", "accuracy"), ("Precision", "precision"),
                              ("Recall", "recall"), ("F1-Score", "f1"), ("FPR", "fpr")]:
            v_scelta = metriche_test[chiave] * 100
            v_default = d[chiave] * 100
            print(f"  {nome:<12} | {v_scelta:13.2f}% | {v_default:11.2f}% | {v_scelta - v_default:+7.2f}pp")
        print("  ▸ Nota: la soglia è scelta su un validation set separato (ritagliato dal train "
              "PRIMA dell'undersampling, quindi con la vera distribuzione sbilanciata), non sul "
              "test set -- evita sia il leakage (tarare la soglia sullo stesso set su cui si "
              "riportano le metriche finali) sia la distorsione di calibrazione che una soglia "
              "scelta su un train ribilanciato erediterebbe.")

        sv = metriche_test["soglia_vincolata_fpr"]
        print(f"\n3c. OPERATING POINT ALTERNATIVO -- VINCOLO DI BUSINESS FPR <= "
              f"{TARGET_FPR_CONSTRAINT*100:.1f}% (criterio di Neyman-Pearson)")
        print(LINEA_SINGOLA)
        if sv is not None:
            print(f"  ▸ Soglia: {sv['soglia']:.4f} (scelta sul validation: tra le soglie con FPR "
                  f"<= {TARGET_FPR_CONSTRAINT*100:.1f}%, quella con recall più alta)")
            print(f"  {'Metrica':<12} | {'F1-max':>10} | {'FPR<={:.0f}%'.format(TARGET_FPR_CONSTRAINT*100):>10} | {'Delta':>8}")
            for nome, chiave in [("Accuracy", "accuracy"), ("Precision", "precision"),
                                  ("Recall", "recall"), ("F1-Score", "f1"), ("FPR", "fpr")]:
                v_f1max = metriche_test[chiave] * 100
                v_vinc = sv[chiave] * 100
                print(f"  {nome:<12} | {v_f1max:9.2f}% | {v_vinc:9.2f}% | {v_vinc - v_f1max:+7.2f}pp")
            print("\n  Matrice di Confusione (soglia vincolata FPR):")
            for riga in sv["matrice_confusione"]:
                print(" " * 6 + " ".join(f"[{val:4d}]" for val in riga))
            print(f"  ▸ Nota: soglia scelta e valutata rispettivamente su validation e test set "
                  f"(mai la stessa), stesso principio della soglia F1-max in 3b -- nessun leakage. "
                  f"Utile per un confronto esplicito 'F1-max' (obiettivo bilanciato) vs 'FPR "
                  f"vincolato' (obiettivo operativo tipico di un SOC, dove un FPR alto produce "
                  f"alert fatigue), da motivare in relazione in base al caso d'uso.")
        else:
            print(f"  ▸ Nessuna soglia nel validation set rispetta il vincolo FPR <= "
                  f"{TARGET_FPR_CONSTRAINT*100:.1f}%: operating point alternativo non disponibile "
                  f"con questo modello/validation set.")
    else:
        print(f"  ▸ MSE SUL TEST SET   : {metriche_test['mse']:.4f}")
        print(f"  ▸ RMSE SUL TEST SET  : {metriche_test['rmse']:.4f}")
        print(f"  ▸ MAE SUL TEST SET   : {metriche_test['mae']:.4f}")
        print(f"  ▸ R² SUL TEST SET    : {metriche_test['r2']:.4f}")

    if feature_importance_scores is not None:
        print(f"\n3d. IMPORTANZA DELLE FEATURE (permutation importance OOB, "
              f"top {TOP_N_FEATURE_IMPORTANCE} tra quelle selezionate)")
        print(LINEA_SINGOLA)
        feature_selezionate = set(dizionario_feature.get("salvate", []))
        top_importance = (
            feature_importance_scores[feature_importance_scores.index.isin(feature_selezionate)]
            .sort_values(ascending=False)
            .head(TOP_N_FEATURE_IMPORTANCE)
        )
        for rank, (feat, score) in enumerate(top_importance.items(), start=1):
            print(f"  {rank:>2}. {feat:<30} {score:8.4f}")
        print("  ▸ Punteggio = percent increase OOB error rate quando la feature è permutata "
              "(più alto = più importante -- vedi CICIDSFeatureSelector).")

        presenti = [f for f in SUSPECT_LEAKAGE_FEATURES if f in feature_selezionate]
        if presenti:
            print(f"\n  [ATTENZIONE] Feature nella watchlist 'socket surrogate/artefatto tool "
                  f"d'attacco' (letteratura CICIDS2018) ANCORA presenti nel set finale: {presenti}. "
                  f"Non è di per sé una prova di leakage, ma vale la pena verificare esplicitamente "
                  f"in relazione se e quanto la loro rimozione cambia le metriche (segnale genuino "
                  f"vs artefatto della configurazione fissa degli strumenti usati per generare gli "
                  f"attacchi nel dataset).")
        else:
            print(f"\n  Nessuna feature della watchlist {SUSPECT_LEAKAGE_FEATURES} presente nel set "
                  f"finale selezionato.")

    print(f"\n4. DIAGNOSTICA TEMPORALE E PROFILAZIONE HARDWARE")
    print(LINEA_SINGOLA)
    print(f"  • Tempo Totale di Tuning (ricerca OOB) : {tempo_tuning:8.4f} s")
    print(f"  • Tempo Medio per Singolo Fit (tuning) : {tempo_medio_fit_tuning:8.4f} s")
    print(f"  • Tempo di Caricamento Dati (I/O)      : {io_time:8.4f} s")
    print(f"  • Tempo di Trasformazione (Process)    : {etl_time:8.4f} s")
    print(f"  • T_seq  - Addestramento MONOCORE (n_jobs=1)   : {t_seq:8.4f} s")
    print(f"  • T_1node - Addestramento MULTICORE (n_jobs=-1): {t_1node_parallel:8.4f} s  "
          f"[{cpu_disponibili} core, speedup locale {speedup_multicore:.2f}x]")
    print(f"  • Tempo Totale di Inferenza (Testing Set) : {tempo_inferenza_totale:8.4f} s")
    print(f"  • Volume dati: train={X_train.shape}  test={X_test.shape}"
          + (f"  validation={X_val.shape}" if X_val is not None else ""))

    print(f"\n5. COME CONFRONTARE QUESTI NUMERI COL CLUSTER")
    print(LINEA_SINGOLA)
    print("  ▸ Confronta il tempo di ADDESTRAMENTO del cluster al NETTO dell'ETL")
    print("    (CentralizedOrchestrator.last_etl_seconds) contro T_seq / T_1node:")
    print("    il totale del cluster include sempre la preparazione dati, la baseline no.")
    print("  ▸ Usa T_seq per speedup ed efficienza 'da manuale', T_1node per rispondere")
    print("    alla domanda pratica 'conviene distribuire invece di usare una macchina sola?'.")
    print("  ▸ Verifica che 'Volume dati' qui sopra coincida con lo shape stampato dal cluster:")
    print("    se differiscono, il confronto NON è valido (controlla config_synthetic.json).")

if __name__ == "__main__":
    run_baseline()