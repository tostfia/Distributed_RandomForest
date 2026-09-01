import os
import json
import time
import numpy as np
import pickle
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # i tuoi print restano l'unico output

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score, f1_score, confusion_matrix
from src.shared.utilities.undersampling import undersample_majority_class
from src.shared.config import SystemConfig

# Import delle utility condivise e del loader con campionamento probabilistico
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
from src.shared.utilities.datasplitter import StratifiedDataSplitter
from src.shared.utilities.featureselection import CICIDSFeatureSelector

# ---------------------------------------------------------------------------
# Configurazione di riferimento FISSA per il task sintetico di REGRESSIONE.
#
# NOTA METODOLOGICA: a differenza del task sintetico di classificazione (dove
# ereditare gli iperparametri dal tuning sul dataset reale ha senso, perché è
# lo stesso algoritmo/task), per la regressione non esiste alcuna garanzia che
# gli iperparametri ottimizzati per un RandomForestClassifier su un problema
# di classificazione binaria sbilanciata siano sensati per un
# RandomForestRegressor su dati sintetici continui e bilanciati.
#
# Si usa quindi una configurazione dichiarata a priori (non derivata da
# tuning) e tenuta IDENTICA in ogni esperimento di scalabilità — baseline
# locale e ogni run del cluster distribuito, a qualunque numero di worker —
# in modo da isolare l'effetto della scalabilità (numero di nodi, dimensione
# del dataset) dalla complessità del modello.
# ---------------------------------------------------------------------------
SYNTHETIC_REGRESSOR_REFERENCE_HP = {

    "n_estimators": 35,
    "max_depth": None,
    "min_samples_split": 2,

    "max_features": 1 / 3,
    "criterion": "squared_error",
    "bootstrap": True,
    "max_samples": 1.0,
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
        #   - class_weight [None,'balanced']: uniche due opzioni discrete
        #     sensate (esclusi dizionari custom).
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
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
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

def optuna_refine_n_estimators(X_train, y_train, fixed_params, candidate_values,
                                random_state, refit_metric="f1", n_jobs=None):
    """
    A parità di TUTTI gli altri iperparametri (fixed_params, tranne
    n_estimators), esplora candidate_values in modo esaustivo (GridSampler).
    Risolve il problema del vecchio scegli_n_estimators: prima confrontava
    trial con iperparametri diversi tra loro, qui ogni riga del risultato
    è comparabile con tutte le altre per costruzione.

    LIMITE NOTO -- NIENTE warm_start, a differenza della verifica indipendente
    più fine in analyze_classification_n_estimators.py: qui ogni valore di
    n_estimators viene fittato con un fit INDIPENDENTE (foresta separata da
    zero), non con una singola foresta fatta crescere incrementalmente. Con
    fit indipendenti, lo schema di seeding interno di scikit-learn assegna
    semi diversi ai singoli alberi a seconda del numero totale di stimatori
    richiesto: la curva qui prodotta confronta quindi foreste leggermente
    diverse punto per punto, non la stessa foresta osservata a stadi di
    crescita diversi (stesso problema, già diagnosticato e risolto altrove
    in questo lavoro passando a warm_start=True, che qui NON è stato
    reintrodotto per restare compatibile con l'interfaccia a
    objective-function di Optuna, dove ogni trial è indipendente per
    costruzione). Trattare questo sweep come indicativo/rapido, non come
    fonte autorevole: per la scelta finale di n_estimators resta
    analyze_classification_n_estimators.py (warm_start, griglia fine a passo
    5) la verifica di riferimento.
    """
    if n_jobs is None:
        cpu_count = os.cpu_count() or 2
        n_jobs = max(1, cpu_count - 1)

    results = []
    total = len(candidate_values)

    def objective(trial):
        n_estimators = trial.suggest_categorical("n_estimators", candidate_values)
        params = {**fixed_params, "n_estimators": n_estimators}

        print(f"[Optuna n_est sweep] n_estimators={n_estimators} "
              f"({len(results) + 1}/{total}) in corso ...", flush=True)

        rf = RandomForestClassifier(**params, random_state=random_state,
                                     oob_score=True, n_jobs=n_jobs)
        start = time.perf_counter()
        rf.fit(X_train, y_train)
        fit_time = time.perf_counter() - start

        metrics = _oob_classification_metrics(rf, y_train)
        print(f"[Optuna n_est sweep] n_estimators={n_estimators} completato in {fit_time:.1f}s — "
              f"OOB F1={metrics['oob_f1']:.4f}", flush=True)

        results.append({"params": params, "fit_time": fit_time, **metrics})
        return metrics[f"oob_{refit_metric}"]

    sampler = optuna.samplers.GridSampler({"n_estimators": candidate_values}, seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=total)

    results.sort(key=lambda r: r["params"]["n_estimators"])
    return results


# ---------------------------------------------------------------------------
# SELEZIONE INTERATTIVA DI n_estimators A VALLE DEL TUNING OOB
#
# MOTIVAZIONE: la ricerca OOB (oob_hyperparameter_search) massimizza l'F1 su
# tutto lo spazio degli iperparametri esplorato, ma il guadagno marginale di
# alberi aggiuntivi tende a saturare rapidamente (proprietà nota delle random
# forest, si veda anche l'esempio ufficiale scikit-learn "OOB Errors for
# Random Forests") mentre il costo di training cresce all'incirca linearmente
# con n_estimators. Per la baseline locale — che deve girare su CPU in tempi
# ragionevoli, si veda il ricevimento del 22/05/2026 con il Prof. Russo Russo
# ("riducete il numero di alberi banalmente" se l'addestramento è troppo
# lento) — può convenire accettare un F1 leggermente più basso in cambio di
# un training molto più rapido.
#
# La scelta finale resta ESPLICITA a runtime, sullo stesso modello di
# SKIP_TUNING più sopra: chi lancia lo script vede il trade-off misurato e
# decide, invece di ereditare in silenzio un valore scritto una volta e mai
# più rivisto.
# ---------------------------------------------------------------------------
def scegli_n_estimators(best_params, oob_search_results, tolerance=0.005):
    """
    Restituisce (best_params_aggiornato, motivazione_str) dopo aver chiesto
    all'utente se accettare l'n_estimators trovato dal tuning OOB o applicare
    una riduzione per contenere i tempi della baseline locale.

    tolerance: soglia di F1 OOB (in punti, es. 0.005 = 0.5pp) entro cui un
    n_estimators più basso di quello ottimo è considerato "equivalente" ai
    fini pratici.

    Se oob_search_results è None (tuning saltato, iperparametri riusati da
    un manifesto esistente), non c'è alcuna analisi di trade-off disponibile
    per QUESTO run: la funzione lo dichiara esplicitamente e ritorna
    best_params invariato.
    """
    best_params = dict(best_params)

    if oob_search_results is None:
        print(f"\n  [INFO] Tuning saltato (iperparametri riusati da manifesto esistente): "
              f"nessuna analisi di trade-off su n_estimators disponibile in questo run. "
              f"Uso n_estimators dal manifesto: {best_params.get('n_estimators')}.")
        return best_params, "n_estimators riusato da manifesto esistente (tuning saltato in questo run)."

    tuned_n = best_params.get("n_estimators")
    tuned_row = next(r for r in oob_search_results if r["params"]["n_estimators"] == tuned_n)
    best_f1 = tuned_row["oob_f1"]

    candidates = [r for r in oob_search_results if best_f1 - r["oob_f1"] <= tolerance]
    suggested_row = min(candidates, key=lambda r: r["params"]["n_estimators"])
    suggested_n = suggested_row["params"]["n_estimators"]

    print(f"\n  n_estimators trovato dal tuning: {tuned_n} "
          f"(OOB F1={best_f1 * 100:.2f}%, fit_time={tuned_row['fit_time']:.1f}s).")
    if suggested_n < tuned_n:
        print(f"  Il più piccolo n_estimators esplorato con OOB F1 entro "
              f"{tolerance * 100:.1f}pp dal migliore è {suggested_n} "
              f"(OOB F1={suggested_row['oob_f1'] * 100:.2f}%, fit_time={suggested_row['fit_time']:.1f}s).")
    else:
        print("  Nessun n_estimators più basso resta entro la tolleranza: "
              "il valore del tuning è già il più efficiente disponibile.")

    print("\n  [1] Usa il valore ottimo del tuning")
    if suggested_n < tuned_n:
        print(f"  [2] Applica la riduzione suggerita ({suggested_n}) — default")
    else:
        print("  [2] (non disponibile: nessuna riduzione entro tolleranza)")
    print("  [3] Inserisci un valore custom")
    scelta_override = input("  Scelta [Default: 2]: ").strip() or "2"

    if scelta_override == "1" or (scelta_override == "2" and suggested_n >= tuned_n):
        motivazione = (f"n_estimators confermato al valore del tuning: {tuned_n} "
                        f"(OOB F1={best_f1 * 100:.2f}%).")
        print(f"[OK] {motivazione}")
        return best_params, motivazione

    if scelta_override == "3":
        custom = input("  Inserisci n_estimators desiderato: ").strip()
        try:
            custom_n = int(custom)
        except ValueError:
            motivazione = f"Valore custom non valido, mantengo il valore del tuning ({tuned_n})."
            print(f"[ATTENZIONE] {motivazione}")
            return best_params, motivazione

        custom_row = next((r for r in oob_search_results
                            if r["params"]["n_estimators"] == custom_n), None)
        extra = (f" (OOB F1={custom_row['oob_f1'] * 100:.2f}%, misurato)"
                 if custom_row else " (non esplorato dal tuning, nessuna stima OOB disponibile)")
        motivazione = (f"n_estimators: {tuned_n} -> {custom_n} "
                        f"(valore custom inserito dall'utente{extra}).")
        print(f"[OVERRIDE] {motivazione}")
        best_params["n_estimators"] = custom_n
        return best_params, motivazione

    # scelta_override == "2" e suggested_n < tuned_n: applica la riduzione suggerita.
    motivazione = (f"n_estimators: {tuned_n} (OOB F1={best_f1 * 100:.2f}%, "
                    f"fit_time={tuned_row['fit_time']:.1f}s) -> {suggested_n} "
                    f"(OOB F1={suggested_row['oob_f1'] * 100:.2f}%, "
                    f"fit_time={suggested_row['fit_time']:.1f}s, "
                    f"delta F1 <= {tolerance * 100:.1f}pp). Scelta per contenere i tempi della "
                    f"baseline locale su CPU (vedi Ricevimento 22/05/2026).")
    print(f"[OVERRIDE] {motivazione}")
    best_params["n_estimators"] = suggested_n
    return best_params, motivazione


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
    SAMPLE_FRACTION = 0.05
    # Campionamento RIBILANCIATO per giorno di cattura, al posto della
    # sample_fraction uniforme (che favorisce sistematicamente i giorni più
    # grandi — es. Thuesday-20-02-2018, da solo quasi la metà del dataset
    # totale). Vincolo: deve restare sotto il file più piccolo tra i 10
    # (Thursday-01-03-2018, 331.125 righe), altrimenti quel giorno smette di
    # contribuire alla pari con gli altri. 100.000 è stato scelto per
    # restare con margine sotto quella soglia, pur raddoppiando circa il
    # volume grezzo complessivo rispetto al precedente SAMPLE_FRACTION=0.05
    # (~811.650 -> ~1.000.000 righe), per aumentare la varietà del train set
    # e verificare se questo sposta in avanti il plateau di n_estimators
    # osservato nella diagnostica OOB (vedi analyze_classification_n_estimators.py).
    # SAMPLE_FRACTION resta definita sopra ma viene ignorata quando
    # TARGET_ROWS_PER_DAY non è None (vedi RawCSVDataLoader).
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
    # Under-sampling della classe maggioritaria (Benign), solo sul train set
    # -- vedi undersampling.py per la motivazione completa. ratio=1.0
    # produce un bilanciamento 1:1 (maggioritaria ridotta alla stessa
    # dimensione della minoritaria), lo standard di riferimento più comune
    # in letteratura come punto di partenza prima di esplorare rapporti
    # diversi.
    UNDERSAMPLING_RATIO = 1.0
    # Soglia di tolleranza per la riduzione di n_estimators post-tuning
    # (vedi scegli_n_estimators): 0.005 = 0.5 punti percentuali di F1 OOB.
    OOB_F1_TOLERANCE = 0.005

    target_col = "Label"
    BOOT_CONFIG_PATH = os.path.join("./.local_storage", "config.json")
    OUTPUT_DIR = "./outputs_baseline"
    REAL_CONFIG_PATH = os.path.join(OUTPUT_DIR, "config_real.json")
    SYNTHETIC_CONFIG_PATH = os.path.join(OUTPUT_DIR, "config_synthetic.json")
    dataset_type = "real"
    user_tree_type = "classifier"

    sys_cfg = SystemConfig()
    print(f" • Ambiente infrastrutturale rilevato: {sys_cfg.env.upper()}")

    # ---------------------------------------------------------
    # SCELTA 1: eseguire il tuning OOB (FASE 2, ~50 fit, costoso) o riusare
    # gli iperparametri già presenti in un config_real.json esistente
    # (salta la ricerca, va dritto alla FASE 4 con quegli iperparametri).
    # Utile per iterare rapidamente su altre parti della pipeline (es.
    # preprocessing, feature selection, dimensione campione) senza dover
    # rifare mezz'ora di ricerca ogni volta che non serve.
    # ---------------------------------------------------------
    print("\nEseguire il tuning iperparametrico (FASE 2) o riusare gli iperparametri "
          "già presenti in un manifesto esistente?")
    print("  [1] Esegui il tuning (default) ")
    print("  [2] Salta il tuning — riusa gli iperparametri già presenti in "
          "'outputs_baseline/config_real.json'")
    scelta_tuning = input("  Scelta [Default: 1]: ").strip() or "1"
    SKIP_TUNING = (scelta_tuning == "2")

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
            print(" [INFO] Task REGRESSOR: il tuning sul dataset reale non è richiesto "
                  "(si usa la configurazione di riferimento fissa dichiarata a inizio file).")

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

        # Default allineato a SyntheticDataLoader (n_samples=300000). Prima qui
        # il default era 500000: in assenza di un manifesto la baseline generava
        # un dataset 1.67x più grande di quello del cluster, e i tempi di
        # addestramento non erano confrontabili.
        n_samples = tmp_cfg.get("n_samples", 300000)
        n_features = tmp_cfg.get("n_features", 30)
        # 50% delle feature informative, l'altra metà rumore puro per
        # costruzione (non correlato al target): scelta dichiarata, non
        # arbitraria. Non serve applicare feature selection sul sintetico
        # (dizionario_feature resta vuoto più sotto) proprio perché la
        # separazione segnale/rumore è nota e controllata a priori dal
        # generatore — a differenza del reale, dove va stimata empiricamente
        # (da qui la permutation importance OOB in CICIDSFeatureSelector).
        n_informative_reg = tmp_cfg.get("n_informative_reg", int(n_features * 0.5))
        # noise=10.0 verificato empiricamente in una diagnostica precedente
        # (script rimosso da questo progetto): con n_informative=15/30 la
        # deviazione standard del segnale "pulito" (noise=0) è ≈212.8, quindi
        # noise=10.0 corrisponde a circa il 4.7% della variabilità naturale
        # del target (SNR ≈ 21). Livello moderato: rende il problema non
        # banale senza far dominare il rumore sul segnale.
        noise = tmp_cfg.get("noise", 10.0)
        # Stessi default interni di SyntheticDataLoader per la classificazione,
        # ma calcolati QUI e passati esplicitamente al costruttore. Prima
        # venivano solo scritti nel manifesto (dataset_gen_params) senza mai
        # essere passati al loader: quest'ultimo ricadeva sui suoi default
        # interni rileggendo 'outputs_baseline/config_synthetic.json', un file
        # che al momento di questa chiamata è o assente (primo run) o ancora
        # quello della run PRECEDENTE (questa run lo sovrascrive solo più
        # avanti) — quindi il manifesto "nuovo" non controllava mai davvero
        # cosa veniva effettivamente generato.
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
            dataset_gen_params["n_informative_reg"] = n_informative_reg
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
            loader_kwargs["n_informative_reg"] = n_informative_reg
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
            sample_fraction=SAMPLE_FRACTION,  # ignorata: target_rows_per_day ha priorità
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
            dizionario_feature = fs.feature_summary_
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
            dizionario_feature = best_fs.feature_summary_

        # Ricostruisce X_train/y_train/X_test/y_test dal train/test ORA
        # ridotto alle feature selezionate (nel ramo SKIP_TUNING appena
        # calcolate sopra; nel ramo di tuning, quelle del max_features
        # vincente) -- necessario perché la costruzione precedente usava
        # ancora il dataset completo a 69 feature, dato che la feature
        # selection è stata spostata dopo quel punto.
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        print(f" • Volume Train (DOPO la feature selection): {X_train.shape} | "
              f"Volume Test: {X_test.shape}")

        # ---------------------------------------------------------------
        # SCELTA 2: accettare l'n_estimators trovato dal tuning OOB, o
        # applicare una riduzione per contenere i tempi della baseline
        # locale? Vedi il commento esteso sopra scegli_n_estimators() per
        # la motivazione completa. La riga [OVERRIDE]/[OK] stampata qui
        # sostituisce il vecchio "N_ESTIMATORS_OVERRIDE = 30" hardcoded.
        # ---------------------------------------------------------------
        tuned_n = best_params["n_estimators"]
        candidate_n_estimators = sorted(
            {v for v in [10, 20, 30, 40, 60, 80, 100, 150, 200] if v <= tuned_n} | {tuned_n}
        )

        n_estimators_results = optuna_refine_n_estimators(
            X_train, y_train,
            fixed_params=best_params,
            candidate_values=candidate_n_estimators,
            random_state=RANDOM_SEED,
        )

        best_params, motivazione_n_estimators = scegli_n_estimators(
            best_params, n_estimators_results, tolerance=OOB_F1_TOLERANCE
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

    else:
        print("\n>>> FASE 2: SALTATA — riuso iperparametri ottenuti dal tuning sul dataset reale")
        tempo_tuning = 0.0
        tempo_medio_fit_tuning = 0.0
        oob_search_results = None

        if user_tree_type == "classifier":
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
            # Task REGRESSOR: non ereditiamo gli iperparametri del classificatore
            # reale (algoritmo e distribuzione dei dati diversi). Si usa la
            # configurazione di riferimento fissa definita a inizio file.
            print(" [INFO] Task REGRESSOR: uso configurazione di riferimento fissa "
                  "(non ereditata dal tuning sul dataset reale).")
            hp_sintetici = dict(SYNTHETIC_REGRESSOR_REFERENCE_HP)

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
    local_preds = tree_clf.predict(X_test)
    
    # Gestione delle probabilità legata al TASK, non al dataset
    if user_tree_type == "classifier":
        local_proba = tree_clf.predict_proba(X_test)[:, 1]
    else:
        local_proba = None
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

        metriche_test = {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "roc_auc": test_roc_auc
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
        print(f"\n1-2. TUNING SALTATO — iperparametri riusati dal tuning sul dataset reale")
        print(LINEA_SINGOLA)
        print(f"  ▸ Iperparametri applicati: {hp}")

    print(f"\n3. PERFORMANCE REALI SUL TEST SET ({user_tree_type.upper()})")
    print(LINEA_SINGOLA)
    if user_tree_type == "classifier":
        print(f"  ▸ ACCURACY SUL TEST SET   : {metriche_test['accuracy'] * 100:6.2f}%")
        print(f"  ▸ PRECISION SUL TEST SET  : {metriche_test['precision'] * 100:6.2f}%")
        print(f"  ▸ RECALL SUL TEST SET     : {metriche_test['recall'] * 100:6.2f}%")
        print(f"  ▸ F1-SCORE SUL TEST SET   : {metriche_test['f1'] * 100:6.2f}%")
        print(f"  ▸ ROC-AUC SUL TEST SET    : {metriche_test['roc_auc']:6.4f}")
        print("\n  Matrice di Confusione sul Test Set:")
        for riga in cm:
            print(" " * 6 + " ".join(f"[{val:4d}]" for val in riga))
    else:
        print(f"  ▸ MSE SUL TEST SET   : {metriche_test['mse']:.4f}")
        print(f"  ▸ RMSE SUL TEST SET  : {metriche_test['rmse']:.4f}")
        print(f"  ▸ MAE SUL TEST SET   : {metriche_test['mae']:.4f}")
        print(f"  ▸ R² SUL TEST SET    : {metriche_test['r2']:.4f}")

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
    print(f"  • Volume dati: train={X_train.shape}  test={X_test.shape}")

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