"""
Diagnostica per la config di riferimento della regressione sintetica
(SYNTHETIC_REGRESSOR_REFERENCE_HP in run_baseline.py).

PASSO 2 di 2 nel processo di scelta della configurazione di riferimento:
esegui PRIMA analyze_regression_hyperparameter_search.py (Passo 1), che
studia max_depth/min_samples_split/criterion/max_samples con n_estimators
fissato a un valore generoso e scrive il vincitore in
outputs_baseline/config_synthetic_regressor_search.json. Questo script
(Passo 2) legge quel manifesto e studia SOLO n_estimators, verificando se
un valore più basso di quello usato nel Passo 1 raggiunge già prestazioni
equivalenti. Se il manifesto non esiste ancora, questo script gira comunque
con valori di default neutri, ma il risultato non va usato per aggiornare
run_baseline.py finché il Passo 1 non è stato eseguito.

Il dataset e' interamente sintetico (sklearn.make_regression): nessun file
esterno necessario, lo script gira standalone e riproduce esattamente la
ricetta usata da SyntheticDataLoader per il task di regressione.

METODOLOGIA -- allineata all'esempio ufficiale di scikit-learn:
    "OOB Errors for Random Forests"
    https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
    (The scikit-learn developers, BSD-3-Clause)

    Come nell'esempio ufficiale, la curva OOB e' costruita con
    warm_start=True: UNA SOLA foresta cresce incrementalmente (si
    aggiungono alberi via via), non tante foreste indipendenti create da
    zero per ogni n_estimators. E' sia piu' efficiente sia piu' corretto:
    con warm_start la foresta a 40 alberi CONTIENE esattamente i 20 alberi
    della foresta a 20 alberi piu' altri 20 nuovi, quindi la curva descrive
    la vera traiettoria di una foresta che cresce (con fit indipendenti,
    invece, lo schema di seeding interno di sklearn assegna semi diversi
    agli alberi a seconda di quanti stimatori totali vengono richiesti, e
    la curva confronta foreste diverse punto per punto, non la stessa
    foresta osservata a diversi stadi).

    LETTURA: come nell'esempio ufficiale, il criterio per scegliere
    n_estimators e' la lettura VISIVA del grafico -- il professore lo
    dichiara esplicitamente ("the resulting plot allows a practitioner to
    approximate a suitable value of n_estimators at which the error
    stabilizes"). Non viene usato alcun algoritmo di knee-detection
    automatico (rimosso: il Kneedle semplificato usato in una versione
    precedente di questo script tende a segnalare un ginocchio
    sistematicamente troppo basso su curve che saturano rapidamente,
    perche' individua il punto di massima curvatura e non il punto di
    reale stabilita' -- vedi discussione a parte). Come riferimento
    numerico supplementare resta il punto in cui la copertura OOB
    raggiunge il 100% (limite INFERIORE assoluto, non un target).

CORREZIONE IMPORTANTE (bias nell'OOB score di sklearn a n_estimators bassi):
    Quando pochi alberi sono stati addestrati, alcuni campioni di training
    non sono MAI out-of-bag per nessun albero (non e' un caso raro: con
    n_estimators=10 circa l'1% dei campioni non ha ancora una predizione
    OOB valida; con n_estimators=5 circa il 10%; con n_estimators=1 circa
    il 64%, come da teoria del bootstrap: P(mai OOB dopo n alberi) =
    (1-1/e)^n. sklearn riempie questi campioni con una predizione di
    default = 0.0 (verificato empiricamente su questa installazione:
    numero di zeri esatti in oob_prediction_ == numero di campioni "mai
    OOB" calcolato dagli indici di bootstrap). Il problema e' che
    rf.oob_score_ (l'attributo che tutti usano per leggere l'R^2 OOB) INCLUDE
    questi zeri fittizi nel calcolo -- verificato numericamente: coincide
    esattamente con l'R^2 "ingenuo" calcolato su TUTTI i campioni, zeri
    compresi. Questo significa che rf.oob_score_ e' artificialmente
    PESSIMISTA a n_estimators bassi, perche' conta come "errori" campioni
    che semplicemente non sono mai stati valutati.

    Questo script calcola quindi DUE curve:
      • "naive" -- rf.oob_score_ cosi' come lo fornisce sklearn (quello che
        userebbe chiunque prenda l'attributo cosi' com'e', incluso
        l'esempio ufficiale se applicato con un min_estimators troppo
        basso -- il professore parte da min_estimators=15, che gia'
        garantisce una copertura ~99.9% per la classificazione,
        aggirando implicitamente il problema senza dichiararlo);
      • "corretta" -- R^2/MSE calcolati SOLO sui campioni che hanno
        davvero almeno una predizione OOB, usando gli indici di bootstrap
        di ogni albero (sklearn.ensemble._forest._generate_sample_indices,
        API privata ma e' la stessa funzione che sklearn usa internamente
        per costruire i bootstrap sample -- comportamento verificato su
        questa installazione, sklearn 1.8.0; su versioni diverse verificare
        che l'import non sia cambiato).
    Il grafico mostra entrambe, cosi' la lettura visiva si fa sulla curva
    corretta, non su quella distorta.

Produce quattro evidenze:

  1. n_estimators -- curva OOB (naive vs corretta) al crescere del numero
     di alberi via warm_start. Nessun punto "ottimo" automatico: la scelta
     resta la lettura visiva del grafico.

  2. max_features=1/3 -- NON selezionato empiricamente: e' un valore
     dichiarato a priori in run_baseline.py, giustificato per citazione
     (vedi commento "NOTA CITAZIONE" li'). Sweep descrittiva di supporto.

  3. noise=10.0 -- rapporto segnale/rumore calcolato.

  4. n_informative_reg = int(n_features * 0.5) -- nota descrittiva.

Uso:
    python -m src.baseline.analyze_regression_reference_config
"""
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

try:
    from sklearn.ensemble._forest import _generate_sample_indices
    _COVERAGE_CHECK_AVAILABLE = True
except ImportError:
    _COVERAGE_CHECK_AVAILABLE = False
    print("[ATTENZIONE] sklearn.ensemble._forest._generate_sample_indices non "
          "disponibile in questa versione di sklearn: la correzione del bias OOB "
          "verra' saltata, verra' mostrata solo la curva 'naive'. Verificare "
          "manualmente la versione di sklearn installata.")

# ---------------------------------------------------------------------------
# PASSO 2 di 2: questo script assume che PASSO 1
# (analyze_regression_hyperparameter_search.py) sia già stato eseguito.
# Legge il suo manifesto per max_depth/min_samples_split/criterion/
# max_samples, e studia SOLO n_estimators via curva OOB a warm_start.
# Stesso pattern già usato da analyze_classification_n_estimators.py con
# config_real.json (load_tuned_hyperparameters()).
# ---------------------------------------------------------------------------
SEARCH_MANIFEST_PATH = os.path.join("outputs_baseline", "config_synthetic_regressor_search.json")


def load_searched_hyperparameters():
    """
    Ritorna il dizionario 'hyperparameters' scritto da
    analyze_regression_hyperparameter_search.py (Passo 1) se esiste,
    altrimenti None. Non decide nulla: si limita a leggere cosa la ricerca
    ha già scelto, così la curva di n_estimators viene costruita intorno
    alla combinazione realmente vincente, non a un'ipotesi arbitraria.
    """
    if not os.path.exists(SEARCH_MANIFEST_PATH):
        print(f"[ATTENZIONE] '{SEARCH_MANIFEST_PATH}' non trovato: il Passo 1 "
              f"(analyze_regression_hyperparameter_search.py) non è ancora stato "
              f"eseguito. Uso valori di default neutri (max_depth=None, "
              f"min_samples_split=2, criterion='squared_error', max_samples=1.0) "
              f"SOLO per un'esplorazione preliminare — il risultato di questa run "
              f"NON va usato per aggiornare SYNTHETIC_REGRESSOR_REFERENCE_HP finché "
              f"il Passo 1 non è stato eseguito e questo script rilanciato.")
        return None
    with open(SEARCH_MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
    hp = manifest.get("hyperparameters")
    if not hp:
        print(f"[ATTENZIONE] '{SEARCH_MANIFEST_PATH}' trovato ma senza sezione 'hyperparameters'.")
        return None
    print(f"[INFO] Iperparametri dal Passo 1 letti da '{SEARCH_MANIFEST_PATH}': {hp}")
    return hp

# ---------------------------------------------------------------------------
# Config del dataset -- stessi valori di run_baseline.py / SyntheticDataLoader
# per la regressione, TRANNE n_samples: qui usiamo un sottocampione (30.000
# invece di 300.000) solo per rendere la diagnostica veloce da eseguire. La
# FORMA della curva OOB e' governata dal numero di alberi, non dal numero di
# campioni -- e' una proprieta' robusta a questa riduzione. L'esperimento
# finale nella baseline resta sui 300.000 campioni completi.
# ---------------------------------------------------------------------------
RANDOM_SEED = 123
N_SAMPLES = 300000
N_FEATURES = 30
N_INFORMATIVE_REG = int(N_FEATURES * 0.5)  # 15, come in run_baseline.py
NOISE = 10.0

FIXED_MAX_FEATURES = 1 / 3  # da run_baseline.py, giustificato per citazione (non qui)

# Griglia uniforme (stile esempio ufficiale sklearn: min/max/step), ma qui
# partiamo da 5 invece che da 15 apposta per rendere visibile ed esplicito
# il bias OOB a basso n_estimators, invece di aggirarlo implicitamente.
MIN_ESTIMATORS = 5
MAX_ESTIMATORS = 200
STEP_ESTIMATORS = 5
N_ESTIMATORS_GRID = list(range(MIN_ESTIMATORS, MAX_ESTIMATORS + 1, STEP_ESTIMATORS))

MAX_FEATURES_GRID = [1 / 10, 1 / 5, 1 / 3, 1 / 2, 1.0]  # solo descrittivo, vedi punto 2

# n_estimators usato SOLO per lo sweep descrittivo di max_features (punto 2),
# non e' una proposta di valore finale. Aggiornalo a mano dopo aver letto a
# occhio il grafico del punto 1 (oob_curve_warmstart_regression.png).
REFERENCE_N_ESTIMATORS_FOR_SWEEP = 100


def analyze_n_estimators(X, y, searched_hp):
    if searched_hp:
        base_kwargs = dict(
            max_depth=searched_hp.get("max_depth"),
            min_samples_split=int(searched_hp.get("min_samples_split", 2)),
            criterion=searched_hp.get("criterion", "squared_error"),
            max_samples=float(searched_hp.get("max_samples", 1.0)),
        )
    else:
        base_kwargs = dict(
            max_depth=None, min_samples_split=2,
            criterion="squared_error", max_samples=1.0,
        )

    print("=" * 84)
    print("  1. CURVA OOB CON warm_start (metodo: esempio ufficiale scikit-learn)")
    print(f"     seed={RANDOM_SEED}, griglia {MIN_ESTIMATORS}..{MAX_ESTIMATORS} step {STEP_ESTIMATORS}")
    print(f"     altri iperparametri (dal Passo 1): {base_kwargs}")
    print("=" * 84)

    rf = RandomForestRegressor(
        warm_start=True,
        oob_score=True,
        max_features=FIXED_MAX_FEATURES,
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        **base_kwargs,
    )

    covered = np.zeros(N_SAMPLES, dtype=bool)
    prev_n_trees = 0
    cumulative_time = 0.0
    rows = []

    for n in N_ESTIMATORS_GRID:
        rf.set_params(n_estimators=n)
        t0 = time.perf_counter()
        rf.fit(X, y)
        cumulative_time += time.perf_counter() - t0

        if _COVERAGE_CHECK_AVAILABLE:
            for tree in rf.estimators_[prev_n_trees:n]:
                in_bag = _generate_sample_indices(tree.random_state, N_SAMPLES, N_SAMPLES)
                in_bag_mask = np.zeros(N_SAMPLES, dtype=bool)
                in_bag_mask[in_bag] = True
                covered |= ~in_bag_mask
            prev_n_trees = n
            coverage_pct = covered.mean() * 100
            if covered.any():
                r2_corrected = r2_score(y[covered], rf.oob_prediction_[covered])
                mse_corrected = mean_squared_error(y[covered], rf.oob_prediction_[covered])
            else:
                r2_corrected = np.nan
                mse_corrected = np.nan
        else:
            coverage_pct = np.nan
            r2_corrected = np.nan
            mse_corrected = np.nan

        rows.append(dict(
            n=n,
            r2_naive=rf.oob_score_,
            mse_naive=mean_squared_error(y, rf.oob_prediction_),
            r2_corrected=r2_corrected,
            mse_corrected=mse_corrected,
            coverage_pct=coverage_pct,
            cum_time=cumulative_time,
        ))

    print(f"  {'n_est':<7} | {'copertura%':<11} | {'R² naive':<10} | {'R² corretto':<12} | "
          f"{'MSE naive':<12} | {'MSE corretto':<12} | {'t cumulato(s)'}")
    print("  " + "-" * 94)
    for r in rows:
        cov_str = f"{r['coverage_pct']:.2f}" if not np.isnan(r['coverage_pct']) else "n/d"
        r2c_str = f"{r['r2_corrected']:.5f}" if not np.isnan(r['r2_corrected']) else "n/d"
        msec_str = f"{r['mse_corrected']:.2f}" if not np.isnan(r['mse_corrected']) else "n/d"
        print(f"  {r['n']:<7} | {cov_str:<11} | {r['r2_naive']:<10.5f} | {r2c_str:<12} | "
              f"{r['mse_naive']:<12.2f} | {msec_str:<12} | {r['cum_time']:8.2f}")

    if _COVERAGE_CHECK_AVAILABLE:
        # Punto in cui la copertura raggiunge (per la prima volta) il 100%:
        # oltre quel punto naive e corretta devono coincidere esattamente.
        fully_covered = [r for r in rows if r["coverage_pct"] >= 100.0 - 1e-9]
        first_full = fully_covered[0]["n"] if fully_covered else None
        print(f"\n  Copertura OOB 100% raggiunta per la prima volta a n_estimators={first_full}.")
        print("  PRIMA di quel punto, R² naive (sklearn) e R² corretto DIVERGONO: naive e'")
        print("  artificialmente pessimista perche' conta come errori campioni mai valutati")
        print("  OOB (sklearn li riempie con predizione=0.0 invece di escluderli). Verificare")
        print("  nella tabella sopra quanto e' ampio lo scarto ai valori piu' bassi di n.")

    # --- Grafico: naive vs corretta, stile esempio ufficiale ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 11), dpi=150, sharex=True)

    ax1.plot(N_ESTIMATORS_GRID, [r["r2_naive"] for r in rows], marker='o', markersize=3,
             color='#9ca3af', linewidth=1.5, label="R² OOB naive (rf.oob_score_, sklearn)")
    if _COVERAGE_CHECK_AVAILABLE:
        ax1.plot(N_ESTIMATORS_GRID, [r["r2_corrected"] for r in rows], marker='o', markersize=3,
                 color='#2563eb', linewidth=2, label="R² OOB corretto (solo campioni davvero OOB)")
    ax1.set_ylabel("OOB R²")
    ax1.set_title("Curva OOB al crescere di n_estimators (warm_start)\n"
                   "Leggi a occhio dove la curva BLU si appiattisce -- criterio primario")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=8)

    if _COVERAGE_CHECK_AVAILABLE:
        ax2.plot(N_ESTIMATORS_GRID, [r["coverage_pct"] for r in rows], marker='o', markersize=3,
                 color='#7c3aed', linewidth=2)
        ax2.axhline(y=100, color='#9ca3af', linestyle=':', linewidth=1)
        ax2.set_ylabel("Copertura OOB (%)")
        ax2.grid(True, alpha=0.3)

    ax3.plot(N_ESTIMATORS_GRID, [r["cum_time"] for r in rows], marker='o', markersize=3,
             color='#dc2626', linewidth=2)
    ax3.set_xlabel("n_estimators")
    ax3.set_ylabel("Tempo cumulato di training (s)")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = "oob_curve_warmstart_regression.png"
    fig.savefig(out_path)
    print(f"\n  Grafico salvato in: {out_path}")
    print("  Usa questo grafico per la lettura visiva del punto di stabilizzazione:")
    print("  guarda la curva blu (corretta) nel pannello superiore -- dove smette di")
    print("  salire in modo apprezzabile e' il valore da riportare in run_baseline.py.")


def analyze_max_features_descriptive(X, y, reference_n_estimators, searched_hp):
    if searched_hp:
        base_kwargs = dict(
            max_depth=searched_hp.get("max_depth"),
            min_samples_split=int(searched_hp.get("min_samples_split", 2)),
            criterion=searched_hp.get("criterion", "squared_error"),
            max_samples=float(searched_hp.get("max_samples", 1.0)),
        )
    else:
        base_kwargs = dict(
            max_depth=None, min_samples_split=2,
            criterion="squared_error", max_samples=1.0,
        )

    print("\n" + "=" * 84)
    print(f"  2. max_features -- SWEEP DESCRITTIVA (n_estimators={reference_n_estimators})")
    print("=" * 84)
    print("  NOTA: max_features=1/3 in run_baseline.py NON e' scelto da questa sweep, ma")
    print("  per citazione (vedi commento 'NOTA CITAZIONE' in run_baseline.py). Questa")
    print("  tabella serve solo a verificare che 1/3 cada in una zona di plateau.\n")
    print(f"  {'max_features':<14} | {'OOB R² (naive)':<16} | {'Tempo (s)'}")
    print("  " + "-" * 48)

    for mf in MAX_FEATURES_GRID:
        rf = RandomForestRegressor(
            n_estimators=reference_n_estimators, max_features=mf,
            bootstrap=True, n_jobs=-1, random_state=RANDOM_SEED, oob_score=True,
            **base_kwargs,
        )
        start = time.perf_counter()
        rf.fit(X, y)
        elapsed = time.perf_counter() - start
        marker = "  <-- valore usato in run_baseline.py (per citazione)" if mf == 1 / 3 else ""
        print(f"  {mf:<14.3f} | {rf.oob_score_:<16.5f} | {elapsed:8.2f}{marker}")


def analyze_noise():
    print("\n" + "=" * 84)
    print("  3. RAPPORTO SEGNALE/RUMORE PER noise=10.0")
    print("=" * 84)

    _, y_clean = make_regression(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_informative=N_INFORMATIVE_REG, noise=0.0, random_state=RANDOM_SEED,
    )
    signal_std = y_clean.std()

    _, y_noisy = make_regression(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_informative=N_INFORMATIVE_REG, noise=NOISE, random_state=RANDOM_SEED,
    )
    noisy_std = y_noisy.std()

    snr_ratio = signal_std / NOISE
    noise_pct_of_signal = (NOISE / signal_std) * 100

    print(f"  • Deviazione standard del target SENZA rumore (segnale puro): {signal_std:.4f}")
    print(f"  • Deviazione standard del rumore iniettato (noise={NOISE}):    {NOISE:.4f}")
    print(f"  • Deviazione standard del target CON rumore:                   {noisy_std:.4f}")
    print(f"  • Rapporto segnale/rumore (signal_std / noise):                {snr_ratio:.2f}")
    print(f"  • Il rumore corrisponde a circa il {noise_pct_of_signal:.1f}% della deviazione "
          f"standard del segnale puro.")

    return signal_std, snr_ratio


def main():
    print(f"Generazione dataset sintetico di riferimento "
          f"(n_samples={N_SAMPLES}, n_features={N_FEATURES}, "
          f"n_informative={N_INFORMATIVE_REG}, noise={NOISE}, seed={RANDOM_SEED})...\n")
    X, y = make_regression(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_informative=N_INFORMATIVE_REG, noise=NOISE, random_state=RANDOM_SEED,
    )

    searched_hp = load_searched_hyperparameters()

    analyze_n_estimators(X, y, searched_hp)
    # Valore di riferimento per lo sweep descrittivo di max_features: fisso,
    # non piu' derivato da un ginocchio automatico. Se dopo la lettura
    # visiva del grafico punto 1 scegli un n_estimators diverso, aggiorna
    # questa costante e rilancia solo questa parte.
    reference_n = REFERENCE_N_ESTIMATORS_FOR_SWEEP
    analyze_max_features_descriptive(X, y, reference_n, searched_hp)
    analyze_noise()

    print("\n" + "=" * 84)
    print("  4. n_informative_reg = int(n_features * 0.5)")
    print("=" * 84)
    print(f"  • n_features={N_FEATURES}, n_informative_reg={N_INFORMATIVE_REG} "
          f"({N_INFORMATIVE_REG/N_FEATURES*100:.0f}% delle feature totali).")
    print("  • Scelta dichiarata (design del dataset): metà delle feature è informativa,")
    print("    l'altra metà è rumore puro per costruzione.")

    print("\n" + "=" * 84)
    print("  PROSSIMO PASSO")
    print("=" * 84)
    if searched_hp is None:
        print("  ATTENZIONE: questa run ha usato valori di default neutri per max_depth/")
        print("  min_samples_split/criterion/max_samples (Passo 1 non ancora eseguito). Esegui")
        print("  prima analyze_regression_hyperparameter_search.py, poi rilancia questo script")
        print("  prima di usare i numeri qui sopra per aggiornare run_baseline.py.")
    print("  1. Apri oob_curve_warmstart_regression.png e leggi a occhio dove la curva blu")
    print("     (R² OOB corretto) si appiattisce -- questo e' l'unico criterio.")
    print("  2. Verifica che il valore scelto sia >= al n_estimators a cui la copertura OOB")
    print("     raggiunge il 100% (stampato sopra): sotto quella soglia la stima OOB e'")
    print("     matematicamente incompleta, a prescindere da come appare la curva.")
    print("  3. Solo dopo aver deciso il valore finale, aggiorna")
    print("     SYNTHETIC_REGRESSOR_REFERENCE_HP in run_baseline.py (n_estimators E gli")
    print("     altri iperparametri, che ora vengono dal Passo 1, non piu' da valori")
    print("     dichiarati a priori) e il relativo commento, citando: (a) l'esempio")
    print("     ufficiale scikit-learn per il metodo warm_start, (b) la correzione del")
    print("     bias di copertura OOB spiegata in questo script, (c) la ricerca OOB")
    print("     congiunta del Passo 1 per gli altri iperparametri.")


if __name__ == "__main__":
    main()