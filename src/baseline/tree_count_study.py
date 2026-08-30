"""
Studio (via curva OOB a warm_start) del numero di alberi necessari per il
RandomForestRegressor sul dataset sintetico di stress test, agli
iperparametri FISSI già dichiarati in SYNTHETIC_REGRESSOR_REFERENCE_HP di
run_baseline.py (max_depth, min_samples_split, max_features, criterion,
max_samples — bootstrap=True implicito, richiesto dalla stima OOB).

Questa classe NON fa alcuna ricerca di iperparametri — quella parte è stata
deliberatamente rimossa dal progetto. Risponde a una sola domanda: "quanti
alberi servono perché la curva OOB si stabilizzi?", non "quale combinazione
di iperparametri è la migliore".
  
METODOLOGIA — stessa già validata per la classificazione e per la versione
precedente (rimossa) della regressione:
  - Curva costruita con warm_start=True: UNA foresta che cresce
    incrementalmente, non fit indipendenti ad ogni punto della griglia
    (esempio ufficiale scikit-learn "OOB Errors for Random Forests").
  - Correzione del bias di copertura OOB: sklearn riempie con 0.0 le righe
    mai out-of-bag per nessun albero; per la regressione una predizione
    esattamente 0.0 è indistinguibile da un vero 0.0, quindi non basta un
    controllo "somma == 0" — si ricostruisce la copertura dagli indici di
    bootstrap (sklearn.ensemble._forest._generate_sample_indices) e R²/MSE
    sono calcolati SOLO sulle righe realmente coperte da almeno un albero.
  - NESSUN algoritmo di knee-detection automatico (Kneedle rimosso
    deliberatamente, vedi discussione): il criterio è la lettura VISIVA del
    grafico prodotto, più il vincolo "n_estimators scelto >= punto di
    copertura OOB 100%" — sotto quella soglia la stima OOB è matematicamente
    incompleta, a prescindere da come appare la curva.

USO:
    from tree_count_study import RegressorTreeCountStudy

    # IDENTICI a SYNTHETIC_REGRESSOR_REFERENCE_HP in run_baseline.py —
    # aggiorna qui se li cambi anche lì.
    FIXED_HP = dict(
        max_depth=None, min_samples_split=2,
        max_features=0.2, criterion="squared_error", max_samples=1.0,
    )

    study = RegressorTreeCountStudy(fixed_hp=FIXED_HP, min_estimators=5,
                                     max_estimators=350, step=10)

    study.run(label="500k", n_samples=500_000, n_features=150,
              n_informative_reg=15, noise=50.0)
    study.run(label="800k", n_samples=800_000, n_features=250,
              n_informative_reg=12, noise=80.0)

Nota tempi: con dataset di questa scala (500k-800k righe, 150-250 feature,
griglia fino a 350 alberi) ogni run può richiedere diversi minuti. Usa
'diagnostic_sample_size' per un giro di prova veloce prima del numero
definitivo (vedi parametro in run()).
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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
          "verrà saltata, verrà mostrata solo la curva 'naive' (pessimisticamente "
          "distorta ai valori bassi di n_estimators).")


class RegressorTreeCountStudy:
    """
    Costruisce e salva la curva OOB (R²/MSE, naive e corretta) al crescere
    di n_estimators, per una o più "ricette" di dataset sintetico, tenendo
    fissi gli altri iperparametri della foresta (passati una sola volta al
    costruttore). Non decide un valore al posto tuo: produce grafico e
    tabella per ogni ricetta — la scelta finale resta la lettura visiva di
    dove la curva si appiattisce.
    """

    def __init__(self, fixed_hp: dict, random_seed: int = 123,
                 min_estimators: int = 5, max_estimators: int = 150,
                 step: int = 10, output_dir: str = "."):
        self.fixed_hp = fixed_hp
        self.random_seed = random_seed
        self.min_estimators = min_estimators
        self.max_estimators = max_estimators
        self.step = step
        self.output_dir = output_dir
        self.grid = list(range(min_estimators, max_estimators + 1, step))
        # label -> risultati completi di quella ricetta (per confronti
        # successivi, es. tra 500k e 800k, senza dover rileggere i print).
        self.results_: dict = {}

    def _generate_dataset(self, n_samples, n_features, n_informative_reg, noise,
                           diagnostic_sample_size):
        print(f"  Generazione dataset sintetico (n_samples={n_samples}, "
              f"n_features={n_features}, n_informative_reg={n_informative_reg}, "
              f"noise={noise}, seed={self.random_seed})...")
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features,
            n_informative=n_informative_reg, noise=noise,
            random_state=self.random_seed,
        )
        if diagnostic_sample_size and diagnostic_sample_size < n_samples:
            print(f"  Sottocampionamento diagnostico: {n_samples} -> "
                  f"{diagnostic_sample_size} righe (seed={self.random_seed}). Solo "
                  f"per questa diagnostica — il run reale di run_baseline.py userà "
                  f"il dataset completo.")
            rng = np.random.RandomState(self.random_seed)
            idx = rng.choice(n_samples, size=diagnostic_sample_size, replace=False)
            X, y = X[idx], y[idx]
        return X, y

    def run(self, label: str, n_samples: int, n_features: int,
            n_informative_reg: int, noise: float,
            diagnostic_sample_size: int = None):
        """
        Esegue lo studio per UNA ricetta di dataset e ritorna la lista di
        risultati per checkpoint (anche salvata in self.results_[label]).
        """
        print("\n" + "=" * 84)
        print(f"  STUDIO n_estimators — RICETTA '{label}'")
        print(f"  Iperparametri fissi: {self.fixed_hp}")
        print("=" * 84)

        X, y = self._generate_dataset(n_samples, n_features, n_informative_reg,
                                       noise, diagnostic_sample_size)
        n_rows_used = X.shape[0]

        rf = RandomForestRegressor(
            warm_start=True, oob_score=True, bootstrap=True,
            random_state=self.random_seed, n_jobs=-1,
            **self.fixed_hp,
        )

        rows = []
        start_total = time.perf_counter()
        for i, n in enumerate(self.grid, start=1):
            print(f"  [{i}/{len(self.grid)}] fit warm_start fino a n_estimators={n} ...")
            step_start = time.perf_counter()
            rf.n_estimators = n
            rf.fit(X, y)
            step_elapsed = time.perf_counter() - step_start
            cum_elapsed = time.perf_counter() - start_total

            if _COVERAGE_CHECK_AVAILABLE:
                covered = np.zeros(n_rows_used, dtype=bool)
                for tree in rf.estimators_:
                    in_bag = _generate_sample_indices(tree.random_state, n_rows_used, n_rows_used)
                    in_bag_mask = np.zeros(n_rows_used, dtype=bool)
                    in_bag_mask[in_bag] = True
                    covered |= ~in_bag_mask
                coverage_pct = covered.mean() * 100
                y_valid = y[covered]
                pred_valid = rf.oob_prediction_[covered]
            else:
                coverage_pct = float("nan")
                y_valid = y
                pred_valid = rf.oob_prediction_

            r2_naive = r2_score(y, rf.oob_prediction_)
            r2_corr = r2_score(y_valid, pred_valid) if len(y_valid) else float("nan")
            mse_corr = mean_squared_error(y_valid, pred_valid) if len(y_valid) else float("nan")

            rows.append(dict(n=n, coverage_pct=coverage_pct, r2_naive=r2_naive,
                              r2_corr=r2_corr, mse_corr=mse_corr, t_cum=cum_elapsed))
            print(f"  [{i}/{len(self.grid)}] completato in {step_elapsed:.2f}s "
                  f"(cumulato: {cum_elapsed:.2f}s)")

        self._print_table(rows, label)
        self._plot(rows, label)
        self.results_[label] = dict(
            rows=rows, n_rows_used=n_rows_used, n_features=n_features,
            noise=noise, n_informative_reg=n_informative_reg,
        )
        return rows

    def _print_table(self, rows, label):
        print(f"\n  n_est   | copert.%  | R² naive   | R² corr.   | MSE corr.     | Δ R² corr. | t cum(s)")
        print("  " + "-" * 96)
        prev_r2 = None
        first_full_cov = None
        for r in rows:
            if first_full_cov is None and r["coverage_pct"] >= 99.999:
                first_full_cov = r["n"]
            delta = "" if prev_r2 is None else f"{r['r2_corr'] - prev_r2:+.5f}"
            print(f"  {r['n']:<7} | {r['coverage_pct']:<9.3f} | {r['r2_naive']:<10.5f} | "
                  f"{r['r2_corr']:<10.5f} | {r['mse_corr']:<13.4f} | {delta:<10} | {r['t_cum']:8.2f}")
            prev_r2 = r["r2_corr"]

        if first_full_cov is not None:
            print(f"\n  Copertura OOB 100% raggiunta per la prima volta a n_estimators={first_full_cov}.")
            print("  Sotto quella soglia la stima OOB è matematicamente incompleta:")
            print("  qualunque valore tu scelga, deve essere >= a questo numero.")
        out_path = f"oob_curve_regression_{label}.png"
        print(f"\n  Grafico salvato in: {out_path}")
        print("  Leggi a occhio dove la curva R² CORRETTA (blu, pannello superiore) si")
        print("  appiattisce — unico criterio, nessun algoritmo automatico la sostituisce.")

    def _plot(self, rows, label):
        ns = [r["n"] for r in rows]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), dpi=150, sharex=True)

        ax1.plot(ns, [r["r2_naive"] for r in rows], marker='o', markersize=3,
                 color='#94a3b8', linewidth=1.5, linestyle='--', label='R² OOB naive')
        ax1.plot(ns, [r["r2_corr"] for r in rows], marker='o', markersize=3,
                 color='#2563eb', linewidth=2, label='R² OOB corretto')
        ax1.set_ylabel("R²")
        ax1.set_title(f"Curva OOB warm_start — ricetta '{label}'")
        ax1.yaxis.set_major_locator(MultipleLocator(0.02))
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right')

        ax2.plot(ns, [r["coverage_pct"] for r in rows], marker='o', markersize=3,
                 color='#16a34a', linewidth=2)
        ax2.axhline(y=100, color='gray', linestyle=':', linewidth=1)
        ax2.set_ylabel("Copertura OOB (%)")
        ax2.set_xlabel("n_estimators")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = os.path.join(self.output_dir, f"oob_curve_regression_{label}.png")
        fig.savefig(out_path)
        plt.close(fig)

    def print_comparison(self):
        """
        Confronto rapido tra tutte le ricette già eseguite: n_estimators a
        cui la copertura OOB raggiunge il 100%, e R² corretto al tetto della
        griglia — utile per vedere a colpo d'occhio se le due dimensioni
        (500k/800k) si comportano in modo simile o no.
        """
        if not self.results_:
            print("Nessuna ricetta eseguita ancora — chiama .run(...) prima.")
            return
        print("\n" + "=" * 78)
        print("  CONFRONTO TRA RICETTE")
        print("=" * 78)
        print(f"  {'Ricetta':<10} | {'Righe usate':<12} | {'Copert.100% a n=':<18} | {'R² corr. al tetto griglia'}")
        print("  " + "-" * 70)
        for label, res in self.results_.items():
            rows = res["rows"]
            first_full = next((r["n"] for r in rows if r["coverage_pct"] >= 99.999), None)
            last_r2 = rows[-1]["r2_corr"]
            print(f"  {label:<10} | {res['n_rows_used']:<12} | {str(first_full):<18} | {last_r2:.5f}")


if __name__ == "__main__":
    # IDENTICI a SYNTHETIC_REGRESSOR_REFERENCE_HP in run_baseline.py —
    # aggiorna qui se li cambi anche lì (n_estimators NON va qui: è proprio
    # quello che questa classe deve aiutarti a scegliere).
    FIXED_HP = dict(
        max_depth=None,
        min_samples_split=2,
        max_features=0.2,
        criterion="squared_error",
        max_samples=1.0,
    )

    study = RegressorTreeCountStudy(
        fixed_hp=FIXED_HP,
        min_estimators=5,
        max_estimators=150,
        step=10,
    )


    study.run(
        label="500k",
        n_samples=500_000, n_features=150, n_informative_reg=15, noise=50.0,
        diagnostic_sample_size=None,
    )
    

    study.print_comparison()