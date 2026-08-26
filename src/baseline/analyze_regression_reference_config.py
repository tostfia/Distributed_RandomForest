"""
Diagnostica per la config di riferimento della regressione sintetica
(SYNTHETIC_REGRESSOR_REFERENCE_HP in run_baseline.py).

A differenza dell'analisi sulla soglia di correlazione, qui il dataset è
INTERAMENTE sintetico (sklearn.make_regression): nessun file esterno
necessario, lo script gira standalone e riproduce esattamente la ricetta
usata da SyntheticDataLoader per il task di regressione.

Produce due evidenze:

  1. n_estimators=40 — curva dell'errore OOB (R² e MSE) al crescere del
     numero di alberi, per verificare a che punto si stabilizza e se 40 è
     un compromesso ragionevole tra accuratezza e tempo di training.

  2. noise=10.0 — rapporto segnale/rumore: genera lo stesso dataset con
     noise=0 (segnale "pulito") per stimare la scala naturale del target,
     poi confronta con la deviazione standard del rumore iniettato (10.0)
     per dare un numero interpretabile (es. "il rumore è pari a circa X%
     della variabilità naturale del target").

Uso:
    python -m src.baseline.analyze_regression_reference_config
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# Stessi valori di run_baseline.py / SyntheticDataLoader per la regressione,
# TRANNE n_samples: qui usiamo un sottocampione (30.000 invece di 300.000)
# solo per rendere la diagnostica veloce da eseguire. La FORMA della curva
# OOB (dove si stabilizza il delta R² tra un n_estimators e il successivo)
# è governata dal numero di alberi, non dal numero di campioni — è una
# proprietà robusta a questa riduzione. L'esperimento finale nella baseline
# resta sui 300.000 campioni completi.
RANDOM_SEED = 123
N_SAMPLES = 30000
N_FEATURES = 30
N_INFORMATIVE_REG = int(N_FEATURES * 0.5)  # 15, come in run_baseline.py
NOISE = 10.0

FIXED_RF_KWARGS = dict(
    max_depth=None,
    min_samples_split=2,
    max_features=1 / 3,
    criterion="squared_error",
    bootstrap=True,
    max_samples=1.0,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    oob_score=True,
)

N_ESTIMATORS_GRID = [5, 10, 20, 30, 40, 60, 80, 120, 160, 200]

def analyze_n_estimators(X, y):
    print("=" * 70)
    print("  1. STABILIZZAZIONE DELL'ERRORE OOB AL CRESCERE DI n_estimators")
    print("=" * 70)
    print(f"  {'n_estimators':<14} | {'OOB R²':<10} | {'Delta R² vs prec.':<18} | {'Tempo (s)'}")
    print("  " + "-" * 62)

    import time
    prev_r2 = None
    results = []
    for n in N_ESTIMATORS_GRID:
        kwargs = dict(FIXED_RF_KWARGS)
        kwargs["n_estimators"] = n
        rf = RandomForestRegressor(**kwargs)
        start = time.perf_counter()
        rf.fit(X, y)
        elapsed = time.perf_counter() - start
        oob_r2 = rf.oob_score_
        delta = "" if prev_r2 is None else f"{oob_r2 - prev_r2:+.5f}"
        marker = "  <-- config attuale" if n == 80 else ""
        print(f"  {n:<14} | {oob_r2:<10.5f} | {delta:<18} | {elapsed:8.2f}{marker}")
        results.append((n, oob_r2, elapsed))
        prev_r2 = oob_r2

    print("\n  Interpretazione: se il delta R² tra un n_estimators e il successivo è")
    print("  già piccolo (es. <0.001-0.002) attorno a 40, la scelta è difendibile come")
    print("  'punto di stabilizzazione della curva OOB, oltre il quale il guadagno")
    print("  marginale non giustifica il tempo di training aggiuntivo'.")

    grid_vals = [r[0] for r in results]
    r2s = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    ax.plot(grid_vals, r2s, marker='o', color='#2563eb', linewidth=2, label='OOB R²')
    ax.axvline(x=80, color='#16a34a', linestyle='--', linewidth=1.5,
               label='n_estimators=80 (config attuale)')
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("OOB R²")
    ax.set_title("Stabilizzazione dell'errore OOB al crescere di n_estimators\n"
                  "(Random Forest Regressor, dataset sintetico di riferimento)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()
    out_path = "oob_r2_vs_n_estimators_regression.png"
    fig.savefig(out_path)
    print(f"\n  Grafico salvato in: {out_path}")

    return results


def analyze_noise(X_informative_only=None):
    print("\n" + "=" * 70)
    print("  2. RAPPORTO SEGNALE/RUMORE PER noise=10.0")
    print("=" * 70)

    # Genera il "segnale puro" (noise=0) per stimare la scala naturale del target.
    _, y_clean = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE_REG,
        noise=0.0,
        random_state=RANDOM_SEED,
    )
    signal_std = y_clean.std()

    # Con rumore (quello davvero usato in run_baseline.py / SyntheticDataLoader).
    _, y_noisy = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE_REG,
        noise=NOISE,
        random_state=RANDOM_SEED,
    )
    noisy_std = y_noisy.std()

    # In sklearn.make_regression il rumore è additivo gaussiano con
    # deviazione standard = parametro noise, quindi il confronto diretto è
    # signal_std (deviazione standard del segnale "pulito") vs NOISE.
    snr_ratio = signal_std / NOISE
    noise_pct_of_signal = (NOISE / signal_std) * 100

    print(f"  • Deviazione standard del target SENZA rumore (segnale puro): {signal_std:.4f}")
    print(f"  • Deviazione standard del rumore iniettato (noise={NOISE}):    {NOISE:.4f}")
    print(f"  • Deviazione standard del target CON rumore:                   {noisy_std:.4f}")
    print(f"  • Rapporto segnale/rumore (signal_std / noise):                {snr_ratio:.2f}")
    print(f"  • Il rumore corrisponde a circa il {noise_pct_of_signal:.1f}% della deviazione "
          f"standard del segnale puro.")
    print("\n  Interpretazione: un SNR di questo ordine indica un problema 'non banale ma non")
    print("  impossibile' — il rumore è presente e misurabile ma non domina il segnale.")
    print("  Riporta questi numeri in relazione per giustificare noise=10.0 in modo")
    print("  quantitativo invece di lasciarlo come valore isolato.")

    return signal_std, snr_ratio


def main():
    print(f"Generazione dataset sintetico di riferimento "
          f"(n_samples={N_SAMPLES}, n_features={N_FEATURES}, "
          f"n_informative={N_INFORMATIVE_REG}, noise={NOISE})...\n")
    X, y = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE_REG,
        noise=NOISE,
        random_state=RANDOM_SEED,
    )

    analyze_n_estimators(X, y)
    analyze_noise()

    print("\n" + "=" * 70)
    print("  3. n_informative_reg = int(n_features * 0.5)")
    print("=" * 70)
    print(f"  • n_features={N_FEATURES}, n_informative_reg={N_INFORMATIVE_REG} "
          f"({N_INFORMATIVE_REG/N_FEATURES*100:.0f}% delle feature totali).")
    print("  • Scelta dichiarata: metà delle feature è informativa, l'altra metà è")
    print("    rumore puro per costruzione (non correlata al target). Non serve feature")
    print("    selection sul sintetico proprio perché la separazione segnale/rumore è")
    print("    nota e controllata a priori dal generatore, a differenza del reale dove")
    print("    va stimata empiricamente (da qui la permutation importance sul reale).")


if __name__ == "__main__":
    main()