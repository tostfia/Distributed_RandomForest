import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_friedman1

from src.shared.utilities.loader.datasetLoader import DatasetLoader
from src.shared.config import SystemConfig

RANDOM_SEED = 123

class SyntheticDataLoader(DatasetLoader):
    """
    Generatore di dataset sintetico tramite scikit-learn.

    Usato sia per il task "controllato" di classificazione (richiesto dalla
    traccia per valutare la scalabilità a dimensione nota), sia per quello 
    di regressione. 

    Legge i parametri di generazione (n_samples, n_features, ecc.) dal
    manifesto della baseline (config_synthetic.json) se non passati 
    esplicitamente. In questo modo la stessa identica ricetta di dataset 
    può essere riprodotta sia dalla baseline locale sia dai worker 
    distribuiti (centralizzati o federati, questi ultimi con un offset 
    di seed per worker per ottenere shard diversi ma dallo stesso spazio 
    campionario).

    Restituisce un DataFrame già coerente con la pipeline: feature numeriche
    e una colonna target (binaria 0/1 per classificazione, continua per regressione).

    ---
    DETTAGLI E SCELTE IMPLEMENTATIVE

    Supporta due task distinti:
    - "classification" (default): usa `make_classification`.
    - "regression": usa `make_friedman1`.

    PERCHÉ MAKE_FRIEDMAN1 PER LA REGRESSIONE?
    La traccia del progetto chiede genericamente di usare "i generatori 
    di scikit-learn" (pagina generale, non una funzione specifica). 
    Si è scelto `make_friedman1` (Friedman 1991; Breiman 1996, "Bagging 
    predictors" — stesso lavoro citato nel progetto) invece del classico 
    `make_regression` perché quest'ultimo genera relazioni puramente lineari, 
    dove una Random Forest risulta quasi sprecata ("overkill"). 
    
    Friedman #1 genera invece un problema NON lineare:
    y = 10·sin(π·X0·X1) + 20·(X2−0.5)² + 10·X3 + 5·X4 + noise·N(0,1)
    
    L'interazione (sin(X0·X1)) e il termine quadratico non sono catturabili 
    da un modello lineare, motivando in modo forte e naturale la scelta di un 
    ensemble di alberi. 
    
    Inoltre, per costruzione, le feature informative di Friedman #1 sono 
    SEMPRE esattamente 5 (X0..X4). Tutte le restanti (n_features - 5) sono 
    rumore puro introdotto per testare la robustezza del modello, indipendenti 
    dal target. Questa proprietà ("segnale/rumore noto a priori") giustifica 
    il fatto che sul dataset sintetico non sia necessaria alcuna feature selection.
    """

    def __init__(
        self,
        task: str = "regression",
        n_samples: int = None,
        n_features: int = None,
        random_seed: int = RANDOM_SEED,
        target_column: str = None,
        n_informative: int = None,
        n_redundant: int = None,
        n_clusters_per_class: int = None,
        flip_y: float = None,
        weight: list = None,
        noise: float = None,
        output_dir: str = "synthetic/",
    ):
        
        if task not in ("classification", "regression"):
            raise ValueError(f"Task non supportato: '{task}'. Usare 'classification' o 'regression'.")
        self.task = task

        config_path = "outputs_baseline/config_synthetic.json"
        config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
            except Exception as e:
                print(f"Errore durante la lettura del file di configurazione: {e}")

        self.n_samples = n_samples if n_samples is not None else config.get("n_samples", 300000)
        self.n_features = n_features if n_features is not None else config.get("n_features", 30)
        self.random_seed = random_seed
        self.filename = filename if (filename := config.get("filename")) is not None else f"synthetic_dataset_{self.task}.csv"
        self.output_dir = output_dir if output_dir is not None else config.get("output_dir", "synthetic/")

        if self.task == "classification":
            self.n_informative = n_informative if n_informative is not None else config.get("n_informative", int(self.n_features * 0.35))
            self.n_redundant = n_redundant if n_redundant is not None else config.get("n_redundant", 5)
            self.n_clusters_per_class = n_clusters_per_class if n_clusters_per_class is not None else config.get("n_clusters_per_class", 2)
            self.flip_y = flip_y if flip_y is not None else config.get("flip_y", 0.01)
            self.weight = weight if weight is not None else config.get("weight", [0.9, 0.1])
            self.target_column = target_column if target_column is not None else config.get("target_column", "Label")
        else:  # regression -- make_friedman1: 5 feature informative FISSE (non
            # parametrizzabili), il resto rumore puro. Niente n_informative_reg:
            # non avrebbe senso con questo generatore (vedi docstring classe).
            self.noise = noise if noise is not None else config.get("noise", 0.5)
            self.target_column = target_column if target_column is not None else config.get("target_column", "Target")

        self._validate_parameters()

    #Genera il dataset sintetico e lo restituisce come DataFrame.
    def load(self) -> pd.DataFrame:
        print(
            f"Generazione dataset sintetico "
            f"({self.n_samples} campioni, {self.n_features} feature)..."
        )

        feature_columns = [f"Feature_{i}" for i in range(self.n_features)]

        if self.task == "classification":
            X, y = make_classification(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_informative=self.n_informative,
                n_redundant=self.n_redundant,
                n_clusters_per_class=self.n_clusters_per_class,
                flip_y=self.flip_y,
                weights=self.weight,
                random_state=self.random_seed,
            )

            df = pd.DataFrame(X, columns=feature_columns)
            df[self.target_column] = y.astype(np.int8)

            unique, counts = np.unique(y, return_counts=True)
            print("\nDistribuzione classi nel dataset sintetico:")
            for cls, count in zip(unique, counts):
                print(
                    f" • Classe {cls}: {count} campioni "
                    f"({count / self.n_samples * 100:.2f}%)"
                )

        else:  # regression -- Friedman #1 (Friedman 1991; Breiman 1996)
            X, y = make_friedman1(
                n_samples=self.n_samples,
                n_features=self.n_features,
                noise=self.noise,
                random_state=self.random_seed,
            )

            df = pd.DataFrame(X, columns=feature_columns)
            df[self.target_column] = y.astype(np.float64)

            print("\nStatistiche del target sintetico (regressione, Friedman #1):")
            print(
                f" • Media: {y.mean():.4f}  •  Std: {y.std():.4f}  "
                f"•  Min: {y.min():.4f}  •  Max: {y.max():.4f}"
            )

        print("\n[OK] Dataset sintetico generato.")
        print(f" • Numero di righe:   {df.shape[0]}")
        print(f" • Numero di colonne: {df.shape[1]}")
        
        env = SystemConfig().env.strip().lower()
        if env == "local":
            os.makedirs(self.output_dir, exist_ok=True)
            final_path = os.path.join(self.output_dir, self.filename)
            df.to_csv(final_path, index=False)
            print(f" • Dataset salvato in: {final_path}")
        else:
            print(f" • Salvataggio su disco saltato (ambiente '{env}': il dataset "
                  f"resta solo in memoria, nessun consumatore rilegge questo file).")


        return df

    def _validate_parameters(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples deve essere maggiore di 0.")

        if self.n_features <= 0:
            raise ValueError("n_features deve essere maggiore di 0.")
        
        if not isinstance(self.random_seed, int):
            raise TypeError("random_seed deve essere un intero.")

        if self.task == "classification":
            if self.n_informative <= 0:
                raise ValueError("n_informative deve essere maggiore di 0.")
            if self.n_informative + self.n_redundant > self.n_features:
                raise ValueError(
                    "n_informative + n_redundant non può superare n_features."
                )
        else:
            # make_friedman1 richiede almeno 5 feature (le uniche informative,
            # per costruzione -- vedi docstring classe).
            if self.n_features < 5:
                raise ValueError(
                    "n_features deve essere >= 5 per make_friedman1 (5 feature "
                    "informative fisse, richieste dalla formula di Friedman #1)."
                )