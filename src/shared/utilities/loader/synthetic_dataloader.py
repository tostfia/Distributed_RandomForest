import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from src.shared.utilities.loader.datasetLoader import DatasetLoader

RANDOM_SEED = 123

class SyntheticDataLoader(DatasetLoader):
    """
    Generatore di dataset sintetico tramite sklearn.

    Supporta due task distinti, selezionabili tramite il parametro `task`:
    - "classification" (default): usa make_classification.
    - "regression": usa make_regression.

    Restituisce un DataFrame già coerente con la pipeline:
    - feature numeriche;
    - colonna target (Label binaria 0/1 per la classificazione, valore continuo per la regressione).
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
        n_informative_reg: int = None,
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
        self.filename = filename if (filename := config.get("filename")) is not None else "synthetic_dataset.csv"
        self.output_dir = output_dir if output_dir is not None else config.get("output_dir", "synthetic/")

        if self.task == "classification":
            self.n_informative = n_informative if n_informative is not None else config.get("n_informative", int(self.n_features * 0.35))
            self.n_redundant = n_redundant if n_redundant is not None else config.get("n_redundant", 5)
            self.n_clusters_per_class = n_clusters_per_class if n_clusters_per_class is not None else config.get("n_clusters_per_class", 2)
            self.flip_y = flip_y if flip_y is not None else config.get("flip_y", 0.01)
            self.weight = weight if weight is not None else config.get("weight", [0.9, 0.1])
            self.target_column = target_column if target_column is not None else config.get("target_column", "Label")
        else:  # regression
            self.n_informative_reg = n_informative_reg if n_informative_reg is not None else config.get("n_informative_reg", int(self.n_features * 0.5))
            self.noise = noise if noise is not None else config.get("noise", 10.0)
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

        else:  # regression
            X, y = make_regression(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_informative=self.n_informative_reg,
                noise=self.noise,
                random_state=self.random_seed,
            )

            df = pd.DataFrame(X, columns=feature_columns)
            df[self.target_column] = y.astype(np.float64)

            print("\nStatistiche del target sintetico (regressione):")
            print(
                f" • Media: {y.mean():.4f}  •  Std: {y.std():.4f}  "
                f"•  Min: {y.min():.4f}  •  Max: {y.max():.4f}"
            )

        print("\n[OK] Dataset sintetico generato.")
        print(f" • Numero di righe:   {df.shape[0]}")
        print(f" • Numero di colonne: {df.shape[1]}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        final_path = os.path.join(self.output_dir, self.filename)
        df.to_csv(final_path, index=False)
        print(f" • Dataset salvato in: {final_path}")


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
            if self.n_informative_reg <= 0 or self.n_informative_reg > self.n_features:
                raise ValueError(
                    "n_informative_reg deve essere maggiore di 0 e non superare n_features."
                )