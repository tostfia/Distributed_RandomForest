import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.shared.utilities.loader.datasetLoader import DatasetLoader


RANDOM_SEED = 123


class SyntheticDataLoader(DatasetLoader):
    """
    Generatore di dataset sintetico tramite sklearn.make_classification.

    Usato per stress test e valutazione della scalabilità.
    Restituisce un DataFrame già coerente con la pipeline:
    - feature numeriche;
    - Label binaria 0/1.
    """

    def __init__(
        self,
        n_samples: int = None,
        n_features: int = None,
        random_seed: int = RANDOM_SEED,
        target_column: str = None,
        n_informative: int = None,
        n_redundant: int = None,
        n_clusters_per_class: int = None,
        flip_y: float = None,
        weight: list = None,
    ):
        config_path = "synthetic/synthetic_config.json"
        config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
            except Exception as e:
                print(f"Errore durante la lettura del file di configurazione: {e}")

        # La proporzione di feature informative è fissa all'80% per garantire un certo grado di complessità.
        self.n_samples = n_samples if n_samples is not None else config.get("n_samples", 500000)
        self.n_features = n_features if n_features is not None else config.get("n_features", 30)
        self.n_informative = n_informative if n_informative is not None else config.get("n_informative", int(self.n_features * 0.35))
        self.n_redundant = n_redundant if n_redundant is not None else config.get("n_redundant", 5)
        self.n_clusters_per_class = n_clusters_per_class if n_clusters_per_class is not None else config.get("n_clusters_per_class", 2)
        self.flip_y = flip_y if flip_y is not None else config.get("flip_y", 0.01)
        self.weight = weight if weight is not None else config.get("weight", [0.9, 0.1])
        self.random_seed = random_seed
        self.target_column = target_column if target_column is not None else config.get("target_column", "Label")
        self.filename = filename if (filename := config.get("filename")) is not None else "synthetic_dataset.csv"

        self._validate_parameters()

    #Genera il dataset sintetico e lo restituisce come DataFrame.
    def load(self) -> pd.DataFrame:
        print(
            f"Generazione dataset sintetico "
            f"({self.n_samples} campioni, {self.n_features} feature)..."
        )

        #Invocazione del motore di generazione di sklearn con i parametri specificati
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

        #Mappatura in un DataFrame standardizzato
        feature_columns = [
            f"Feature_{i}"
            for i in range(self.n_features)
        ]

        df = pd.DataFrame(X, columns=feature_columns)
        df[self.target_column] = y.astype(np.int8)

        unique, counts = np.unique(y, return_counts=True)

        print("\nDistribuzione classi nel dataset sintetico:")
        for cls, count in zip(unique, counts):
            print(
                f" • Classe {cls}: {count} campioni "
                f"({count / self.n_samples * 100:.2f}%)"
            )

        print("\n[OK] Dataset sintetico generato.")
        print(f" • Numero di righe:   {df.shape[0]}")
        print(f" • Numero di colonne: {df.shape[1]}")
        output_dir = "synthetic/"
        os.makedirs(output_dir, exist_ok=True)
        final_path = os.path.join(output_dir, self.filename)
        df.to_csv(final_path, index=False)
        print(f" • Dataset salvato in: {final_path}")


        return df

    def _validate_parameters(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples deve essere maggiore di 0.")

        if self.n_features <= 0:
            raise ValueError("n_features deve essere maggiore di 0.")

        if self.n_informative <= 0:
            raise ValueError("n_informative deve essere maggiore di 0.")

        if self.n_informative + self.n_redundant > self.n_features:
            raise ValueError(
                "n_informative + n_redundant non può superare n_features."
            )

        if not isinstance(self.random_seed, int):
            raise TypeError("random_seed deve essere un intero.")