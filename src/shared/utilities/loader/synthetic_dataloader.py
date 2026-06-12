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
        n_samples: int = 100000,
        n_features: int = 20,
        random_seed: int = RANDOM_SEED,
        target_column: str = "target"
    ):
        # La proporzione di feature informative è fissa all'80% per garantire un certo grado di complessità.
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = int(n_features * 0.8)
        self.random_seed = random_seed
        self.target_column = target_column

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
            n_redundant=2,
            n_clusters_per_class=2,
            flip_y=0.01,
            weights=[0.9, 0.1],
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

        return df

    def _validate_parameters(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples deve essere maggiore di 0.")

        if self.n_features <= 0:
            raise ValueError("n_features deve essere maggiore di 0.")

        if self.n_informative <= 0:
            raise ValueError("n_informative deve essere maggiore di 0.")

        if self.n_informative + 2 > self.n_features:
            raise ValueError(
                "n_informative + n_redundant non può superare n_features."
            )

        if not isinstance(self.random_seed, int):
            raise TypeError("random_seed deve essere un intero.")