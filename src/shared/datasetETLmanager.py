import os
from typing import Optional

import pandas as pd

from shared.utilities.loader import DatasetLoaderStrategy
from shared.utilities.preprocessing import CICIDSPreprocessor


class DatasetETLManager:
    """
    Coordina la pipeline ETL del dataset.

    Extract:
        usa un DatasetLoaderStrategy.

    Transform:
        applica opzionalmente un preprocessor.

    Load:
        salva il DataFrame risultante su path locale o S3.

    Questa classe è il punto di aggancio con l'Orchestratore.
    """

    def __init__(
        self,
        loader: DatasetLoaderStrategy,
        preprocessor: Optional[CICIDSPreprocessor] = None,
    ):
        self.loader = loader
        self.preprocessor = preprocessor

    def run(self, output_url: str) -> str:
        """
        Esegue Extract, Transform e Load.

        Parameters
        ----------
        output_url:
            Path locale o S3 dove salvare il dataset pulito.

        Returns
        -------
        str
            Lo stesso output_url, da passare poi ai Worker.
        """

        df = self.run_to_dataframe()
        self._save_dataframe(df, output_url)

        return output_url

    def run_to_dataframe(self) -> pd.DataFrame:
        """
        Esegue Extract e Transform, restituendo il DataFrame.

        Utile per test locali o notebook.
        """

        df = self.loader.load()

        if self.preprocessor is not None:
            df = self.preprocessor.process(df)

        return df

    @staticmethod
    def _save_dataframe(
        df: pd.DataFrame,
        output_url: str,
    ) -> None:
        """
        Salva un DataFrame su file locale o S3.
        """

        print(f"Salvataggio dataset su: {output_url}")

        storage_options = None

        if output_url.startswith("s3://"):
            storage_options = {"anon": False}
        else:
            output_dir = os.path.dirname(output_url)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

        try:
            df.to_csv(
                output_url,
                index=False,
                storage_options=storage_options,
            )
        except Exception as exc:
            raise IOError(
                f"Errore nel salvataggio del dataset su "
                f"'{output_url}': {exc}"
            )

        print("[OK] Dataset salvato correttamente.")