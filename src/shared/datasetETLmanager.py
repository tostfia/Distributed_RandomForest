import os
from typing import Optional
import pandas as pd

from src.shared.config import SystemConfig
from src.dataset.dataset_dao_factory import DatasetDAOFactory
from src.shared.utilities.loader.datasetLoader import DatasetLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor


class DatasetETLManager:
    """
    Coordina la pipeline ETL del dataset rispettando l'ambiente (.env).

    Extract:
        Usa un DatasetLoader (Grezzo, Sintetico, ecc.).
    Transform:
        Applica opzionalmente il CICIDSPreprocessor.
    Load:
        Salva il DataFrame risultante tramite il DAO corretto (Locale o S3).
    """

    def __init__(
        self,
        loader: DatasetLoader,
        preprocessor: Optional[CICIDSPreprocessor] = None,
    ):
        self.loader = loader
        self.preprocessor = preprocessor
        # Inizializziamo la configurazione di sistema legata all'env
        self.cfg = SystemConfig()

    def run(self, output_url: str) -> str:
        """
        Esegue Extract, Transform e Load.
        """
        df = self.run_to_dataframe()
        self._save_dataframe(df, output_url)
        return output_url

    def run_to_dataframe(self) -> pd.DataFrame:
        """
        Esegue Extract e Transform, restituendo il DataFrame.
        """
        df = self.loader.load()

        if self.preprocessor is not None:
            df = self.preprocessor.process(df)

        return df

    def _save_dataframe(
        self,
        df: pd.DataFrame,
        output_url: str,
    ) -> None:
        """
        Salva un DataFrame delegando al DAO corretto in base all'ambiente.
        """
        print(f"[ETL-Manager] Salvataggio dataset richiesto su: {output_url}")
        print(f"[ETL-Manager] Infrastruttura rilevata dal .env: {self.cfg.env.upper()}")

        try:
            # Sfruttiamo la Factory per ottenere il DAO corretto (LocalFileSystemDAO o AwsS3DAO)
            dao = DatasetDAOFactory.get_dao(self.cfg.env)
            
            # Utilizziamo il metodo del DAO per persistere il file.
            # Nota: Assicurati che nel tuo DAO ci sia un metodo per salvare (es: save_dataset o to_csv)
            dao.save_dataset(df, output_url)
            
        except Exception as exc:
            raise IOError(
                f"Errore nell'operazione di LOAD dell'ETL su "
                f"'{output_url}': {exc}"
            )

        print("[OK] Fase di LOAD completata con successo tramite DAO.")