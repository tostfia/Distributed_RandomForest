import pandas as pd

from shared.utilities.loader import DatasetLoaderStrategy


class CleanCSVDataLoader(DatasetLoaderStrategy):
    """
    Loader minimale per dataset già campionato e preprocessato.

    Usato dai Worker nella versione base.

    Responsabilità:
    - leggere un CSV già pulito da locale o S3;
    - restituire un DataFrame pronto per il training.

    """

    def __init__(
        self,
        dataset_url: str,
        s3_anon: bool = False,
    ):
        self.dataset_url = dataset_url
        self.s3_anon = s3_anon

    #Load del dataset in RAM
    def load(self) -> pd.DataFrame:
        storage_options = None

        if self.dataset_url.startswith("s3://"):
            storage_options = {"anon": self.s3_anon}

        try:
            df = pd.read_csv(
                self.dataset_url,
                storage_options=storage_options,
            )
        except Exception as exc:
            raise IOError(
                f"Errore nella lettura del dataset pulito "
                f"'{self.dataset_url}': {exc}"
            )

        print("[OK] Dataset pulito caricato.")
        print(f" • Righe:   {df.shape[0]}")
        print(f" • Colonne: {df.shape[1]}")

        return df