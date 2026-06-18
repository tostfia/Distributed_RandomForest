import pandas as pd
from shared.utilities.loader.datasetLoader import DatasetLoader
from src.dataset.dataset_dao_factory import DatasetDAOFactory
from src.shared.config import SystemConfig

class CleanCSVDataLoader(DatasetLoader):
    """
    Loader minimale per dataset già campionato e preprocessato.
    Sfrutta l'architettura DAO globale per rispettare l'ambiente del file .env.
    """

    def __init__(self, dataset_url: str):
        self.dataset_url = dataset_url
        self.cfg = SystemConfig()

    def load(self) -> pd.DataFrame:
        print(f"[Worker-Loader] Richiesta caricamento dataset tramite DAO. Ambiente: {self.cfg.env.upper()}")
        
        try:
            # Sfrutta la Factory che legge il file .env (Locale o AWS)
            # e ottiene il DAO corretto in modo polimorfo
            dao = DatasetDAOFactory.get_dao(self.cfg.env)
            df = dao.load_dataset(self.dataset_url)
            
        except Exception as exc:
            raise IOError(
                f"Errore nel caricamento del dataset tramite DAO "
                f"'{self.dataset_url}': {exc}"
            )

        print("[OK] Dataset pulito caricato correttamente dal DAO.")
        print(f" • Righe:   {df.shape[0]}")
        print(f" • Colonne: {df.shape[1]}")

        return df