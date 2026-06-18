from src.dataset.dataset_dao import AwsS3DAO, DatasetDAO, LocalFileSystemDAO
from src.shared.config import SystemConfig

class DatasetDAOFactory:
    """Factory per ottenere l'istanza corretta di DatasetDAO."""
    
    @staticmethod
    def get_dao() -> DatasetDAO:
        # Legge l'ambiente direttamente dalla config globale
        cfg = SystemConfig()
        env = cfg.env.strip().lower()
        
        if env == "local":
            return LocalFileSystemDAO()
        elif env == "aws":
            return AwsS3DAO()
        else:
            raise ValueError(f"Ambiente '{cfg.env}' non supportato dalla Factory DAO.")