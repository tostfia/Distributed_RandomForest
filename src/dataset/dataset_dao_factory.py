from src.dataset.dataset_dao import AwsS3DAO, DatasetDAO, LocalFileSystemDAO


class DatasetDAOFactory:
    """Factory per ottenere l'istanza corretta di DatasetDAO."""
    
    @staticmethod
    def get_dao(environment: str) -> DatasetDAO:
        env = environment.strip().lower()
        
        if env == "local":
            return LocalFileSystemDAO()
        elif env == "aws":
            return AwsS3DAO()
        else:
            raise ValueError(f"Ambiente '{environment}' non supportato dalla Factory DAO.")