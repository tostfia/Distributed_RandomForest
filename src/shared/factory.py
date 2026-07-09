from src.shared.mock_aws.interfaces import SQSQueueInterface, StateManagerInterface
from src.dataset.dataset_dao import DatasetDAO, LocalFileSystemDAO, AwsS3DAO
from src.shared.mock_aws.sqs import sqs_queue
from src.shared.mock_aws.statemanager import state_manager

def get_aws_services(environment: str) -> tuple[SQSQueueInterface, StateManagerInterface]:
    """
    Factory polimorfa per il disaccoppiamento dell'infrastruttura cloud.
    In base all'ambiente richiesto ('aws' o 'local'), istanzia e restituisce
    i componenti corretti (Mock basati su file o Client reali basati su boto3).
    """
    env = environment.strip().lower()
    
    if env == "aws":
        try:
            print("[FACTORY] Inizializzazione dei servizi AWS reali (Boto3)...")
            
            # Caricamento dinamico dei client reali (da implementare nel tuo pacchetto aws reale)
            # Esempio concettuale di quello che restituirai in produzione:
            # from src.shared.aws_real.sqs import AwsSQSQueue
            # from src.shared.aws_real.statemanager import AwsStateManager
            # return AwsSQSQueue(), AwsStateManager()
            
            # ATTENZIONE: Se non hai ancora scritto le classi AWS reali, 
            # usiamo un fallback temporaneo sui mock invece di fare sys.exit(1)
            raise ImportError("Classi AWS reali non ancora collegate.")
            
        except ImportError as e:
            print(f"\n[FACTORY] [FALLBACK] {e}")
            print("[FACTORY] Deviazione automatica sui Mock persistenti del File System...\n")
            return sqs_queue, state_manager
    else:
        # Ambiente locale: usiamo direttamente le istanze dei mock JSON persistenti
        print("[FACTORY] Modalità LOCAL attiva: caricamento dei Mock basati su File JSON.")
        return sqs_queue, state_manager


class DatasetDAOFactory:
    """Factory per ottenere l'istanza corretta di DatasetDAO in base al file .env."""
    
    @staticmethod
    def get_dao(environment: str) -> DatasetDAO:
        env = environment.strip().lower()
        
        if env == "local":
            print("[DAO-FACTORY] Istanzio LocalFileSystemDAO per l'I/O locale.")
            return LocalFileSystemDAO()
        elif env == "aws":
            print("[DAO-FACTORY] Istanzio AwsS3DAO per l'I/O su Cloud Storage Amazon S3.")
            return AwsS3DAO()
        else:
            raise ValueError(f"Ambiente '{environment}' non supportato dalla Factory DAO.")