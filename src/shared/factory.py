from src.shared.mock_aws.sqs.sqs_aws import AwsSQSQueue
from src.shared.mock_aws.interfaces import SQSQueueInterface, StateManagerInterface
from src.shared.mock_aws.sqs.sqs_gateway import LambdaGatewaySQSQueue
from src.shared.mock_aws.statemanager.awsstatemanager import AwsStateManager
from src.dataset.dataset_dao import DatasetDAO, LocalFileSystemDAO, AwsS3DAO
from src.shared.mock_aws.sqs.sqs import MockSQSQueue
from src.shared.mock_aws.statemanager.statemanager import MockStateManager

def get_aws_services(environment: str, role: str = "worker") -> tuple[SQSQueueInterface, StateManagerInterface]:
    """
    Factory polimorfa per il disaccoppiamento dell'infrastruttura cloud.
    In base all'ambiente richiesto ('aws' o 'local'), istanzia e restituisce
    i componenti corretti (Mock basati su file o Client reali basati su boto3).
    Parametro 'role':
      - 'client'  -> usato dal client CLI (main.py), che deve SOLO inviare job.
                     In ambiente 'aws' passa da API Gateway -> Lambda -> SQS.
      - 'worker'  -> (default) usato da orchestratori/worker, che devono anche
                     leggere e cancellare messaggi. In ambiente 'aws' usa boto3
                     diretto su SQS (AwsSQSQueue), come prima di questa modifica.
    """
    env = environment.strip().lower()
    
    if env == "aws":
        try:
            if role == "client":
                print("[FACTORY] Inizializzazione client AWS (API Gateway -> Lambda -> SQS)...")
                queue = LambdaGatewaySQSQueue()
            else:
                print("[FACTORY] Inizializzazione worker/orchestratore AWS (Boto3 diretto su SQS)...")
                queue = AwsSQSQueue()
            return queue, AwsStateManager()
        except Exception as e:
            print(f"\n[FACTORY] [FALLBACK] {e}")
            print("[FACTORY] Deviazione automatica sui Mock persistenti del File System...\n")
            return MockSQSQueue(), MockStateManager()
    else:
        # Ambiente locale: usiamo direttamente le istanze dei mock JSON persistenti
        print("[FACTORY] Modalità LOCAL attiva: caricamento dei Mock basati su File JSON.")
        return MockSQSQueue(), MockStateManager()


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
        
