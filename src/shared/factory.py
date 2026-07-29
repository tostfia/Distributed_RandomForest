from src.shared.mock_aws.statemanager.statemanager_gateway import ApiGatewayStateManager
from src.shared.mock_aws.sqs.sqs_aws import AwsSQSQueue
from src.shared.mock_aws.interfaces import SQSQueueInterface, StateManagerInterface
from src.shared.mock_aws.sqs.sqs_gateway import LambdaGatewaySQSQueue
from src.shared.mock_aws.statemanager.awsstatemanager import AwsStateManager
from src.dataset.dataset_dao import DatasetDAO, LocalFileSystemDAO, AwsS3DAO
from src.shared.mock_aws.sqs.sqs_mock import MockSQSQueue
from src.shared.mock_aws.statemanager.localstatemanager import MockStateManager

def get_aws_services(environment: str, role: str = "worker") -> tuple[SQSQueueInterface, StateManagerInterface]:
    env = environment.strip().lower()
    
    if env == "aws":
        try:
            if role == "client":
                print("[FACTORY] Client AWS (API Gateway -> Lambda -> SQS / StateManager)...")
                queue = LambdaGatewaySQSQueue()
                state_manager = ApiGatewayStateManager()
            else:
                print("[FACTORY] Worker/Orchestrator AWS (Boto3 diretto)...")
                queue = AwsSQSQueue()
                state_manager = AwsStateManager()
                
            return queue, state_manager
        except Exception as e:
            print(f"\n[FACTORY] [FALLBACK] {e}")
            return MockSQSQueue(), MockStateManager()
    else:
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
        
