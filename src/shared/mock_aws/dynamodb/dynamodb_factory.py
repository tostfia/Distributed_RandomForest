"""
Factory che restituisce l'implementazione corretta di accesso a DynamoDB
in base all'ambiente (locale -> mock su file, aws -> boto3 reale).

Uso previsto (stesso pattern di DatasetDAOFactory.get_dao(environment)):

    from src.shared.factory.dynamodb_factory import DynamoDBFactory
    db = DynamoDBFactory.get_db(self.environment)   # self.environment: "local" | "aws"
    db.put_item("workers_registry", worker_name, {...})

In questo modo tutto il codice che oggi importa direttamente
`dynamo_db` da dynamodb.py va aggiornato per passare invece attraverso
questa factory, esattamente come già fatto per il DAO dei dataset (S3).
"""

from typing import Optional


class DynamoDBFactory:

    _mock_instance = None
    _aws_instance = None

    @classmethod
    def get_db(cls, environment: str, region_name: Optional[str] = None):
        environment = (environment or "local").lower()

        if environment == "aws":
            if cls._aws_instance is None:
                from src.shared.mock_aws.dynamodb.dynamodb_aws import AwsDynamoDB
                cls._aws_instance = AwsDynamoDB(region_name=region_name)
            return cls._aws_instance

        if cls._mock_instance is None:
            from src.shared.mock_aws.dynamodb.dynamodb import MockDynamoDB
            cls._mock_instance = MockDynamoDB()
        return cls._mock_instance