import json
import boto3
import os
import time
import uuid

# Inizializzazione globale per riutilizzo del contesto di esecuzione (Warm Start)
dynamodb = boto3.resource('dynamodb')
sqs = boto3.client('sqs')

def lambda_handler(event, context):
    try:
        # 1. Parsing del payload da API Gateway
        body = json.loads(event.get('body', '{}'))
        path = event.get('rawPath', '')  # Es. /jobs/centralized o /jobs/federated
        
        job_id = body.get('job_id')
        if not job_id:
            job_id = str(uuid.uuid4())
            body['job_id'] = job_id
            
        request_type = body.get('request_type', 'TRAINING').upper()

        # 2. Scrittura stato iniziale su DynamoDB (Solo per Training)
        if request_type == 'TRAINING':
            table_name = os.environ.get('DYNAMODB_TABLE', 'ModelStatus')
            table = dynamodb.Table(table_name)
            table.put_item(
                Item={
                    'job_id': job_id,
                    'status': 'QUEUED',
                    'dataset_path': body.get('dataset_path', ''),
                    'timestamp': int(time.time()),
                    'retries': 0,
                    'last_orchestrator': None,
                    'alberi_addestrati': 0,
                    'base_random_state': body.get('seed', 123)
                }
            )

        # 3. Routing sulla coda SQS corretta
        if 'federated' in path:
            queue_url = os.environ['SQS_FEDERATED_URL']
            group_id = 'ML-Federated-Group'
        else:
            queue_url = os.environ['SQS_CENTRALIZED_URL']
            group_id = 'ML-Centralized-Group'

        sqs_params = {
            'QueueUrl': queue_url,
            'MessageBody': json.dumps(body)
        }

        # Gestione obbligatoria per code FIFO
        if queue_url.endswith('.fifo'):
            sqs_params['MessageGroupId'] = group_id
            sqs_params['MessageDeduplicationId'] = str(uuid.uuid4())

        # 4. Inoltro del messaggio
        response = sqs.send_message(**sqs_params)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Job ricevuto e accodato con successo',
                'job_id': job_id,
                'message_id': response.get('MessageId')
            })
        }

    except Exception as e:
        print(f"[ERRORE LAMBDA] {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }