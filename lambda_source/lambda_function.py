import json
import boto3
import os
import time
import uuid

# Inizializzazione globale per riutilizzo del contesto di esecuzione (Warm Start)
dynamodb = boto3.resource('dynamodb')
sqs = boto3.client('sqs')
JOBS_TABLE = os.environ.get('DYNAMODB_TABLE', 'ModelStatus')


def lambda_handler(event, context):
    try:
        # Estrazione del metodo HTTP (GET o POST). Se non è specificato, si assume sia un POST.
        method = event.get('requestContext', {}).get('http', {}).get('method', 'POST')
        path_params = event.get('pathParameters') or {}

        if method == 'GET':
            return handle_status(path_params)
        return handle_submit(event, path_params)

    except Exception as e:
        print(f"[ERRORE LAMBDA] {str(e)}")
        return _response(500, {"error": str(e)})


def handle_status(path_params: dict) -> dict:
    """Gestisce GET /jobs/{job_id}/status leggendo direttamente da DynamoDB."""
    job_id = path_params.get('job_id')
    if not job_id:
        return _response(400, {"error": "Parametro 'job_id' mancante nel path."})

    # Connessione alla tabella e ricerca del record tramite la chiave primaria (job_id)
    table = dynamodb.Table(JOBS_TABLE)
    result = table.get_item(Key={'job_id': job_id})
    item = result.get('Item')

    if not item:
        return _response(404, {"error": f"Job con ID '{job_id}' non trovato."})

    # Restituzione dello stato del job
    return _response(200, {
        'job_id': job_id,
        'status': item.get('status'),
        'timestamp': int(item.get('timestamp', 0)),
    })

def handle_submit(event: dict, path_params: dict) -> dict:
    """Gestisce POST /jobs/{mode} (mode = centralized | federated)."""
    body = json.loads(event.get('body') or '{}')

    path_mode = (path_params or {}).get('mode', '').lower()
    body_mode = body.get('mode', '').lower()
    raw_path = (event.get('rawPath') or event.get('path') or '').lower()

    if path_mode == 'federated' or body_mode == 'federated' or 'federated' in raw_path:
        mode = 'federated'
    else:
        mode = 'centralized'

    print(f"[DEBUG] path_params={path_params} | path_mode='{path_mode}' | body_mode='{body_mode}' | raw_path='{raw_path}' | mode_scelto='{mode}'")

    job_id = body.get('job_id') or str(uuid.uuid4())
    body['job_id'] = job_id

    request_type = body.get('request_type', 'TRAINING').upper()

    """Scrittura stato iniziale su DynamoDB solo per il training. Le richieste di inferenza referenziano un job_id già esistente"""
    if request_type == 'TRAINING':
        table = dynamodb.Table(JOBS_TABLE)
        table.put_item(
            Item={
                'job_id': job_id,
                'status': 'QUEUED',
                'dataset_path': body.get('dataset_path', ''),
                'timestamp': int(time.time()),
                'retries': 0,
                'last_orchestrator': None,
                'alberi_addestrati': 0,
                'base_random_state': body.get('seed', 123),
            }
        )

    if mode == 'centralized':
        queue_url = os.environ['SQS_CENTRALIZED_URL']
        group_id = 'ML-Centralized-Group'
    else:
        queue_url = os.environ['SQS_FEDERATED_URL']
        group_id = 'ML-Federated-Group'

    sqs_params = {
        'QueueUrl': queue_url,
        'MessageBody': json.dumps(body),
    }
    
    if queue_url.endswith('.fifo'):
        sqs_params['MessageGroupId'] = group_id
        sqs_params['MessageDeduplicationId'] = str(uuid.uuid4())

    response = sqs.send_message(**sqs_params)

    return _response(200, {
        'message': f"Job '{job_id}' inviato con successo alla coda SQS ({mode}).",
        'job_id': job_id,
        'sqs_message_id': response.get('MessageId'),
    })

def _response(status_code: int, body: dict) -> dict:
    return {
        'statusCode': status_code,
        'body': json.dumps(body),
    }