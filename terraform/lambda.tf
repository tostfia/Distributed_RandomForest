# ---------------------------------------------------------------------
# Lambda RFGatewayLambda: router HTTP unico per submit (POST /jobs/{mode})
# e status (GET /jobs/{job_id}/status | GET /jobs/{job_id}/details).
# Codice copiato 1:1 dall'account Learner Lab funzionante
# (terraform/lambda/lambda_function.py) — nessuna modifica alla logica.
#
# NOTA comportamento esistente (non un bug da questa consegna): sia
# /status che /details invocano handle_status() e restituiscono la
# stessa risposta, perché lambda_handler smista solo su GET vs POST,
# non sul path esatto.
# ---------------------------------------------------------------------

data "archive_file" "rf_gateway_lambda" {
  type        = "zip"
  source_file = "${path.module}/lambda/lambda_function.py"
  output_path = "${path.module}/lambda/lambda_function.zip"
}

resource "aws_lambda_function" "rf_gateway" {
  function_name = "RFGatewayLambda"
  runtime       = "python3.12"
  handler       = "lambda_function.lambda_handler"
  role          = data.aws_iam_role.lab_role.arn

  filename         = data.archive_file.rf_gateway_lambda.output_path
  source_code_hash = data.archive_file.rf_gateway_lambda.output_base64sha256

  timeout     = 15
  memory_size = 128

  # Variabili d'ambiente puntate alle risorse create da QUESTO modulo
  # (dynamodb.tf / sqs.tf), non ai valori letterali dell'altro account.
  environment {
    variables = {
      DYNAMODB_TABLE      = aws_dynamodb_table.model_status.name
      SQS_CENTRALIZED_URL = aws_sqs_queue.centralized_queue.url
      SQS_FEDERATED_URL   = aws_sqs_queue.federated_queue.url
    }
  }

  tags = { Project = var.project_name }
}

# Permesso esplicito per API Gateway di invocare la funzione.
# Wildcard su stage/metodo/risorsa: copre tutte e 4 le rotte, coerente
# con la policy 'AllowAllMyAPIRoutes' osservata sull'account funzionante.
resource "aws_lambda_permission" "apigw_invoke" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.rf_gateway.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.mljobs.execution_arn}/*/*"
}
