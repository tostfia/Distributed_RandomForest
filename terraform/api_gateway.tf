# ---------------------------------------------------------------------
# API Gateway HTTP API "MLJobsAPI" — replica 1:1 delle 4 rotte trovate
# sull'account Learner Lab funzionante (apigatewayv2, nessuna auth,
# stage $default con autodeploy). Tutte le rotte puntano alla stessa
# integrazione AWS_PROXY verso RFGatewayLambda, che fa da router interno.
# ---------------------------------------------------------------------

resource "aws_apigatewayv2_api" "mljobs" {
  name          = "MLJobsAPI"
  protocol_type = "HTTP"

  tags = { Project = var.project_name }
}

resource "aws_apigatewayv2_integration" "rf_gateway" {
  api_id                 = aws_apigatewayv2_api.mljobs.id
  integration_type       = "AWS_PROXY"
  integration_method     = "POST"
  integration_uri        = aws_lambda_function.rf_gateway.invoke_arn
  payload_format_version = "2.0"
  timeout_milliseconds   = 30000
}

resource "aws_apigatewayv2_route" "get_status" {
  api_id    = aws_apigatewayv2_api.mljobs.id
  route_key = "GET /jobs/{job_id}/status"
  target    = "integrations/${aws_apigatewayv2_integration.rf_gateway.id}"
}

resource "aws_apigatewayv2_route" "get_details" {
  api_id    = aws_apigatewayv2_api.mljobs.id
  route_key = "GET /jobs/{job_id}/details"
  target    = "integrations/${aws_apigatewayv2_integration.rf_gateway.id}"
}

resource "aws_apigatewayv2_route" "post_centralized" {
  api_id    = aws_apigatewayv2_api.mljobs.id
  route_key = "POST /jobs/centralized"
  target    = "integrations/${aws_apigatewayv2_integration.rf_gateway.id}"
}

resource "aws_apigatewayv2_route" "post_federated" {
  api_id    = aws_apigatewayv2_api.mljobs.id
  route_key = "POST /jobs/federated"
  target    = "integrations/${aws_apigatewayv2_integration.rf_gateway.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.mljobs.id
  name        = "$default"
  auto_deploy = true

  tags = { Project = var.project_name }
}
