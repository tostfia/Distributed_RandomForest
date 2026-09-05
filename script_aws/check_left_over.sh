#!/bin/bash
# =====================================================================
# CHECK LEFTOVER: controllo READ-ONLY (nessuna modifica) di risorse AWS
# che continuano a fatturare anche quando non stai testando attivamente.
#
# Pensato per essere lanciato a inizio giornata o dopo ogni sessione di
# lavoro, per assicurarti che tra un teardown.sh e l'altro non sia
# rimasto acceso nulla per errore (task interrotti, cluster secondari,
# NAT Gateway, Load Balancer, VPC Endpoint dimenticati).
#
# Uso:
#   ./check_leftover.sh
# =====================================================================
set -u

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
FOUND_ANYTHING=0

echo "===================================================================="
echo " CHECK LEFTOVER  -  Account: $(aws sts get-caller-identity --query Account --output text --region "$REGION" 2>/dev/null || echo '???')  -  Regione: $REGION"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

if ! aws sts get-caller-identity --region "$REGION" > /dev/null 2>&1; then
  echo "[ERRORE] Credenziali AWS non valide o scadute (Learner Lab, scadono ogni ~4h)."
  echo "         Aggiornale e riprova."
  exit 1
fi

# ---------------------------------------------------------------------
# 1) TUTTI i cluster ECS (non solo forest-cluster): servizi attivi e task
# ---------------------------------------------------------------------
echo ""
echo "--- [1/5] Cluster ECS: servizi attivi e task in esecuzione ---------"

CLUSTER_ARNS=$(aws ecs list-clusters --region "$REGION" --query "clusterArns[]" --output text 2>/dev/null || echo "")

if [ -z "$CLUSTER_ARNS" ]; then
  echo "  Nessun cluster ECS trovato nell'account."
else
  for cluster_arn in $CLUSTER_ARNS; do
    cluster_name="${cluster_arn##*/}"
    echo "  Cluster: $cluster_name"

    SERVICE_ARNS=$(aws ecs list-services --cluster "$cluster_name" --region "$REGION" \
      --query "serviceArns[]" --output text 2>/dev/null || echo "")

    if [ -n "$SERVICE_ARNS" ]; then
      SERVICE_NAMES=$(echo "$SERVICE_ARNS" | tr '\t' '\n' | sed 's#.*/##' | tr '\n' ' ')
      SVC_DETAILS=$(aws ecs describe-services --cluster "$cluster_name" --services $SERVICE_NAMES \
        --region "$REGION" \
        --query "services[?desiredCount > \`0\`].[serviceName,desiredCount,runningCount]" \
        --output text 2>/dev/null || echo "")
      if [ -n "$SVC_DETAILS" ]; then
        FOUND_ANYTHING=1
        echo "    [ATTIVO] Servizi con desired-count > 0:"
        echo "$SVC_DETAILS" | while IFS=$'\t' read -r svc_name desired running; do
          echo "      - $svc_name (desired=$desired, running=$running)"
          echo "        Per fermarlo: aws ecs update-service --cluster $cluster_name --service $svc_name --desired-count 0 --region $REGION"
        done
      fi
    fi

    TASK_ARNS=$(aws ecs list-tasks --cluster "$cluster_name" --region "$REGION" \
      --desired-status RUNNING --query "taskArns[]" --output text 2>/dev/null || echo "")
    if [ -n "$TASK_ARNS" ]; then
      FOUND_ANYTHING=1
      TASK_COUNT=$(echo "$TASK_ARNS" | wc -w)
      echo "    [ATTIVO] $TASK_COUNT task RUNNING (inclusi eventuali task one-off tipo test-engine):"
      for t in $TASK_ARNS; do
        echo "      - $t"
      done
      echo "        Per fermarli: aws ecs stop-task --cluster $cluster_name --task <TASK_ARN> --region $REGION"
    fi

    if [ -z "$SERVICE_ARNS" ] && [ -z "$TASK_ARNS" ]; then
      echo "    Pulito (0 servizi attivi, 0 task in esecuzione)."
    fi
  done
fi

# ---------------------------------------------------------------------
# 2) NAT Gateway attivi (costo orario fisso indipendente dal traffico)
# ---------------------------------------------------------------------
echo ""
echo "--- [2/5] NAT Gateway attivi ----------------------------------------"

NAT_INFO=$(aws ec2 describe-nat-gateways --region "$REGION" \
  --filter "Name=state,Values=available" \
  --query "NatGateways[].[NatGatewayId,VpcId,SubnetId]" --output text 2>/dev/null || echo "")

if [ -n "$NAT_INFO" ]; then
  FOUND_ANYTHING=1
  echo "  [ATTIVO] NAT Gateway trovati (~0,045 USD/ora ciascuno + traffico, indipendentemente dall'uso):"
  echo "$NAT_INFO" | while IFS=$'\t' read -r nat_id vpc_id subnet_id; do
    echo "    - $nat_id (VPC: $vpc_id, Subnet: $subnet_id)"
    echo "      Per eliminarlo: aws ec2 delete-nat-gateway --nat-gateway-id $nat_id --region $REGION"
  done
else
  echo "  Pulito (nessun NAT Gateway attivo)."
fi

# ---------------------------------------------------------------------
# 3) Load Balancer attivi (ALB/NLB/CLB: costo orario fisso ciascuno)
# ---------------------------------------------------------------------
echo ""
echo "--- [3/5] Load Balancer attivi --------------------------------------"

ALB_NLB_INFO=$(aws elbv2 describe-load-balancers --region "$REGION" \
  --query "LoadBalancers[].[LoadBalancerArn,LoadBalancerName,Type,State.Code]" --output text 2>/dev/null || echo "")
CLB_INFO=$(aws elb describe-load-balancers --region "$REGION" \
  --query "LoadBalancerDescriptions[].LoadBalancerName" --output text 2>/dev/null || echo "")

if [ -n "$ALB_NLB_INFO" ]; then
  FOUND_ANYTHING=1
  echo "  [ATTIVO] Application/Network Load Balancer trovati:"
  echo "$ALB_NLB_INFO" | while IFS=$'\t' read -r lb_arn lb_name lb_type lb_state; do
    echo "    - $lb_name (tipo: $lb_type, stato: $lb_state)"
    echo "      Per eliminarlo: aws elbv2 delete-load-balancer --load-balancer-arn $lb_arn --region $REGION"
  done
fi

if [ -n "$CLB_INFO" ]; then
  FOUND_ANYTHING=1
  echo "  [ATTIVO] Classic Load Balancer trovati:"
  for clb_name in $CLB_INFO; do
    echo "    - $clb_name"
    echo "      Per eliminarlo: aws elb delete-load-balancer --load-balancer-name $clb_name --region $REGION"
  done
fi

if [ -z "$ALB_NLB_INFO" ] && [ -z "$CLB_INFO" ]; then
  echo "  Pulito (nessun Load Balancer attivo)."
fi

# ---------------------------------------------------------------------
# 4) Elastic IP non associati + VPC Endpoint di tipo Interface
#    (entrambi fatturati a ore indipendentemente dall'uso)
# ---------------------------------------------------------------------
echo ""
echo "--- [4/5] Elastic IP non associati e VPC Endpoint Interface --------"

UNUSED_EIP=$(aws ec2 describe-addresses --region "$REGION" \
  --query "Addresses[?AssociationId==null].[AllocationId,PublicIp]" --output text 2>/dev/null || echo "")

if [ -n "$UNUSED_EIP" ]; then
  FOUND_ANYTHING=1
  echo "  [ATTIVO] Elastic IP non associati (fatturati quando inattivi):"
  echo "$UNUSED_EIP" | while IFS=$'\t' read -r alloc_id public_ip; do
    echo "    - $public_ip (AllocationId: $alloc_id)"
    echo "      Per rilasciarlo: aws ec2 release-address --allocation-id $alloc_id --region $REGION"
  done
else
  echo "  Pulito (nessun Elastic IP non associato)."
fi

VPCE_INFO=$(aws ec2 describe-vpc-endpoints --region "$REGION" \
  --filters "Name=vpc-endpoint-type,Values=Interface" "Name=vpc-endpoint-state,Values=available" \
  --query "VpcEndpoints[].[VpcEndpointId,ServiceName]" --output text 2>/dev/null || echo "")

if [ -n "$VPCE_INFO" ]; then
  FOUND_ANYTHING=1
  echo "  [ATTIVO] VPC Endpoint di tipo Interface (~0,01 USD/ora ciascuno per AZ):"
  echo "$VPCE_INFO" | while IFS=$'\t' read -r vpce_id svc_name; do
    echo "    - $vpce_id ($svc_name)"
    echo "      Per eliminarlo: aws ec2 delete-vpc-endpoints --vpc-endpoint-ids $vpce_id --region $REGION"
  done
else
  echo "  Pulito (nessun VPC Endpoint Interface attivo)."
fi

# ---------------------------------------------------------------------
# 5) Istanze EC2 attive (orchestrator ora gira qui, non più su ECS:
#    NON coperto da nessun controllo sopra, che guarda solo cluster ECS)
# ---------------------------------------------------------------------
echo ""
echo "--- [5/5] Istanze EC2 attive -----------------------------------------"

EC2_INFO=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=instance-state-name,Values=running,pending" \
  --query "Reservations[].Instances[].[InstanceId,InstanceType,Tags[?Key=='Name']|[0].Value,LaunchTime]" \
  --output text 2>/dev/null || echo "")

if [ -n "$EC2_INFO" ]; then
  FOUND_ANYTHING=1
  echo "  [ATTIVO] Istanze EC2 in esecuzione (es. orchestrator-ec2-1/-2, r5.large ~0,126 USD/ora ciascuna):"
  echo "$EC2_INFO" | while IFS=$'\t' read -r inst_id inst_type inst_name launch_time; do
    echo "    - $inst_id ($inst_type, nome: ${inst_name:-nessuno}, avviata: $launch_time)"
    echo "      Per fermarla (pausa, EBS ancora fatturato): aws ec2 stop-instances --instance-ids $inst_id --region $REGION"
    echo "      Per distruggerla (coerente con Terraform, poi 'terraform apply' la ricrea se serve): aws ec2 terminate-instances --instance-ids $inst_id --region $REGION"
  done
else
  echo "  Pulito (nessuna istanza EC2 in esecuzione)."
fi

# ---------------------------------------------------------------------
# Riepilogo finale
# ---------------------------------------------------------------------
echo ""
echo "===================================================================="
if [ "$FOUND_ANYTHING" -eq 0 ]; then
  echo " RISULTATO: TUTTO PULITO. Nessuna risorsa a rischio di costo trovata."
else
  echo " RISULTATO: trovate risorse ATTIVE sopra elencate."
  echo " Controlla se sono volute (test in corso) o dimenticate da una sessione precedente."
fi
echo "===================================================================="