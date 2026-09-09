import time
import re
from src.testing.scenarios.base import BaseTestScenario

# ---------------------------------------------------------------------
# Stessa tabella/chiave del lock di leadership usate da orchestrator_fault.py
# (vedi dynamodb_aws.py/try_acquire_lock): riletta qui solo per identificare
# quale istanza EC2 sta facendo da leader al momento della terminazione, un
# dato utile per il report ma non necessario alla verifica principale di
# questo scenario (che è a livello di infrastruttura, non applicativo).
# ---------------------------------------------------------------------
_LOCK_TABLE = "OrchestratorLocks"
_LOCK_KEY = "global_orchestrator_leader_lock"

# Nome dell'Auto Scaling Group, fisso in orchestrator_ec2.tf
# (resource "aws_autoscaling_group" "orchestrator" { name = "orchestrator-asg" ... }).
_DEFAULT_ASG_NAME = "orchestrator-asg"

# Tag propagato a ogni istanza dall'ASG (propagate_at_launch = true), stesso
# usato da _resolve_ec2_instance_by_ip in orchestrator_fault.py per trovare
# le istanze dell'orchestrator senza passare da un nome di service ECS.
_PROJECT_TAG_VALUE = "rf-distributed"


def _resolve_aws_infra(config: dict):
    """
    Region e nome dell'ASG: letti da config['aws'] quando presenti,
    altrimenti fallback su default fisso. Vedi lo stesso pattern in
    orchestrator_fault.py/_resolve_aws_infra.
    """
    import os
    aws_cfg = config.get("aws", {}) or {}
    region = aws_cfg.get("region") or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    asg_name = aws_cfg.get("orchestrator_asg_name", _DEFAULT_ASG_NAME)
    return region, asg_name


def _merge_aws_overrides(config: dict, key: str) -> dict:
    """Vedi il commento gemello in orchestrator_fault.py: stesso meccanismo di override."""
    merged = dict(config.get(key, {}) or {})
    if (config.get("aws", {}) or {}).get("suggested_overrides", {}).get(key):
        overrides = config["aws"]["suggested_overrides"][key]
        merged.update({k: v for k, v in overrides.items() if not k.startswith("_")})
    return merged


def _get_current_leader_name_aws(state_manager):
    """
    Vedi il commento gemello in orchestrator_fault.py. Usata qui SOLO per
    arricchire il report (quale istanza era leader al momento della
    terminazione): la verifica di questo scenario non dipende dalla
    leadership applicativa, solo dal fatto che l'ASG rimpiazzi l'istanza.
    """
    try:
        item = state_manager._db.get_item(_LOCK_TABLE, _LOCK_KEY)
        return item.get("Item", {}).get("leader")
    except Exception:
        return None


def _describe_asg_instances(asg_client, ec2_client, asg_name):
    """
    Ritorna la lista delle istanze CONOSCIUTE DALL'ASG (qualunque
    LifecycleState: Pending, InService, Terminating, ...) come dict
    {instance_id: {"lifecycle_state": ..., "health_status": ..., "private_ip": ...}}.

    Usiamo l'API autoscaling (non una describe_instances per tag) come fonte
    di verità: è l'ASG stesso, non una ricostruzione a posteriori via tag,
    a dover confermare che considera l'istanza rimpiazzata come InService —
    lo stesso segnale che governerebbe un vero controllo di produzione.
    L'IP privato (non esposto da describe_auto_scaling_groups) viene
    arricchito con una singola describe_instances aggiuntiva, solo per le
    istanze effettivamente restituite dall'ASG.
    """
    resp = asg_client.describe_auto_scaling_groups(AutoScalingGroupNames=[asg_name])
    groups = resp.get("AutoScalingGroups", [])
    if not groups:
        return {}
    instances = groups[0].get("Instances", [])
    result = {
        inst["InstanceId"]: {
            "lifecycle_state": inst.get("LifecycleState"),
            "health_status": inst.get("HealthStatus"),
            "private_ip": None,
        }
        for inst in instances
    }
    if not result:
        return result

    try:
        details = ec2_client.describe_instances(InstanceIds=list(result.keys()))
        for r in details.get("Reservations", []):
            for inst in r.get("Instances", []):
                iid = inst.get("InstanceId")
                if iid in result:
                    result[iid]["private_ip"] = inst.get("PrivateIpAddress")
    except Exception as e:
        # Non fatale: l'IP è solo un dettaglio di log/report, la verifica
        # principale (InService, nuovo instance ID) non dipende da esso.
        print(f"[TEST WARN] Impossibile arricchire con l'IP privato: {e}")

    return result


class OrchestratorAsgReplacementScenario(BaseTestScenario):
    """
    Copre lo Scenario: sostituzione REALE, a livello di infrastruttura, di
    un'istanza EC2 dell'orchestrator da parte dell'Auto Scaling Group
    'orchestrator-asg' (vedi orchestrator_ec2.tf) dopo una terminazione.

    Distinto da OrchestratorFailoverScenario (orchestrator_fault.py):
    quello scenario verifica il FAILOVER APPLICATIVO (lock/lease, il job
    continua sull'altra istanza già viva); questo verifica che l'ASG
    ripristini davvero la CAPACITÀ perduta con una nuova istanza — il pezzo
    di cui BaseOrchestrator/DynamoDB non possono occuparsi da soli, perché
    riguarda l'infrastruttura sotto di loro, non la loro logica applicativa.

    Applicabile SOLO su AWS: un Auto Scaling Group non ha equivalente
    locale/Docker Compose. Su environment != 'aws' lo scenario si limita a
    dichiararsi non applicabile (status 'SKIPPED'), senza tentare nulla.
    """

    def run(self) -> dict:
        environment = getattr(self.orchestrator, "environment", "local")
        if environment != "aws":
            print("\n[TEST] Scenario 'Sostituzione ASG dell'Orchestratore' applicabile solo "
                  "su AWS (nessun equivalente locale/Docker per un Auto Scaling Group): saltato.")
            return {
                "scenario_description": "Verifica della sostituzione automatica, da parte "
                                         "dell'Auto Scaling Group, di un'istanza EC2 "
                                         "dell'orchestrator terminata. Applicabile solo su AWS.",
                "status": "SKIPPED",
                "reason": f"environment='{environment}', nessun Auto Scaling Group in locale/Docker.",
            }

        import boto3

        cfg = _merge_aws_overrides(self.config, "orchestrator_asg_replacement")
        region, asg_name = _resolve_aws_infra(self.config)
        max_timeout = cfg.get("max_wait_for_replacement_seconds", 420)
        check_interval = cfg.get("check_interval_seconds", 10)

        asg_client = boto3.client("autoscaling", region_name=region)
        ec2_client = boto3.client("ec2", region_name=region)

        print(f"\n--- [TEST] Sostituzione ASG dell'Orchestratore ('{asg_name}', region '{region}') ---")

        # 1. Baseline: istanze attualmente InService secondo l'ASG stesso.
        try:
            baseline = _describe_asg_instances(asg_client, ec2_client, asg_name)
        except Exception as e:
            print(f"[TEST ERRORE] Impossibile leggere l'Auto Scaling Group '{asg_name}': {e}")
            return {"status": "FAILED", "duration_seconds": 0,
                    "error": f"describe_auto_scaling_groups fallita: {e}"}

        baseline_inservice = {iid: d for iid, d in baseline.items() if d["lifecycle_state"] == "InService"}
        if len(baseline_inservice) < 2:
            print(f"[TEST ERRORE] Solo {len(baseline_inservice)} istanza/e InService trovate in "
                  f"'{asg_name}' (attese >= 2, Leader+Standby): lo scenario richiede che l'ASG sia "
                  f"già a regime prima di simulare un guasto, altrimenti un fallimento qui sarebbe "
                  f"indistinguibile da un ASG che non si è ancora stabilizzato da un run precedente.")
            return {"status": "FAILED", "duration_seconds": 0,
                    "error": f"Solo {len(baseline_inservice)} istanza/e InService (attese >= 2).",
                    "baseline_instances": list(baseline_inservice.keys())}

        baseline_ids = set(baseline_inservice.keys())
        print(f"[TEST] Baseline: {len(baseline_ids)} istanze InService -> "
              f"{[(iid, d['private_ip']) for iid, d in baseline_inservice.items()]}")

        # 2. Scegli il target: preferibilmente il leader applicativo corrente
        # (arricchisce il report), altrimenti una qualunque istanza baseline.
        state_manager = getattr(self.orchestrator, "state_manager", None)
        leader_name = _get_current_leader_name_aws(state_manager) if state_manager else None
        target_id = None
        if leader_name:
            ip_match = re.search(r"ip-(\d+-\d+-\d+-\d+)\.ec2\.internal", leader_name)
            if ip_match:
                ip_dashed = ip_match.group(1)
                for iid, d in baseline_inservice.items():
                    if d["private_ip"] and d["private_ip"].replace(".", "-") == ip_dashed:
                        target_id = iid
                        break
        was_leader = target_id is not None
        if target_id is None:
            target_id = next(iter(baseline_ids))
            print(f"[TEST] Leader applicativo non identificato dal lock (o non corrispondente a "
                  f"un'istanza InService): termino un'istanza arbitraria della baseline.")

        target_ip = baseline_inservice[target_id]["private_ip"]
        print(f"\n[TEST TRIGGER] !!! Termino DAVVERO l'istanza EC2 {target_id} "
              f"({target_ip}{', LEADER applicativo corrente' if was_leader else ''}) via "
              f"ec2:TerminateInstances — non un docker kill: qui l'obiettivo è che l'istanza "
              f"stessa sparisca, per verificare che l'ASG ne lanci una di rimpiazzo !!!")

        try:
            ec2_client.terminate_instances(InstanceIds=[target_id])
        except Exception as e:
            print(f"[TEST ERRORE] ec2:TerminateInstances fallita su {target_id}: {e}")
            return {"status": "FAILED", "duration_seconds": 0,
                    "error": f"TerminateInstances fallita: {e}", "target_instance": target_id}

        start_time = time.perf_counter()

        # 3. Monitoraggio: attendiamo che (a) compaia un instance ID MAI visto
        # nella baseline e (b) l'ASG lo consideri InService, ripristinando la
        # capacità InService totale al livello della baseline. Il primo
        # avvistamento (qualunque LifecycleState, es. Pending) e il momento in
        # cui diventa InService sono misurati separatamente: la differenza fra
        # i due è, in pratica, il tempo di boot dell'istanza + avvio Docker.
        replacement_id = None
        replacement_first_seen_seconds = None
        replacement_inservice_seconds = None
        final_snapshot = {}
        elapsed = 0

        print(f"[TEST] Monitoraggio ASG '{asg_name}' (timeout: {max_timeout}s, poll ogni {check_interval}s)...")
        while elapsed < max_timeout:
            time.sleep(check_interval)
            elapsed += check_interval
            try:
                snapshot = _describe_asg_instances(asg_client, ec2_client, asg_name)
            except Exception as e:
                print(f"[TEST MONITOR WARNING] describe_auto_scaling_groups fallita: {e}")
                continue

            final_snapshot = snapshot
            new_ids = set(snapshot.keys()) - baseline_ids - {target_id}

            if new_ids and replacement_id is None:
                # Più di un ID nuovo non dovrebbe accadere qui (min=max=desired
                # invariati durante lo scenario), ma se capitasse prendiamo il
                # primo per determinismo del report.
                replacement_id = sorted(new_ids)[0]
                replacement_first_seen_seconds = round(time.perf_counter() - start_time, 2)
                print(f"[TEST MONITOR] Nuova istanza rilevata dall'ASG dopo "
                      f"{replacement_first_seen_seconds:.1f}s: {replacement_id} "
                      f"(stato: {snapshot[replacement_id]['lifecycle_state']}).")

            if replacement_id and snapshot.get(replacement_id, {}).get("lifecycle_state") == "InService":
                replacement_inservice_seconds = round(time.perf_counter() - start_time, 2)
                print(f"[TEST MONITOR] Istanza di rimpiazzo {replacement_id} InService dopo "
                      f"{replacement_inservice_seconds:.1f}s "
                      f"(IP: {snapshot[replacement_id]['private_ip']}).")
                break

            current_inservice = [iid for iid, d in snapshot.items() if d["lifecycle_state"] == "InService"]
            print(f"[TEST MONITOR] InService: {len(current_inservice)}/{len(baseline_ids)} "
                  f"(target rimosso: {target_id not in snapshot or snapshot.get(target_id, {}).get('lifecycle_state') != 'InService'})")

        duration = round(time.perf_counter() - start_time, 2)

        if replacement_inservice_seconds is not None:
            test_status = "SUCCESS"
            print(f"\n[TEST PASSED] L'ASG ha sostituito l'istanza terminata: {replacement_id} "
                  f"è InService dopo {replacement_inservice_seconds:.1f}s.")
        else:
            test_status = "FAILED"
            reason = ("nessuna nuova istanza rilevata entro il timeout" if replacement_id is None
                       else "nuova istanza rilevata ma mai diventata InService entro il timeout")
            print(f"\n[TEST FAILED] Sostituzione non verificata: {reason}.")

        result = {
            "scenario_description": "Verifica che l'Auto Scaling Group 'orchestrator-asg' sostituisca "
                                     "con una nuova istanza EC2 quella terminata (ec2:TerminateInstances "
                                     "diretto, non un docker kill: qui è l'istanza stessa a sparire). "
                                     "Complementare al failover applicativo di OrchestratorFailoverScenario: "
                                     "questo scenario verifica il ripristino della CAPACITÀ infrastrutturale, "
                                     "non la continuità del job in corso.",
            "status": test_status,
            "asg_name": asg_name,
            "terminated_instance_id": target_id,
            "terminated_instance_ip": target_ip,
            "terminated_instance_was_leader": was_leader,
            "replacement_instance_id": replacement_id,
            "replacement_instance_ip": final_snapshot.get(replacement_id, {}).get("private_ip") if replacement_id else None,
            "replacement_first_seen_seconds": replacement_first_seen_seconds,
            "replacement_inservice_seconds": replacement_inservice_seconds,
            "duration_seconds": duration,
            "monitor_timeout_seconds": max_timeout,
            "check_interval_seconds": check_interval,
        }
        return result