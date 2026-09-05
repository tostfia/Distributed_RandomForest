"""
Helper condiviso per gli scenari di test eseguiti contro l'infrastruttura
AWS: distingue esecuzione locale/Docker da esecuzione su AWS.

Storicamente questo modulo conteneva anche helper ECS-specifici (risoluzione
del task ARN del leader tramite runtimeId, stop_task, ecc.), rimasti però
inutilizzati: fault.py/fault_inf.py e orchestrator_fault.py/
orchestrator_fault_inf.py hanno sempre implementato localmente le proprie
versioni di queste funzioni (con logica leggermente diversa, basata sull'IP
del leader anziché sul runtimeId del container). Rimossi per pulizia —
vedi git history se mai servisse recuperarli.

is_aws_environment è l'unica funzione di questo modulo realmente importata
e usata, da network.py (scenario 3), per decidere se iniettare latenza via
'tc' (locale/Docker) o limitarsi a misurare la latenza RPC reale (AWS).
"""

import os

def is_aws_environment(orchestrator) -> bool:
    """
    True se lo scenario deve usare i meccanismi/misure specifici per AWS
    invece di quelli locale/Docker (es. 'tc netem' per iniettare latenza).
    """
    env_attr = getattr(orchestrator, "environment", "") or ""
    return env_attr == "aws" or os.environ.get("ENV_MODE", "").lower() == "aws"