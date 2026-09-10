"""
Helper condiviso per gli scenari di test eseguiti contro l'infrastruttura
AWS: distingue esecuzione locale/Docker da esecuzione su AWS.
"""

import os

def is_aws_environment(orchestrator) -> bool:
    """
    True se lo scenario deve usare i meccanismi/misure specifici per AWS
    invece di quelli locale/Docker (es. 'tc netem' per iniettare latenza).
    """
    env_attr = getattr(orchestrator, "environment", "") or ""
    return env_attr == "aws" or os.environ.get("ENV_MODE", "").lower() == "aws"