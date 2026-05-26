import sys

from src.shared.mock_aws.interfaces import SQSQueueInterface, StateManagerInterface

def get_aws_services(environment: str) -> tuple[SQSQueueInterface, StateManagerInterface]:
    """
    Factory polimorfa per il disaccoppiamento dell'infrastruttura cloud.
    In base all'ambiente richiesto ('aws' o 'local'), istanzia e restituisce
    i componenti corretti (Mock basati su file o Client reali basati su boto3).
    """
    
    if environment == "aws":
        try:
            print("[FACTORY] Tentativo di inizializzazione dei servizi AWS reali (Boto3)...")
            # Caricamento dinamico: evita errori se boto3 non è installato in locale
            sys.exit(1)
            
        except ImportError:
            print("\n[FACTORY] [FALLBACK] Moduli AWS reali non trovati o boto3 mancante.")
            print("[FACTORY] Deviazione automatica sui Mock persistenti del File System...\n")
            
            sys.exit(1)
    else:
        # Ambiente locale: usiamo direttamente le istanze dei mock JSON persistenti
        from src.shared.mock_aws.sqs import sqs_queue
        from src.shared.mock_aws.statemanager import state_manager
        
        return sqs_queue, state_manager