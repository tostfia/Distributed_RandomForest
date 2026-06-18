import os
from dotenv import load_dotenv

class SystemConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            # Carica le variabili dal file .env automaticamente
            load_dotenv() 
            
            cls._instance = super(SystemConfig, cls).__new__(cls)
            
            # Legge le variabili con i tuoi valori di default
            cls._instance.mode = os.getenv("SYS_MODE", "centralized")
            cls._instance.env = os.getenv("SYS_ENV", "local")
            
            # Validazione fondamentale per la robustezza del sistema
            if cls._instance.mode not in ["centralized", "federated"]:
                raise ValueError(f"SYS_MODE non valido nel file .env: {cls._instance.mode}")
                
            print(f"[CONFIG] Sistema caricato: {cls._instance.mode.upper()} | Ambiente: {cls._instance.env.upper()}")
            
        return cls._instance