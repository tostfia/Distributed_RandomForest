import json
import os
import time
from typing import Optional

class MockDynamoDB:
    def __init__(self):
        base = os.environ.get("LOCAL_STORAGE_PATH", os.path.join(".", ".local_storage"))
        self.base_dir = os.path.abspath(os.path.join(base, "db_files"))
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_table_path(self, table_name: str) -> str:
        """Restituisce il percorso del file JSON per una specifica tabella."""
        return os.path.join(self.base_dir, f"{table_name}.json")

    def _load_table(self, table_name: str) -> dict:
        path = self._get_table_path(table_name)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return {}

        # Tentiamo di leggere per un numero massimo di volte (es. 5)
        for attempt in range(5):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, PermissionError):
                # Aspetta un po' prima di riprovare (concorrenza tra processi)
                time.sleep(0.05)
        
        print(f"[ERRORE] Impossibile leggere il file {table_name}.json dopo 5 tentativi.")
        return {}

    def _save_table(self, table_name: str, data: dict):
        """Salva i dati aggiornati della tabella nel file JSON con retry concorrenziale."""
        path = self._get_table_path(table_name)
        
        for attempt in range(5):
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                return
            except (PermissionError, IOError):
                time.sleep(0.05)
        
        print(f"[ERRORE Critico] Impossibile scrivere la tabella {table_name} su disco.")
    
    def _get_primary_key_name(self, table_name: str) -> str:
        """Restituisce il nome della chiave primaria per una tabella specifica."""
        if table_name == 'workers_registry':
            return 'worker_name'
        elif table_name == 'orchestrators_registry':
            return 'orchestrator_name'
        elif table_name == 'ModelStatus':
            return 'job_id'
        elif table_name == 'WorkerTasks':
            return 'task_id'
        else:
            raise ValueError(f"Tabella '{table_name}' non riconosciuta per determinare la chiave primaria.")

    def put_item(self, table_name: str, key: str, value: dict):
        str_key = str(key)
        
        table = self._load_table(table_name)
        table[str_key] = value
        self._save_table(table_name, table)
        
        info = f"Stato: {value.get('status')}" if "status" in value else f"Dati: {list(value.keys())}"
        print(f"[Mock DynamoDB] Tabella '{table_name}' -> Scritto ID: {str_key[:8]}... | {info}")

    def get_item(self, table_name: str, key: str) -> Optional[dict]:
        str_key = str(key)
        table = self._load_table(table_name)
        raw_item = table.get(str_key)

        if raw_item:
            pk_name = self._get_primary_key_name(table_name)
            item_compliant = raw_item.copy()    
            item_compliant[pk_name] = str_key  

            return {"Item": item_compliant}
        
        return {}

    def delete_item(self, table_name: str, key: str) -> bool:
        """Rimuove un record dal file JSON."""
        str_key = str(key)
        table = self._load_table(table_name)
        if str_key in table:
            del table[str_key]
            self._save_table(table_name, table)
            print(f"[Mock DynamoDB] Tabella '{table_name}' -> Eliminato ID: {str_key[:8]}...")
            return True
        return False
    
    def scan_table(self, table_name: str) -> dict:
        """Restituisce tutti i record di una tabella."""
        table_data = self._load_table(table_name)
        items_list = []
        pk_name = self._get_primary_key_name(table_name)

        for key, value in table_data.items():
            item_compliant = value.copy()
            item_compliant[pk_name] = key
            items_list.append(item_compliant)
        
        return {"Items": items_list}
    
# Istanza globale
dynamo_db = MockDynamoDB()