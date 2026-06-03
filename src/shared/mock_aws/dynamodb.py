import json
import os
from typing import Optional

class MockDynamoDB:
    def __init__(self):
        # Definiamo una cartella "db_files" all'interno della cartella del mock
        self.base_dir = os.path.join(os.path.dirname(__file__), "db_files")
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_table_path(self, table_name: str) -> str:
        """Restituisce il percorso del file JSON per una specifica tabella."""
        return os.path.join(self.base_dir, f"{table_name}.json")

    def _load_table(self, table_name: str) -> dict:
        """Legge i dati della tabella dal file JSON corrispondente."""
        path = self._get_table_path(table_name)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # Protezione da letture concorrenti asincrone
                import time
                time.sleep(0.05)
                return self._load_table(table_name)
        return {}

    def _save_table(self, table_name: str, data: dict):
        """Salva i dati aggiornati della tabella nel file JSON."""
        path = self._get_table_path(table_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def _get_primary_key_name(self, table_name: str) -> str:
        """Restituisce il nome della chiave primaria per una tabella specifica."""
        if table_name == 'workers_registry':
            return 'worker_name'
        elif table_name == 'orchestrators_registry':
            return 'orchestrator_name'
        else:
            raise ValueError(f"Tabella '{table_name}' non riconosciuta per determinare la chiave primaria.")

    def put_item(self, table_name: str, key: str, value: dict):
        # Forza la chiave a stringa
        str_key = str(key)
        
        # Carica la tabella dal file, aggiorna il record e salva su disco
        table = self._load_table(table_name)
        table[str_key] = value
        self._save_table(table_name, table)
        
        # Log pulito e dinamico a seconda dei dati presenti nel payload
        info = f"Stato: {value.get('status')}" if "status" in value else f"Dati: {list(value.keys())}"
        print(f"[Mock DynamoDB] Tabella '{table_name}' -> Scritto ID: {str_key[:8]}... | {info}")

    def get_item(self, table_name: str, key: str) -> Optional[dict]:
        str_key = str(key)
        # Carica la tabella in tempo reale dal file JSON
        table = self._load_table(table_name)
        raw_item = table.get(str_key)

        if raw_item:
            pk_name = self._get_primary_key_name(table_name)
            item_compliant = raw_item.copy()    
            item_compliant[pk_name] = str_key  

            return {"Item": item_compliant}
        
        return {}

    def delete_item(self, table_name: str, key: str) -> bool:
        """Rimuove un record dal file JSON (utile per ripulire stati o Service Discovery)."""
        str_key = str(key)
        table = self._load_table(table_name)
        if str_key in table:
            del table[str_key]
            self._save_table(table_name, table)
            print(f"[Mock DynamoDB] Tabella '{table_name}' -> Eliminato ID: {str_key[:8]}...")
            return True
        return False
    
    def scan_table(self, table_name: str) -> dict:
        """Restituisce tutti i record di una tabella (simulando l'operazione di scan)."""
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