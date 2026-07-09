import json
import os
import fcntl
from typing import Optional

class MockDynamoDB:
    def __init__(self):
        base = os.environ.get("LOCAL_STORAGE_PATH", os.path.join(".", ".local_storage"))
        self.base_dir = os.path.abspath(os.path.join(base, "db_files"))
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_table_path(self, table_name: str) -> str:
        return os.path.join(self.base_dir, f"{table_name}.json")

    def _get_lock_path(self, table_name: str) -> str:
        return os.path.join(self.base_dir, f"{table_name}.lock")

    def _load_table(self, table_name: str) -> dict:
        path = self._get_table_path(table_name)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, PermissionError):
            return {}

    def _save_table(self, table_name: str, data: dict):
        path = self._get_table_path(table_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _locked_read_modify_write(self, table_name: str, modify_fn):
        """
        Esegue una read-modify-write atomica sulla tabella usando un file lock.
        modify_fn riceve la tabella (dict) e restituisce (tabella_modificata, risultato).
        """
        lock_path = self._get_lock_path(table_name)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                table = self._load_table(table_name)
                table, result = modify_fn(table)
                self._save_table(table_name, table)
                return result
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _get_primary_key_name(self, table_name: str) -> str:
        mapping = {
            'workers_registry': 'worker_name',
            'orchestrators_registry': 'orchestrator_name',
            'ModelStatus': 'job_id',
            'WorkerTasks': 'task_id',
        }
        if table_name not in mapping:
            raise ValueError(f"Tabella '{table_name}' non riconosciuta.")
        return mapping[table_name]

    def put_item(self, table_name: str, key: str, value: dict):
        str_key = str(key)
        def modify(table):
            table[str_key] = value
            return table, None
        self._locked_read_modify_write(table_name, modify)
        info = f"Stato: {value.get('status')}" if "status" in value else f"Dati: {list(value.keys())}"

    def put_item_if_not_exists(self, table_name: str, key: str, value: dict) -> bool:
        """
        Conditional write: scrive solo se la chiave NON esiste già.
        Restituisce True se ha scritto, False se la chiave era già presente.
        Equivale a DynamoDB attribute_not_exists(pk).
        """
        str_key = str(key)
        def modify(table):
            if str_key in table:
                return table, False
            table[str_key] = value
            return table, True
        
        success = self._locked_read_modify_write(table_name, modify)
        if success:
            print(f"[Mock DynamoDB] Tabella '{table_name}' -> Conditional write OK: {str_key[:8]}...")
        else:
            print(f"[Mock DynamoDB] Tabella '{table_name}' -> Conditional write FALLITA (già esiste): {str_key[:8]}...")
        return success

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
        str_key = str(key)
        def modify(table):
            if str_key in table:
                del table[str_key]
                return table, True
            return table, False
        result = self._locked_read_modify_write(table_name, modify)
        if result:
            print(f"[Mock DynamoDB] Tabella '{table_name}' -> Eliminato ID: {str_key[:8]}...")
        return result

    def scan_table(self, table_name: str) -> dict:
        table_data = self._load_table(table_name)
        pk_name = self._get_primary_key_name(table_name)
        items_list = [
            {**value, pk_name: key}
            for key, value in table_data.items()
        ]
        return {"Items": items_list}
    
    def query_by_index(self, table_name: str, index_name: str, key_name: str, key_value: str) -> dict:
        """
        Simula una query su un GSI filtrando in memoria i dati della tabella.
        (index_name viene ignorato nel mock, ma serve per mantenere l'interfaccia 
        identica a quella di AWS).
        """
        table_data = self._load_table(table_name)
        pk_name = self._get_primary_key_name(table_name)
        items_list = []
        
        for key, value in table_data.items():
            # Controlliamo se l'attributo cercato (es. job_id) corrisponde al valore richiesto
            if value.get(key_name) == key_value:
                # Ricostruiamo l'item inserendo la Primary Key esattamente come fa scan_table
                item_compliant = {**value, pk_name: key}
                items_list.append(item_compliant)
                
        return {"Items": items_list}

dynamo_db = MockDynamoDB()