import json
import os
import fcntl
from typing import Optional
import time

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

    def _locked_read(self, table_name: str) -> dict:
        """
        Lettura protetta da lock CONDIVISO (LOCK_SH). _save_table apre il
        file dati in modalità 'w', che lo TRONCA a zero byte prima di
        riscriverlo per intero: una lettura non protetta che capita in
        quella finestra vede un file vuoto/a metà scritto e interpreta
        (erroneamente) una chiave esistente come assente.

        PRIMA get_item leggeva con _load_table() diretto, senza alcun lock
        -- unico metodo di questa classe a farlo, mentre tutte le scritture
        (put_item, delete_item, try_acquire_lock, ...) passano già da
        _locked_read_modify_write con LOCK_EX. Causa diretta dei warning
        intermittenti "Heartbeat ignorato: ... non risulta registrato" visti
        nei log quando più worker si registrano/aggiornano quasi
        simultaneamente: la lettura del proprio heartbeat capitava,
        occasionalmente, proprio mentre un ALTRO worker stava scrivendo sullo
        stesso file di tabella condiviso.

        LOCK_SH invece di LOCK_EX: più letture concorrenti possono procedere
        insieme (non si bloccano a vicenda), sono bloccate solo da uno
        scrittore attivo (LOCK_EX) -- semantica reader/writer standard,
        compatibile con gli scrittori esistenti senza modificarli.
        """
        lock_path = self._get_lock_path(table_name)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_SH)
            try:
                return self._load_table(table_name)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _get_primary_key_name(self, table_name: str) -> str:
        mapping = {
            'workers_registry': 'worker_name',
            'orchestrators_registry': 'orchestrator_name',
            'ModelStatus': 'job_id',
            'WorkerTasks': 'task_id',
            'OrchestratorLocks': 'lock_key',
            'JobLocks': 'lock_key',
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
        table = self._locked_read(table_name)
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
        # Stesso fix di get_item sopra (vedi _locked_read): senza lock
        # condiviso, uno scan poteva capitare a metà di una scrittura
        # concorrente su questo stesso file (es. un altro worker che si
        # registra) e restituire una tabella momentaneamente vuota/troncata
        # -- rilevante qui perché get_available_workers si basa proprio su
        # scan_table per decidere quali worker sono disponibili.
        table_data = self._locked_read(table_name)
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
        # Stesso fix di get_item/scan_table sopra (vedi _locked_read).
        table_data = self._locked_read(table_name)
        pk_name = self._get_primary_key_name(table_name)
        items_list = []
        
        for key, value in table_data.items():
            # Controlliamo se l'attributo cercato (es. job_id) corrisponde al valore richiesto
            if value.get(key_name) == key_value:
                # Ricostruiamo l'item inserendo la Primary Key esattamente come fa scan_table
                item_compliant = {**value, pk_name: key}
                items_list.append(item_compliant)
                
        return {"Items": items_list}
    def try_acquire_lock(self, table_name: str, lock_key: str, owner: str, ttl: int = 30) -> bool:
        """
        Acquisisce il lock in modo atomico SOLO se: non esiste ancora,
        oppure la lease precedente è scaduta. Equivale a:
        ConditionExpression="attribute_not_exists(lock_key) OR expires_at < :now"
        """
        now = time.time()
        expires_at = now + ttl
 
        def modify(table):
            current = table.get(lock_key)
            if current and current.get("expires_at", 0) >= now:
                return table, False  # lock ancora valido, posseduto da qualcun altro
            table[lock_key] = {"leader": owner, "expires_at": expires_at, "timestamp": now}
            return table, True
 
        acquired = self._locked_read_modify_write(table_name, modify)
        if acquired:
            print(f"[Mock DynamoDB] Lock '{lock_key}' su '{table_name}' acquisito da {owner} (ttl={ttl}s).")
        return acquired
 
    def refresh_lock(self, table_name: str, lock_key: str, owner: str, ttl: int = 30) -> bool:
        """Rinnova un lock SOLO se il possessore attuale coincide con owner."""
        now = time.time()
        expires_at = now + ttl
 
        def modify(table):
            current = table.get(lock_key)
            if not current or current.get("leader") != owner:
                return table, False
            current["expires_at"] = expires_at
            current["timestamp"] = now
            table[lock_key] = current
            return table, True
 
        return self._locked_read_modify_write(table_name, modify)
 
    def release_lock(self, table_name: str, lock_key: str, owner: str) -> bool:
        """Rilascia il lock SOLO se il possessore attuale coincide con owner."""
        def modify(table):
            current = table.get(lock_key)
            if not current or current.get("leader") != owner:
                return table, False
            del table[lock_key]
            return table, True
 
        return self._locked_read_modify_write(table_name, modify)

dynamo_db = MockDynamoDB()