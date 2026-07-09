import json
import os
import time
import uuid

from Distributed_RandomForest.src.shared.mock_aws.statemanager.interfaces import SQSQueueInterface

class MockSQSQueue(SQSQueueInterface):
    def __init__(self):
        # Legge da env, fallback alla cartella locale per retrocompatibilità
        base = os.environ.get("LOCAL_STORAGE_PATH", os.path.join(".", ".local_storage"))
        storage_dir = os.path.abspath(base)
        os.makedirs(storage_dir, exist_ok=True)
        
        self.file_path = os.path.join(storage_dir, "sqs_state.json")
        
        # Se il file non esiste ancora, lo inizializziamo con due code separate e gli in-flight
        if not os.path.exists(self.file_path):
            self._save_state({
                "centralized_queue": [],
                "federated_queue": [],
                "in_flight": {}  # Struttura: {receipt_handle: {"queue_name": str, "message": dict, "time_out": float}}
            })

    def _load_state(self) -> dict:
        """Legge lo stato corrente della coda dal file JSON (Versione Sicura)."""
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            return {"centralized_queue": [], "federated_queue": [], "in_flight": {}}

        # Proviamo a leggere il file al massimo 5 volte per gestire la concorrenza
        for attempt in range(5):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # Se il file è temporaneamente vuoto o bloccato da un altro processo, aspetta
                time.sleep(0.05)
        
        print("[MOCK SQS - WARNING] File sqs_state.json corrotto detectato. Reset dello stato della coda.")
        return {"centralized_queue": [], "federated_queue": [], "in_flight": {}}

    def _save_state(self, state: dict):
        """Salva lo stato aggiornato della coda nel file JSON con retry concorrenziale."""
        for attempt in range(5):
            try:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                return
            except (PermissionError, IOError):
                time.sleep(0.05)
        print("[MOCK SQS - ERRORE CRITICO] Impossibile scrivere lo stato di SQS su disco.")

    def send_message(self, queue_name: str, message_dict: dict) -> None:
        """Invia il messaggio a una coda specifica (centralized_queue o federated_queue)."""
        if "job_id" not in message_dict:
            raise ValueError("[MOCK SQS]: Il messaggio deve contenere un 'job_id' univoco.")
        
        state = self._load_state()
        
        if queue_name not in state or queue_name == "in_flight":
            raise ValueError(f"[MOCK SQS]: La coda '{queue_name}' non è valida.")
        
        state[queue_name].append(message_dict)
        self._save_state(state)
        
        print(f"[MOCK SQS] Messaggio registrato in '{queue_name}' - Job ID: {message_dict['job_id'][:8]}...")

    def receive_message(self, queue_name: str, visibility_timeout: int = 300) -> dict | None:
        """Fa polling selettivo e gestisce istantaneamente il riciclo dei messaggi scaduti."""
        state = self._load_state()
        now = time.time()
        updated = False

        for receipt_handle, data in list(state["in_flight"].items()):
            if data["queue_name"] == queue_name and now >= data["time_out"]:
                print(f"[MOCK SQS] Visibility Timeout SCADUTO in '{queue_name}'. Il messaggio torna visibile.")
                state[queue_name].append(data["message"])
                del state["in_flight"][receipt_handle]
                updated = True

        if queue_name not in state or not state[queue_name]:
            if updated: 
                self._save_state(state)
            return None
        
        msg = state[queue_name].pop(0)
        new_receipt_handle = f"MB_RECEIPT_{str(uuid.uuid4())[:8]}"

        state["in_flight"][new_receipt_handle] = {
            "queue_name": queue_name,
            "message": msg,
            "time_out": now + visibility_timeout
        }

        self._save_state(state)

        return {
            "Body": msg,
            "ReceiptHandle": new_receipt_handle
        }
    
    def delete_message(self, receipt_handle: str) -> bool:
        """Elimina il messaggio usando il ReceiptHandle fornito dall'Orchestrator."""
        state = self._load_state()
        
        if receipt_handle in state["in_flight"]:
            job_id = state["in_flight"][receipt_handle]["message"]["job_id"]
            queue_source = state["in_flight"][receipt_handle]["queue_name"]
            
            del state["in_flight"][receipt_handle]
            self._save_state(state)
            print(f"[MOCK SQS] [OK] Messaggio eliminato da '{queue_source}' tramite ReceiptHandle: {receipt_handle} (Job ID: {job_id[:8]}...)")
            return True
        else:
            print(f"[MOCK SQS] [ERRORE CANCELLAZIONE] ReceiptHandle non valido o scaduto: {receipt_handle}")
            return False
    def change_message_visibility(self, queue_name: str, receipt_handle: str, visibility_timeout: int) -> bool:
        """
        Modifica ed estende il Visibility Timeout di un messaggio attualmente in-flight.
        Simula il comportamento dell'omonima API di AWS SQS.
        """
        state = self._load_state()
        now = time.time()
        
        # Verifichiamo se il messaggio è ancora in elaborazione ed appartiene alla coda corretta
        if receipt_handle in state["in_flight"]:
            # Aggiorniamo il timestamp di scadenza: ora attuale + nuova estensione
            state["in_flight"][receipt_handle]["time_out"] = now + visibility_timeout
            
            # Salviamo lo stato aggiornato in modo concorrenziale su JSON
            self._save_state(state)
            
            job_id = state["in_flight"][receipt_handle]["message"].get("job_id", "unknown")
            print(f"[MOCK SQS] [HEARTBEAT OK] Visibilità estesa per {receipt_handle} (Job ID: {job_id[:8]}...) di altri {visibility_timeout}s.")
            return True
        else:
            # Caso in cui il messaggio potrebbe essere già stato eliminato o scaduto prima del heartbeat
            print(f"[MOCK SQS] [HEARTBEAT WARN] Impossibile aggiornare la visibilità. ReceiptHandle non trovato: {receipt_handle}")
            return False

# Istanza globale esportata per la Factory polimorfa
sqs_queue = MockSQSQueue()