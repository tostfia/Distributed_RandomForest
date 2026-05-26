import json
import os
import time
import uuid

from src.shared.mock_aws.interfaces import SQSQueueInterface

class MockSQSQueue(SQSQueueInterface):
    def __init__(self):
        # Definiamo il percorso del file JSON nella stessa cartella del mock
        self.file_path = os.path.join(os.path.dirname(__file__), "sqs_state.json")
        
        # Se il file non esiste ancora, lo inizializziamo con due code separate e gli in-flight
        if not os.path.exists(self.file_path):
            self._save_state({
                "centralized_queue": [],
                "federated_queue": [],
                "in_flight": {}  # Struttura: {receipt_handle: {"queue_name": str, "message": dict, "time_out": float}}
            })

    def _load_state(self) -> dict:
        """Legge lo stato corrente della coda dal file JSON."""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except json.JSONDecodeError:
            # Protezione in caso di lettura concorrente fallita (file temporaneamente vuoto o bloccato)
            time.sleep(0.05)
            return self._load_state()
        return {"centralized_queue": [], "federated_queue": [], "in_flight": {}}

    def _save_state(self, state: dict):
        """Salva lo stato aggiornato della coda nel file JSON."""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def send_message(self, queue_name: str, message_dict: dict) -> None:
        """Invia il messaggio a una coda specifica (centralized_queue o federated_queue)."""
        if "job_id" not in message_dict:
            raise ValueError("[MOCK SQS]: Il messaggio deve contenere un 'job_id' univoco.")
        
        state = self._load_state()
        
        # Controllo di sicurezza se la coda passata non esiste nel JSON
        if queue_name not in state or queue_name == "in_flight":
            raise ValueError(f"[MOCK SQS]: La coda '{queue_name}' non è valida.")
        
        state[queue_name].append(message_dict)
        self._save_state(state)
        
        print(f"[MOCK SQS] Messaggio registrato in '{queue_name}' - Job ID: {message_dict['job_id'][:8]}...")

    def receive_message(self, queue_name: str, visibility_timeout: int = 30) -> dict | None:
        """Fa polling selettivo e gestisce istantaneamente il riciclo dei messaggi scaduti."""
        state = self._load_state()
        now = time.time()
        updated = False

        # 1. Controllo dei timeout specifico per la coda che sta chiamando
        for receipt_handle, data in list(state["in_flight"].items()):
            if data["queue_name"] == queue_name and now >= data["time_out"]:
                print(f"[MOCK SQS] Visibility Timeout SCADUTO in '{queue_name}'. Il messaggio torna visibile.")
                state[queue_name].append(data["message"])
                del state["in_flight"][receipt_handle]
                updated = True

        # CORREZIONE: Se dopo il controllo dei timeout la coda è ANCORA disperatamente vuota, allora esci
        if queue_name not in state or not state[queue_name]:
            if updated: 
                self._save_state(state)  # Salva lo sblocco sul JSON anche se la coda era vuota per davvero
            return None
        
        # 2. Estrazione FIFO dalla coda specifica (ora prenderà subito il messaggio appena scaduto!)
        msg = state[queue_name].pop(0)

        # 3. GENERAZIONE DEL RECEIPT HANDLE
        new_receipt_handle = f"MB_RECEIPT_{str(uuid.uuid4())[:8]}"

        # Spostamento in-flight ricordando da quale coda proviene il messaggio
        state["in_flight"][new_receipt_handle] = {
            "queue_name": queue_name,
            "message": msg,
            "time_out": now + visibility_timeout
        }

        self._save_state(state)

        # Restituiamo la struttura standard AWS Boto3
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
            
            # Rimuoviamo il messaggio validando il token di sblocco
            del state["in_flight"][receipt_handle]
            self._save_state(state)
            print(f"[MOCK SQS] [OK] Messaggio eliminato da '{queue_source}' tramite ReceiptHandle: {receipt_handle} (Job ID: {job_id[:8]}...)")
            return True
        else:
            print(f"[MOCK SQS] [ERRORE CANCELLAZIONE] ReceiptHandle non valido o scaduto: {receipt_handle}")
            return False

# Istanza globale esportata per la Factory polimorfa
sqs_queue = MockSQSQueue()