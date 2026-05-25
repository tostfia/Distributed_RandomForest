import json
import os
import time
import uuid

class MockSQSQueue:
    def __init__(self):
        # Definiamo il percorso del file JSON nella stessa cartella del mock
        self.file_path = os.path.join(os.path.dirname(__file__), "sqs_state.json")
        
        # Se il file non esiste ancora, lo inizializziamo con la struttura vuota
        if not os.path.exists(self.file_path):
            self._save_state({"queue": [], "in_flight": {}})

    def _load_state(self) -> dict:
        """Legge lo stato corrente della coda dal file JSON."""
        try:
            if os.path.exists(self.file_path):
                with open(path := self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except json.JSONDecodeError:
            # Protezione in caso di lettura concorrente fallita (file temporaneamente vuoto o bloccato)
            time.sleep(0.05)
            return self._load_state()
        return {"queue": [], "in_flight": {}}

    def _save_state(self, state: dict):
        """Salva lo stato aggiornato della coda nel file JSON."""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def send_message(self, message_dict: dict):
        if "job_id" not in message_dict:
            raise ValueError("[MOCK SQS]: Il messaggio deve contenere un 'job_id' univoco.")
        
        # Carichiamo lo stato dal file, appendiamo il messaggio e salviamo
        state = self._load_state()
        state["queue"].append(message_dict)
        self._save_state(state)
        
        print(f"[MOCK SQS] Messaggio registrato in coda - Job ID: {message_dict['job_id'][:8]}...")

    def receive_message(self, visibility_timeout: int = 30):
        state = self._load_state()
        now = time.time()
        updated = False

        # 1. Controllo dei timeout usando il receipt_handle come chiave di scansione
        for receipt_handle, data in list(state["in_flight"].items()):
            if now >= data["time_out"]:
                print(f"[MOCK SQS] ⚠️ Visibility Timeout SCADUTO per un messaggio in-flight. Torna visibile.")
                state["queue"].append(data["message"])
                del state["in_flight"][receipt_handle]
                updated = True

        # Se non ci sono messaggi disponibili nella coda
        if not state["queue"]:
            if updated: 
                self._save_state(state)  # Salva se abbiamo ripristinato messaggi scaduti
            return None
        
        # 2. Estrazione FIFO dal file
        msg = state["queue"].pop(0)

        # 3. GENERAZIONE DEL RECEIPT HANDLE (Opacità del Middleware richiesta dal Prof)
        # Generiamo un token di ricezione dinamico e temporaneo
        new_receipt_handle = f"MB_RECEIPT_{str(uuid.uuid4())[:8]}"

        # Spostamento in-flight usando il RECEIPT HANDLE come chiave di blocco (lock)
        state["in_flight"][new_receipt_handle] = {
            "message": msg,
            "time_out": now + visibility_timeout
        }

        # Salviamo le modifiche sul file prima di ritornare il messaggio
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
            
            # Rimuoviamo il messaggio validando il token di sblocco
            del state["in_flight"][receipt_handle]
            self._save_state(state)
            print(f"[MOCK SQS] [OK] Messaggio eliminato tramite ReceiptHandle: {receipt_handle} (Job ID applicativo: {job_id[:8]}...)")
            return True
        else:
            print(f"[MOCK SQS] [ERRORE CANCELLAZIONE] ReceiptHandle non valido o scaduto: {receipt_handle}")
            return False

# Istanza globale
sqs_queue = MockSQSQueue()