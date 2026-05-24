import time

class MockSQSQueue:
    def __init__(self):
        self.queue = [] #Lista dei messaggi visibili
        self._in_flight ={} #Dizionario dei messaggi in lavorazione {job_id: (message, visibility_timeout_end_time)}
    
    def send_message(self, message_dict: dict):
        if "job_id" not in message_dict:
            raise ValueError("[MOCK SQS]: Il messaggio deve contenere un 'job_id' univoco.")
        self.queue.append(message_dict)
        print(f"[MOCK SQS] Messaggio inviato - Job ID: {message_dict['job_id']}")

    def receive_message(self, visibility_timeout: int = 30):
        now = time.time()

        # Rimuoviamo i messaggi il cui timeout è scaduto e li rendiamo nuovamente visibili
        for job_id, data in list(self._in_flight.items()):
            if now >= data["time_out"]: #Se il timeout è scaduto
                print(f"[MOCK SQS] Timeout scaduto per Job ID: {job_id}. Il messaggio è nuovamente visibile.")
                self.queue.append(data["message"]) #Rendi il messaggio nuovamente visibile
                del self._in_flight[job_id] #Rimuovi dal dizionario dei messaggi in lavorazione

        if not self.queue:
            return None #Nessun messaggio disponibile
        
        #Estrazione FIFO del messaggio
        msg = self.queue.pop(0) #Prendi il primo messaggio disponibile
        job_id = msg["job_id"]

        #Il messaggio diventa invisibile per un certo periodo di tempo
        self._in_flight[job_id] = {
            "message": msg,
            "time_out": now + visibility_timeout
        }

        return msg
    
    def delete_message(self, job_id: str):
        if job_id in self._in_flight:
            del self._in_flight[job_id]
            print(f"[MOCK SQS] Messaggio con Job ID: {job_id} eliminato dalla coda.")
        else:
            print(f"[MOCK SQS] Nessun messaggio in lavorazione con Job ID: {job_id} trovato.")

sqs_queue = MockSQSQueue()
