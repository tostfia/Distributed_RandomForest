from abc import ABC, abstractmethod
import os
import gc
import ctypes
import signal
import socket
import numpy as np
from rpyc import Service, ThreadedServer
import threading
import time
import pickle
import boto3
import json
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from src.shared.config import SystemConfig
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.utilities.task_storage import (
    load_bytes_from_shared_storage,
    load_task_from_shared_storage,
    save_task_part_to_shared_storage,
    save_task_manifest,
    iter_chunk_parts_from_shared_storage,
)

# Timeout (in secondi) per il completamento di un singolo batch di alberi nel
# ThreadPool. Stesso pattern/nome di FederatedWorker.RPC_SYNC_TIMEOUT_SECONDS,
# ma default 600 per restare coerente col 'sync_request_timeout' già usato in
# start_server (protocol_config) per il path centralizzato.
RPC_SYNC_TIMEOUT_SECONDS = int(os.environ.get("RPC_SYNC_TIMEOUT_SECONDS", 600))

_child_X = None
_child_y = None


def _release_memory_to_os():
    """gc.collect() da solo libera gli oggetti Python non più referenziati,
    ma CPython/glibc spesso NON restituisce quella memoria al sistema
    operativo -- la tiene in riserva per riusarla internamente. L'RSS visto
    da 'docker stats'/cgroup può quindi restare alto anche quando dentro il
    processo non è rimasto nulla di vivo. 'malloc_trim(0)' (glibc, Linux)
    chiede esplicitamente all'allocatore di restituire i blocchi liberi
    all'OS: è quello che chiude il cerchio tra "l'ho liberato in Python" e
    "il container vede meno RAM usata". No-op innocuo se non c'è nulla da
    restituire, e silenziosamente ignorato su piattaforme non-glibc (es.
    macOS in sviluppo locale) tramite l'except sotto.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# Addestramento di un singolo albero, eseguito da un thread del ThreadPool.
# A differenza della precedente versione a processi (multiprocessing.Pool),
# qui _child_X/_child_y sono semplici variabili globali del modulo: i thread
# condividono già la memoria del processo, quindi non serve alcun initializer
# né alcuna copia pickle-ata di X/y per ciascun worker (che con un Pool a
# processi veniva invece duplicata integralmente per OGNI processo,
# moltiplicando l'uso di RAM per pool_size). Vedi lo stesso identico
# ragionamento in FederatedWorker._train_single_fed_tree.
def _train_single_tree_thread(args):
    global _child_X, _child_y
    tree_seed, max_depth, max_samples, bootstrap, tree_class, max_features, min_samples_split, class_weight, criterion = args
    np.random.seed(tree_seed)

    n_samples = _child_X.shape[0]

    if bootstrap:
        size = int(max_samples * n_samples) if max_samples else n_samples
        indices = np.random.choice(n_samples, size=size, replace=True)
        # Zero-copy: invece di X_train = _child_X[indices] (una copia fisica
        # per albero), calcoliamo quante volte ogni riga originale è stata
        # estratta e passiamo questo peso a tree.fit(). Matematicamente
        # equivalente al duplicare le righe (stesso identico calcolo di
        # impurità Gini/MSE che sklearn fa internamente in
        # _parallel_build_trees), ma senza mai allocare una copia fisica
        # dell'array.
        sample_weight = np.bincount(indices, minlength=n_samples).astype(np.float64)
        # Campioni MAI estratti per questo albero (peso 0, ~36.8% atteso con
        # size=n_samples): li conserviamo per poter stimare l'errore
        # Out-Of-Bag "gratis" più avanti, senza dover consumare il test set
        # separato (Breiman, 2001).
        oob_indices = np.flatnonzero(sample_weight == 0)
    else:
        sample_weight = None
        # Senza bootstrap ogni albero vede l'intero training set: non esiste
        # un sottoinsieme "mai visto" su cui stimare l'OOB.
        oob_indices = np.array([], dtype=np.int64)

    X_fit, y_fit = _child_X, _child_y

    # max_features attiva il sottocampionamento casuale delle feature ad ogni
    # split: è ciò che decorrela gli alberi tra loro (Breiman, 2001) e
    # distingue un vero Random Forest da un semplice bagging di alberi.
    # random_state passato esplicitamente invece di affidarsi solo al seed
    # globale np.random.seed sopra, per coerenza col path federato.
    tree_kwargs = dict(
        splitter="best",
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        random_state=tree_seed,
    )
    if criterion is not None:
        tree_kwargs["criterion"] = criterion
    # class_weight è valido solo per gli alberi di classificazione
    if class_weight is not None and "Classifier" in tree_class.__name__:
        tree_kwargs["class_weight"] = class_weight

    tree = tree_class(**tree_kwargs)
    tree.fit(X_fit, y_fit, sample_weight=sample_weight)
    # Attributo "extra" sull'istanza sklearn: sopravvive al pickle esattamente
    # come classes_/n_features_in_, quindi arriva intatto fino all'Orchestratore
    # senza dover cambiare la struttura dati (tree object) che viaggia in RPC.
    tree.oob_sample_indices_ = oob_indices
    return tree

class BaseWorker(Service, ABC):
    def __init__(
        self,
        worker_name: str,
        queue_name: str,
        tree_class_reference,
        url_dataset: str = None,
        max_samples=None,
        bootstrap: bool = True
    ):
        super().__init__()
        # 1. Carichiamo la configurazione centralizzata dal file .env
        self.cfg = SystemConfig()
        self.environment = self.cfg.env

        self.worker_name = worker_name
        self.queue_name = queue_name
        self.url_dataset = url_dataset
        self.tree_class_reference = tree_class_reference
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self._stop_heartbeat = None

        self._cached_X_test_bytes = None
        self._cached_X_eval = None


    @abstractmethod
    def is_regression(self):
        pass

    def release_index_claim(self):
        """Hook per sottoclassi che gestiscono claim di risorse condivise
        (es. FederatedWorker con l'indice shard su AWS). Implementazione di
        base: nessuna azione (usata da CentralizedWorker)."""
        pass

    def _get_my_private_ip(self) -> str:
        """Determina l'IP corretto per il binding di rete in base all'ambiente.
            Funziona sia in locale (con/senza Docker) sia su AWS.
        """
        # Se siamo dentro Docker (Compose imposta solitamente variabili o hostname specifici)
        # o se siamo su AWS, dobbiamo ascoltare su tutte le interfacce (0.0.0.0)
        if self.environment == "aws" or os.environ.get("RUNNING_IN_DOCKER", "false") == "true":
            return "0.0.0.0"

        # Locale puro senza Docker
        return "127.0.0.1"

    def start_server(self, port: int, explicit_host: str = None):
        print(f"\n[{self.worker_name}] Inizializzazione Server RPC in ambiente {self.environment.upper()}...")

        def _handle_sigterm(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, _handle_sigterm)

        advertise_host = os.environ.get("RPC_ADVERTISE_HOST", None)
        is_docker = os.environ.get("RUNNING_IN_DOCKER", "false") == "true"

        # Gestione degli host dinamica
        if self.environment == "aws" or is_docker:
            host_to_bind = "0.0.0.0"  # Permette a RPyC di accettare connessioni esterne/da altri container
            # Se siamo in Docker e non c'è un advertise_host esplicito, usiamo il socket hostname (il nome del container)
            host_to_register = advertise_host if advertise_host else socket.gethostname()
        else:
            # Locale nativo senza Docker
            host_to_bind = explicit_host if explicit_host else "127.0.0.1"
            host_to_register = host_to_bind

        print(f"[{self.worker_name}] Binding su: {host_to_bind}, Registrazione su Registry come: {host_to_register}:{port}")

        # Registrazione del Worker sul Service Registry
        ServiceRegistry.register_worker(worker_name=self.worker_name, host=host_to_register, port=port)

        self._stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, args=(self._stop_heartbeat, 10), daemon=True)
        heartbeat_thread.start()
        print(f"[+] [{self.worker_name}] Thread di Heartbeat avviato con successo.")

        protocol_config = {
            'allow_public_attr': True,
            'allow_pickle': True,
            'sync_request_timeout': 600,
            'keepalive': True
        }

        server = ThreadedServer(self, hostname=host_to_bind, port=port, protocol_config=protocol_config)

        print("\n ==============================================")
        print(f"  SERVER WORKER IN ASCOLTO: {self.worker_name.upper()}")
        print(f"  Indirizzo di ascolto:    {host_to_bind}:{port}")
        print(f"  Ambiente attivo:          {self.environment.upper()}")
        print(" ==============================================\n")

        try:
            server.start()
        except KeyboardInterrupt:
            print(f"\n[-][{self.worker_name} Interruzione manuale rilevata]")
        except Exception as e:
            print(f"\n[!] [{self.worker_name}] Errore durante l'esecuzione del server: {str(e)}")
        finally:
            print(f"\n[+] [{self.worker_name}] Arresto del server in corso...")
            self._stop_heartbeat.set()
            heartbeat_thread.join(timeout=2)

            self.release_index_claim()

            try:
                ServiceRegistry.deregister_worker(self.worker_name)
                print(f"[+] [{self.worker_name}] Server arrestato e worker rimosso dal Service Registry.")
            except Exception as e:
                print(f"[!] [{self.worker_name}] Errore durante la deregistrazione: {str(e)}")

    def _heartbeat_loop(self, stop_event: threading.Event, interval: int = 10):
        while not stop_event.is_set():
            try:
                ServiceRegistry.update_worker_heartbeat(self.worker_name)
            except Exception as e:
                print(f"[!] [{self.worker_name}] Errore durante l'invio dell'heartbeat: {str(e)}")
            for _ in range(interval):
                if stop_event.is_set():
                    break
                time.sleep(1)

    def on_connect(self, conn):
        peer_info = "Orchestratore"
        if hasattr(conn, '_config') and 'peer' in conn._config:
            peer_info = conn._config['peer']
        print(f"[+] Connessione stabilita con successo: {peer_info}")

    def on_disconnect(self, conn):
        print(f"[-] Connessione chiusa dall'Orchestratore.")

    def exposed_ping(self):
        """
        Endpoint RPC leggero, senza alcun accesso a dataset/ETL/training:
        serve esclusivamente a misurare la latenza di rete/RPyC pura tra
        Orchestratore e Worker (vedi BaseOrchestrator._measure_rpc_ping_stats).
        A differenza di exposed_train_subset_forest, qui il tempo di risposta
        riflette SOLO il round-trip RPC, non il tempo di preparazione dati.
        """
        return "pong"

    @abstractmethod
    def _load_data(self, source_info):
        pass

    @abstractmethod
    def _get_tree_class(self):
        pass

    def exposed_train_subset_forest(self, source_info, num_trees, base_seed, max_depth=None, tree_type=None, max_features=None,
                                     min_samples_split=2, class_weight=None, criterion=None,
                                     bootstrap=None, max_samples=None, compute_oob=False):
        print("\n=============================================================")
        print(f" [WORKER RPC] Richiesta elaborazione foresta parziale | Alberi: {num_trees}")
        print("=============================================================\n")
        if tree_type is not None:
            self.tree_type = tree_type
        # Se l'Orchestratore non specifica max_features (manifesti vecchi/non
        # aggiornati), ricadiamo sui default "corretti" di un vero Random
        # Forest invece che su None (= tutte le feature ad ogni split).
        if max_features is None:
            max_features = "sqrt" if not self.is_regression() else (1 / 3)

        # bootstrap/max_samples: fino ad ora venivano presi ESCLUSIVAMENTE dai
        # valori di boot del worker (self.bootstrap / self.max_samples), quindi
        # qualunque cosa dichiarasse il manifesto o la TrainingRequest veniva
        # ignorata — inclusa la forzatura bootstrap=False della modalità
        # federata, che di fatto non aveva alcun effetto sugli alberi.
        # Ora sono parametri OPZIONALI: None = "non specificato dal chiamante",
        # e in quel caso si mantengono i valori di boot, quindi il
        # comportamento di qualunque chiamante esistente resta identico a prima.
        effective_bootstrap = self.bootstrap if bootstrap is None else bootstrap
        effective_max_samples = self.max_samples if max_samples is None else max_samples
        print(f"[{self.worker_name}] Campionamento: bootstrap={effective_bootstrap}, "
              f"max_samples={effective_max_samples} "
              f"({'da richiesta' if bootstrap is not None else 'da configurazione di boot'}).")
        cached_task_bytes = load_task_from_shared_storage(
            source_info, base_seed, num_trees, self.environment, self.worker_name
        )
        if cached_task_bytes is not None:
            print(f"[{self.worker_name}] [SHORT-CIRCUIT] Task già pronto nello storage. Invio solo ack "
                  f"(l'Orchestratore rilegge il blob direttamente dallo storage condiviso).")
            return {"ack": True, "num_trees": num_trees}
        # 1. Recupero dati e classe dell'albero dalle classi figlie
        X, y = self._load_data(source_info)
        tree_class = self._get_tree_class()

        # OOB DISTRIBUITA (12/9/2026): accumulo locale dei contributi OOB man
        # mano che ogni albero viene costruito, invece di lasciare che
        # l'Orchestratore rifaccia .predict() su OGNI albero in sequenza a
        # fine training (collo di bottiglia misurato: 146-159s costanti,
        # indipendenti dal numero di worker - la stima OOB oggi non scala
        # affatto). Ogni worker calcola qui i propri contributi PARZIALI
        # (oob_sum/oob_count, entrambi di lunghezza pari alle righe di X)
        # usando gli alberi appena costruiti e X/y che ha gia' in memoria -
        # zero trasferimento dati aggiuntivo. L'Orchestratore sommera' i
        # contributi di tutti i task (vedi BaseOrchestrator._compute_oob_metrics_distributed).
        #
        # SCOPO LIMITATO DELIBERATAMENTE: solo regressore (tree_type ==
        # "regressor") e solo se il worker espone 'self.dao' (cioe' solo
        # CentralizedWorker - FederatedWorker non ha accesso al training set
        # GLOBALE, solo al proprio shard, quindi non puo' contribuire a una
        # stima OOB GLOBALE nello stesso modo). Il ramo classificatore
        # richiederebbe coordinare lo spazio di classi globale tra worker
        # PRIMA di poter sommare (ogni worker potrebbe vedere classi
        # diverse nei propri alberi) - non affrontato in questa iterazione,
        # ricade sul metodo sequenziale esistente (skip_oob invariato).
        _oob_distributed_enabled = compute_oob and (tree_type == "regressor") and hasattr(self, "dao")
        if _oob_distributed_enabled:
            oob_sum_local = np.zeros(X.shape[0], dtype=np.float64)
            oob_count_local = np.zeros(X.shape[0], dtype=np.int64)

        # 2. CALCOLO DINAMICO DEI CORE
        # Su ECS Fargate ogni task worker ha la propria CPU DEDICATA E ISOLATA
        # (quella assegnata con WORKER_CPU nella task definition in Terraform (ecs_task_definitions.tf)):
        # non condivide MAI la macchina fisica con gli altri worker del cluster,
        # indipendentemente da quanti risultano registrati nel ServiceRegistry.
        # La divisione dei core "per co-locazione" ha senso SOLO in locale/Docker
        # Compose, dove più container worker girano davvero sulla stessa macchina
        # fisica e si contendono gli stessi core. Su AWS usiamo quindi sempre
        # tutta la CPU disponibile localmente al task, senza dividerla per il
        # numero di worker attivi nel fleet (che sono isolati gli uni dagli altri).
        totale_core_macchina = os.cpu_count() or 1

        # Override esplicito per esperimenti di strong scaling: con WORKER_CORES
        # impostata, questo worker usa SEMPRE quel numero fisso di processi,
        # indipendentemente da quanti worker sono attivi sulla stessa macchina.
        # Serve a far sì che ogni worker rappresenti 1 unità di calcolo
        # comparabile a T_seq/T_1node della baseline: senza questo override, il
        # calcolo dinamico sotto tiene volutamente costante la capacità TOTALE
        # del cluster (si ridivide tra i worker attivi), e lo speedup misurato
        # in locale/Docker resta piatto per costruzione qualunque sia il
        # numero di worker.
        _worker_cores_override = os.environ.get("WORKER_CORES")
        allocated_cores = None
        if _worker_cores_override:
            try:
                allocated_cores = max(1, int(_worker_cores_override))
                print(f"[{self.worker_name}] [LOG] WORKER_CORES={allocated_cores} (override esplicito attivo, calcolo dinamico bypassato).")
            except ValueError:
                print(f"[{self.worker_name}] [WARN] WORKER_CORES='{_worker_cores_override}' non è un intero valido: ignorato, ricado sul calcolo dinamico.")

        if allocated_cores is None:
            if self.environment == "aws":
                allocated_cores = max(1, totale_core_macchina - 1) if totale_core_macchina > 2 else totale_core_macchina
            else:
                try:
                    workers_attivi = ServiceRegistry.get_available_workers(self.environment)
                    num_workers = max(1, len(workers_attivi))

                    if num_workers > 1:
                        # Più worker rilevati sulla STESSA macchina fisica (locale/Docker
                        # Compose): dividiamo i core disponibili tra tutti quelli
                        # effettivamente attivi, per evitare sovra-allocazione.
                        core_disponibili_rete = max(1, totale_core_macchina - 1)
                        allocated_cores = max(1, int(core_disponibili_rete / num_workers))
                        print(f"[{self.worker_name}] [LOG] Rilevati {num_workers} worker attivi (ambiente: {self.environment}).")
                        print(f"[{self.worker_name}] [LOG] Allocazione dinamica: {allocated_cores} processi per questo pool.")
                    else:
                        # Un solo worker rilevato: presumibilmente ha la macchina tutta per sé.
                        allocated_cores = max(1, totale_core_macchina - 1) if totale_core_macchina > 2 else totale_core_macchina
                except Exception as e:
                    print(f"[!] Errore lettura ServiceRegistry, fallback su N-1: {e}")
                    allocated_cores = max(1, totale_core_macchina - 1) if totale_core_macchina > 2 else totale_core_macchina

        # 3. Addestramento in thread nativi invece che in un Pool di processi:
        # tree.fit() di scikit-learn rilascia il GIL per la maggior parte del
        # calcolo numerico (Cython/NumPy), quindi i thread parallelizzano
        # davvero, senza pagare né la copia pickle-ata di X/y per ogni
        # processo né il costo di avvio/IPC di multiprocessing.Pool. Stesso
        # approccio già validato in FederatedWorker.exposed_train_local_federated_forest.
        global _child_X, _child_y
        _child_X, _child_y = X, y

        worker_tasks = []
        for i in range(num_trees):
            seed = base_seed + i
            worker_tasks.append((seed, max_depth, effective_max_samples, effective_bootstrap, tree_class, max_features,
                                  min_samples_split, class_weight, criterion))

        # PERSISTENZA INCREMENTALE PER BATCH (invece di accumulo in RAM fino
        # alla fine del task): ogni batch di alberi viene serializzato e
        # scritto sullo storage condiviso APPENA PRONTO, poi liberato dalla
        # memoria del worker prima di passare al batch successivo. Prima di
        # questa modifica, 'local_trees' cresceva monotonicamente fino a
        # contenere l'INTERO chunk (es. 34 alberi con max_depth=None su
        # dataset da 1M righe), e il pickle.dumps() finale duplicava
        # temporaneamente quella stessa memoria in forma serializzata: è
        # il picco di RAM più verosimilmente responsabile degli OOM kill
        # osservati sui worker Fargate durante lo scenario di scalabilità.
        #
        # Con lo streaming per batch, il picco di memoria per gli alberi
        # scende da "num_trees alberi" a "~pool_size*4 alberi" (la dimensione
        # di un batch), indipendentemente da quanto è grande il chunk totale
        # assegnato dall'Orchestratore.
        parts_num_trees = []

        def _persist_batch(batch_trees: list, part_idx: int):
            serialized_part = pickle.dumps(batch_trees)
            try:
                save_task_part_to_shared_storage(
                    source_info, base_seed, num_trees, part_idx,
                    serialized_part, self.environment, self.worker_name
                )
            except Exception as e:
                # Stesso principio di prima: se la persistenza fallisce, il task
                # NON deve risultare completato. Propaghiamo l'eccezione così
                # RPyC la inoltra all'Orchestratore, che marca il task FAILED
                # e lo riaccoda (vedi _prepare_data -> worker_thread_consumer).
                print(f"[!] [{self.worker_name}] ERRORE CRITICO: batch {part_idx} calcolato "
                      f"ma il salvataggio della parte nello storage condiviso è fallito. "
                      f"Dettaglio: {e}")
                raise
            parts_num_trees.append(len(batch_trees))

        if num_trees == 1:
            print("[WORKER] Ottimizzazione: 1 solo albero richiesto. Calcolo diretto senza ThreadPool.")
            single_tree = _train_single_tree_thread(worker_tasks[0])
            _persist_batch([single_tree], 0)
        else:
            pool_size = min(num_trees, allocated_cores)
            print(f"[WORKER] Istanziazione ThreadPool locale con {pool_size} thread "
                  f"(memoria condivisa nativa, nessuna copia/serializzazione tra thread)...")

            # FIX MEMORIA: il moltiplicatore 'x4' teneva in RAM, tra un
            # salvataggio incrementale e l'altro, fino a 4 alberi per ogni
            # thread del pool contemporaneamente. Con max_depth=None su
            # dataset grandi (es. 1M righe) un singolo albero non potato può
            # pesare centinaia di MB: con pool_size=1 (comune quando molti
            # worker girano sulla stessa macchina e si dividono pochi core,
            # vedi 'allocated_cores' sopra) questo significava comunque 4
            # alberi "pesanti" vivi insieme, oltre a X/y. Osservato in pratica
            # come causa di OOM sia a livello di singolo container (cgroup)
            # sia, sommando più worker contemporanei, a livello dell'intera
            # macchina host (constraint=CONSTRAINT_NONE / global_oom in
            # dmesg -- quello NON si risolve alzando il mem_limit di un
            # container, perché non è un limite di container ad essere
            # sforato ma la RAM fisica totale). Moltiplicatore configurabile
            # via env (default 1,= un batch grande quanto il parallelismo
            # reale, non un multiplo arbitrario di esso): a parità di
            # 'pool_size' il picco di alberi-in-RAM-insieme scende fino a 4x,
            # al costo di scritture su storage condiviso più frequenti (più
            # batch, ciascuno più piccolo) -- overhead trascurabile per I/O
            # locale/S3 rispetto al rischio di OOM. Alzabile con
            # WORKER_BATCH_MULTIPLIER se la macchina ha RAM abbondante e si
            # preferisce l'I/O più raro.
            batch_multiplier = int(os.environ.get("WORKER_BATCH_MULTIPLIER", 1))
            BATCH_SIZE = max(1, min(pool_size * batch_multiplier, num_trees))
            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                for part_idx, batch_start in enumerate(range(0, len(worker_tasks), BATCH_SIZE)):
                    batch = worker_tasks[batch_start: batch_start + BATCH_SIZE]
                    print(f"[{self.worker_name}] Batch alberi {batch_start}-{batch_start + len(batch)} "
                          f"di {num_trees}...")
                    futures = [executor.submit(_train_single_tree_thread, task) for task in batch]
                    try:
                        batch_trees = [f.result(timeout=RPC_SYNC_TIMEOUT_SECONDS) for f in futures]
                    except TimeoutError:
                        raise RuntimeError(
                            f"[{self.worker_name}] Timeout ({RPC_SYNC_TIMEOUT_SECONDS}s) durante il "
                            f"training parallelo (ThreadPool)."
                        )
                    print(f"[{self.worker_name}] Batch {part_idx} completato "
                          f"({len(batch_trees)} alberi). Salvataggio incrementale su storage condiviso...")
                    _persist_batch(batch_trees, part_idx)

                    if _oob_distributed_enabled:
                        # Non fatale per design: un errore qui non deve MAI
                        # invalidare un batch di alberi gia' persistito con
                        # successo. Se fallisce, il task intero perdera' il
                        # proprio contributo OOB (vedi 'oob_ready' nel return
                        # finale) - l'Orchestratore rileva questo caso e
                        # ricade sul metodo sequenziale per l'INTERO job,
                        # non solo per questo task (nessun mix parziale).
                        try:
                            for _t in batch_trees:
                                _oob_idx = getattr(_t, "oob_sample_indices_", None)
                                if _oob_idx is not None and len(_oob_idx) > 0:
                                    oob_sum_local[_oob_idx] += _t.predict(X[_oob_idx])
                                    oob_count_local[_oob_idx] += 1
                        except Exception as e_oob_acc:
                            print(f"[{self.worker_name}] [OOB-WARN] Accumulo OOB locale fallito "
                                  f"sul batch {part_idx} ({e_oob_acc}) - il task prosegue comunque, "
                                  f"il contributo OOB di questo worker verra' scartato "
                                  f"dall'Orchestratore.")
                            _oob_distributed_enabled = False

                    # A questo punto 'batch_trees' esce di scope alla prossima
                    # iterazione: gli alberi già scritti su storage non restano
                    # più referenziati da nessuna struttura dati del worker.
                    del batch_trees
                    # BUG SOSPETTATO E CORRETTO (7/9/2026): 'del' rimuove solo il
                    # riferimento, non garantisce la liberazione immediata da
                    # parte del garbage collector ciclico (gli alberi
                    # scikit-learn possono avere riferimenti ciclici interni).
                    # Stesso fix applicato al percorso federato (vedi
                    # FederatedWorker.exposed_train_local_federated_forest per
                    # il pattern di crash osservato empiricamente che ha
                    # motivato questa correzione).
                    _release_memory_to_os()

        # Il manifest viene scritto per ULTIMO, dopo che TUTTE le parti sono
        # sul disco/S3: la sua presenza è ciò che segnala all'Orchestratore
        # (o a un futuro short-circuit di questo stesso worker) che il task è
        # completo e ricomponibile. Contiene solo interi, quindi il suo
        # costo di memoria è trascurabile.
        try:
            save_task_manifest(
                source_info, base_seed, num_trees, parts_num_trees,
                self.environment, self.worker_name
            )
        except Exception as e:
            print(f"[!] [{self.worker_name}] ERRORE CRITICO: tutte le {len(parts_num_trees)} parti "
                  f"sono state salvate ma la scrittura del manifest è fallita. Il task viene "
                  f"segnalato come fallito all'Orchestratore. Dettaglio: {e}")
            raise
        print(f"[+] [{self.worker_name}] Task completato e salvato in {len(parts_num_trees)} parti "
              f"sullo storage condiviso. Invio ack (niente più blob via RPC).")

        oob_ready = False
        if _oob_distributed_enabled and oob_count_local.sum() > 0:
            try:
                job_id_guess = os.path.basename(source_info).replace("shared_train_", "").rsplit(".", 1)[0]
                # Root di storage derivata da source_info stesso (funziona sia per
                # 's3://bucket/distributed_trains/x.csv' sia per un path locale tipo
                # './.local_storage/x.csv' - os.path.dirname tratta entrambi come
                # semplici stringhe POSIX, nessuna differenza di comportamento tra
                # i due schemi): risaliamo di due livelli (dal file al suo folder,
                # dal folder al bucket/root) per affiancare 'oob_contributions' a
                # 'distributed_trains', invece di annidarlo dentro.
                storage_root = os.path.dirname(os.path.dirname(source_info))
                oob_path = f"{storage_root}/oob_contributions/{job_id_guess}/oob_seed_{base_seed}.pkl"
                payload = pickle.dumps({"oob_sum": oob_sum_local, "oob_count": oob_count_local})
                self.dao.save_binary(oob_path, payload)
                oob_ready = True
                print(f"[{self.worker_name}] [OOB] Contributo OOB persistito su '{oob_path}' "
                      f"({int((oob_count_local > 0).sum())} campioni coperti da almeno un albero di questo task).")
            except Exception as e_oob_save:
                print(f"[{self.worker_name}] [OOB-WARN] Salvataggio contributo OOB fallito ({e_oob_save}) - "
                      f"l'Orchestratore ricadra' sul metodo sequenziale per l'intero job.")

        _release_memory_to_os()
        # Non restituiamo più 'serialized_task' per intero via RPyC (fino a 1+ GB
        # su scenari di scalabilità): l'Orchestratore lo rilegge direttamente dallo
        # storage condiviso (S3/locale) con load_task_from_shared_storage, molto
        # più veloce e affidabile di un ritorno RPC su un payload di queste
        # dimensioni — vedi hang osservato in Scenario 2 (Scalabilità).
        return {"ack": True, "num_trees": num_trees, "oob_ready": oob_ready}

    def exposed_predict_subset_forest(self, serialized_trees_or_key, serialized_X_test=None, tree_type=None, global_classes=None):
        """
        Riceve un sottoinsieme di alberi dall'Orchestratore e calcola le
        predizioni parziali sui dati di test sfruttando il C nativo di Scikit-Learn.

        'serialized_trees_or_key' può essere:
          - una stringa: chiave nello storage condiviso (S3/locale) da cui il
            worker scarica da sé il blob. È il caso normale ora: passare
            l'intero chunk di alberi (fino a 1+ GB con pochi worker attivi)
            come argomento RPC causava hang/timeout di sessione (stesso
            problema già risolto per il ritorno degli alberi in fase di
            training - vedi exposed_train_subset_forest).
          - bytes: i byte già serializzati, per retrocompatibilità con
            eventuali chiamanti che li passano ancora direttamente.

        Per la classificazione restituiamo le probabilità per-albero (predict_proba),
        non le etichette dure: è lo stesso meccanismo di "soft voting" che sklearn
        usa internamente in RandomForestClassifier.predict/predict_proba (media delle
        distribuzioni di classe delle foglie), molto più informativo — soprattutto
        per l'AUC — del semplice conteggio di voti maggioritari con granularità
        1/n_alberi. Per la regressione il comportamento resta invariato (predict).
        """
        print(f"\n[WORKER RPC] Ricevuta richiesta di inferenza parziale...")

        # 1. Risolviamo il testing set PRIMA degli alberi: serve per predire
        #    sia nel percorso a streaming sia in quello retrocompatibile.
        if serialized_X_test is not None:
            # 'serialized_X_test' può essere:
            #  - una stringa: chiave nello storage condiviso, da scaricare da sé
            #    (caso normale ora, stesso pattern di 'serialized_trees_or_key').
            #  - bytes: già serializzati, per retrocompatibilità con chiamanti
            #    che non sono ancora passati al pattern a chiave.
            # In entrambi i casi la cache è chiavata sul valore RICEVUTO (la chiave
            # stringa, o i bytes grezzi), NON sul contenuto decodificato: se il
            # worker riceve la stessa chiave/stessi bytes di prima, evita sia il
            # download sia il re-pickle.
            if self._cached_X_test_bytes != serialized_X_test:
                if isinstance(serialized_X_test, str):
                    downloaded = load_bytes_from_shared_storage(
                        serialized_X_test, self.environment, self.worker_name
                    )
                    if downloaded is None:
                        raise RuntimeError(
                            f"[{self.worker_name}] Impossibile scaricare il testing set "
                            f"dalla chiave '{serialized_X_test}' nello storage condiviso."
                        )
                    self._cached_X_eval = pickle.loads(downloaded)
                else:
                    self._cached_X_eval = pickle.loads(serialized_X_test)
                self._cached_X_test_bytes = serialized_X_test
                print(f"[{self.worker_name}] Testing set centralizzato scaricato/decodificato (Shape: {self._cached_X_eval.shape}).")
            else:
                print(f"[{self.worker_name}] Utilizzo del testing set centralizzato già in cache (Shape: {self._cached_X_eval.shape}).")
            X_eval = self._cached_X_eval
        else:
            if getattr(self, 'X_test', None) is None:
                raise ValueError(
                    f"[{self.worker_name}] Errore: Nessun dataset di test locale trovato in memoria. "
                    f"Esegui prima il round di addestramento federato."
                )
            X_eval = self.X_test
            print(f"[{self.worker_name}] Utilizzo del testing set federato locale (Shape: {X_eval.shape}).")

        is_classifier = (tree_type == "classifier") if tree_type is not None else self.is_regression() is False
        global_classes_arr = np.asarray(global_classes) if (is_classifier and global_classes is not None) else None
        n_global_classes = len(global_classes_arr) if global_classes_arr is not None else None

        def _predict_batch(trees_batch):
            """Predice su un batch di alberi già deserializzati. Ritorna la
            lista di array di predizione (uno per albero) -- MOLTO più
            piccoli degli alberi stessi (un array (n_samples,) o
            (n_samples, n_classi) contro un DecisionTree non potato che può
            pesare decine di MB), quindi accumularli per tutta la chiamata
            costa poco anche su chunk grandi."""
            if global_classes_arr is not None:
                # Per la classificazione: probabilità per-albero (soft voting),
                # non etichette dure -- stesso meccanismo che sklearn usa
                # internamente in RandomForestClassifier.predict/predict_proba,
                # più informativo (soprattutto per l'AUC) di un conteggio di
                # voti maggioritari con granularità 1/n_alberi. Un singolo
                # albero, se addestrato su un campione bootstrap che per caso
                # non conteneva tutte le classi, espone tree.classes_ come
                # sottoinsieme di global_classes: rimappiamo le sue colonne di
                # probabilità nello spazio delle classi GLOBALE (0 per le
                # classi non viste da quell'albero) invece di assumere
                # ciecamente che l'ordine coincida.
                batch_predictions = []
                for tree in trees_batch:
                    raw_proba = tree.predict_proba(X_eval)
                    aligned_proba = np.zeros((X_eval.shape[0], n_global_classes), dtype=np.float64)
                    tree_classes = np.asarray(tree.classes_)
                    col_positions = np.searchsorted(global_classes_arr, tree_classes)
                    aligned_proba[:, col_positions] = raw_proba
                    batch_predictions.append(aligned_proba)
                return batch_predictions
            return [tree.predict(X_eval) for tree in trees_batch]

        # 2. Ricostruiamo gli alberi e prediciamo.
        #
        # 'serialized_trees_or_key' può essere:
        #  - una stringa: il PREFISSO di un manifest+parti nello storage
        #    condiviso (vedi save_chunk_in_parts_to_shared_storage lato
        #    Orchestratore). È il caso normale ora: leggiamo, prediciamo e
        #    liberiamo una parte di alberi alla volta, invece di scaricare e
        #    deserializzare l'intero chunk (fino a 1+ GB con pochi worker
        #    attivi) in un colpo solo -- che teneva contemporaneamente in RAM
        #    sia i byte grezzi sia gli oggetti albero appena decodificati, ed
        #    era la causa più probabile dell'OOM osservato sui worker in fase
        #    di inferenza dopo aver ridotto il numero di worker attivi.
        #  - bytes: il blob già serializzato per intero, per retrocompatibilità
        #    con eventuali chiamanti che lo passano ancora così (nessuna
        #    struttura a parti da sfruttare in quel caso: si deserializza e si
        #    predice tutto insieme, come avveniva prima di questo fix).
        if isinstance(serialized_trees_or_key, str):
            sub_predictions = []
            n_trees_total = 0
            for part_trees in iter_chunk_parts_from_shared_storage(
                serialized_trees_or_key, self.environment, self.worker_name
            ):
                n_trees_total += len(part_trees)
                sub_predictions.extend(_predict_batch(part_trees))
                # 'part_trees' esce di scope alla prossima iterazione: gli
                # alberi di questa parte non restano referenziati da nessuna
                # struttura dati del worker una volta predetto su di essi.
            print(f"[{self.worker_name}] Predetto in streaming su {n_trees_total} alberi "
                  f"(chunk '{serialized_trees_or_key}', a parti).")
        else:
            trees = pickle.loads(serialized_trees_or_key)
            print(f"[{self.worker_name}] Decodificati {len(trees)} alberi per il calcolo (blob diretto, non a parti).")
            sub_predictions = _predict_batch(trees)

        print(f"[+] [{self.worker_name}] Calcolo predizioni completato per {len(sub_predictions)} alberi.")
        return pickle.dumps(sub_predictions)


    # NOTA: da questa modifica in poi, exposed_train_subset_forest non chiama
    # più _get_task_storage_paths/_load_task_from_shared_storage/
    # _save_task_to_shared_storage: usa le funzioni equivalenti (e lo schema
    # a parti) importate da src.shared.utilities.task_storage. I tre metodi
    # sotto restano SOLO per compatibilità con eventuali altri chiamanti
    # (es. FederatedWorker, non incluso in questa revisione) che potrebbero
    # ancora dipendere dal formato monolitico. Se nessun altro li usa, sono
    # candidati alla rimozione in un secondo passaggio di pulizia.
    def _get_task_storage_paths(self, source_info: str, base_seed: int, num_trees: int):
        """
        Genera i percorsi per lo storage condiviso basandosi sul TASK.
        Estrae il job_id dal source_info per evitare collisioni tra job diversi.
        """
        # Estrazione sicura del job_id dal path del file (funziona sia per S3 che locale)
        filename = os.path.basename(source_info) # es: shared_train_12345.csv
        job_id = filename.replace("shared_train_", "").replace(".csv", "")

        local_dir = os.path.join("./.local_storage", "trained_tasks")
        base_name = f"task_{job_id}_seed_{base_seed}_trees_{num_trees}"
        local_meta_path = os.path.join(local_dir, base_name + ".meta.json")
        local_bin_path = os.path.join(local_dir, base_name + ".bin")

        s3_bucket = os.environ.get("DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket-759804778194-us-east-1-an")
        s3_key = f"tasks/{job_id}/task_seed_{base_seed}_trees_{num_trees}.pkl"

        return local_dir, local_meta_path, local_bin_path, s3_bucket, s3_key

    def _load_task_from_shared_storage(self, source_info: str, base_seed: int, num_trees: int) -> bytes:
        """Tenta di recuperare i byte serializzati dell'INTERO TASK dallo storage condiviso."""
        local_dir, local_meta_path, local_bin_path, s3_bucket, s3_key = self._get_task_storage_paths(source_info, base_seed, num_trees)

        if self.environment == "local":
            if os.path.exists(local_bin_path):
                try:

                    with open(local_bin_path, "rb") as f:
                        return f.read()
                except Exception as e:
                    print(f"[{self.worker_name}] Errore durante la lettura del task binario locale: {e}")
        else:

            try:

                s3_client = boto3.client("s3")
                response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
                print(f"[{self.worker_name}] [TASK HIT] Trovato task su S3: s3://{s3_bucket}/{s3_key}")
                return response['Body'].read()
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchKey':
                    print(f"[{self.worker_name}] Errore S3 per il task seed {base_seed}: {e}")
            except Exception as e:
                print(f"[{self.worker_name}] Errore imprevisto nel recupero del task da S3: {e}")

        return None

    def _save_task_to_shared_storage(self, source_info: str, base_seed: int, num_trees: int, serialized_trees_bytes: bytes):
        """Persiste in modo atomico i byte dell'intero TASK nello storage condiviso."""
        local_dir, local_meta_path, local_bin_path, s3_bucket, s3_key = self._get_task_storage_paths(source_info, base_seed, num_trees)

        if self.environment == "local":
            try:

                os.makedirs(local_dir, exist_ok=True)
                tmp_bin_path = local_bin_path + ".tmp"
                with open(tmp_bin_path, "wb") as f:
                    f.write(serialized_trees_bytes)
                os.replace(tmp_bin_path, local_bin_path)

                tmp_meta = local_meta_path + ".tmp"
                with open(tmp_meta, "w", encoding="utf-8") as f:
                    json.dump({
                        "base_seed": base_seed,
                        "num_trees": num_trees,
                        "size_bytes": len(serialized_trees_bytes),
                        "timestamp": time.time()
                    }, f, indent=2)
                # Sostituzione atomica per prevenire corruzioni di file
                os.replace(tmp_meta, local_meta_path)
                print(f"[{self.worker_name}] [TASK STORAGE] Task {base_seed} salvato nello storage locale condiviso.")
            except Exception as e:
                print(f"[{self.worker_name}] Errore nel salvataggio del task JSON locale: {e}")
                raise
        else:
            # Ambiente AWS: Scrittura diretta del payload binario su S3
            size_mb = len(serialized_trees_bytes) / (1024 ** 2)
            print(f"[{self.worker_name}] [TASK STORAGE] Avvio upload task su S3 "
                  f"({size_mb:.1f} MB, bucket: {s3_bucket}, key: {s3_key})...")
            start_ts = time.time()
            try:
                s3_client = boto3.client("s3")
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=s3_key,
                    Body=serialized_trees_bytes  # Passi direttamente i byte, senza io.BytesIO
                )
                elapsed = time.time() - start_ts
                print(f"[{self.worker_name}] [TASK STORAGE] Task {base_seed} salvato su S3 "
                      f"in {elapsed:.1f}s ({size_mb:.1f} MB).")
            except Exception as e:
                elapsed = time.time() - start_ts
                print(f"[{self.worker_name}] Errore nel caricamento del task su S3 dopo {elapsed:.1f}s: {e}")
                raise