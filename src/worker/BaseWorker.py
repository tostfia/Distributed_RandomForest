from abc import ABC, abstractmethod
import os
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
from src.shared.utilities.task_storage import load_bytes_from_shared_storage

# Timeout (in secondi) per il completamento di un singolo batch di alberi nel
# ThreadPool. Stesso pattern/nome di FederatedWorker.RPC_SYNC_TIMEOUT_SECONDS,
# ma default 600 per restare coerente col 'sync_request_timeout' già usato in
# start_server (protocol_config) per il path centralizzato.
RPC_SYNC_TIMEOUT_SECONDS = int(os.environ.get("RPC_SYNC_TIMEOUT_SECONDS", 600))

_child_X = None
_child_y = None

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
                                     bootstrap=None, max_samples=None):
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
        cached_task_bytes = self._load_task_from_shared_storage(source_info, base_seed, num_trees)
        if cached_task_bytes is not None:
            print(f"[{self.worker_name}] [SHORT-CIRCUIT] Task già pronto nello storage. Invio solo ack "
                  f"(l'Orchestratore rilegge il blob direttamente dallo storage condiviso).")
            return {"ack": True, "num_trees": num_trees}
        # 1. Recupero dati e classe dell'albero dalle classi figlie
        X, y = self._load_data(source_info)
        tree_class = self._get_tree_class()

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

        if num_trees == 1:
            print("[WORKER] Ottimizzazione: 1 solo albero richiesto. Calcolo diretto senza ThreadPool.")
            local_trees = [_train_single_tree_thread(worker_tasks[0])]
        else:
            pool_size = min(num_trees, allocated_cores)
            print(f"[WORKER] Istanziazione ThreadPool locale con {pool_size} thread "
                  f"(memoria condivisa nativa, nessuna copia/serializzazione tra thread)...")

            BATCH_SIZE = max(1, min(pool_size * 4, num_trees))
            local_trees = []
            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                for batch_start in range(0, len(worker_tasks), BATCH_SIZE):
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
                    local_trees.extend(batch_trees)
                    del batch_trees

        print(f"[+] Calcolo di {num_trees} alberi completato. Salvataggio su storage condiviso...")
        serialized_task = pickle.dumps(local_trees)
        try:
            self._save_task_to_shared_storage(source_info, base_seed, num_trees, serialized_task)
        except Exception as e:
            # Il task NON deve risultare "completato con successo" se non è stato
            # persistito nello storage condiviso: rilanciamo l'eccezione così RPyC
            # la propaga all'Orchestratore, che potrà marcare il task come fallito
            # e decidere se ritentarlo, invece di credere erroneamente che sia andato
            # tutto bene (comportamento precedente, silenziosamente errato).
            print(f"[!] [{self.worker_name}] ERRORE CRITICO: gli alberi sono stati calcolati "
                  f"ma il salvataggio nello storage condiviso è fallito. Il task viene "
                  f"segnalato come fallito all'Orchestratore. Dettaglio: {e}")
            raise
        print(f"[+] [{self.worker_name}] Task salvato nello storage condiviso. Invio ack "
              f"(niente più blob via RPC).")
        # Non restituiamo più 'serialized_task' per intero via RPyC (fino a 1+ GB
        # su scenari di scalabilità): l'Orchestratore lo rilegge direttamente dallo
        # storage condiviso (S3/locale) con load_task_from_shared_storage, molto
        # più veloce e affidabile di un ritorno RPC su un payload di queste
        # dimensioni — vedi hang osservato in Scenario 2 (Scalabilità).
        return {"ack": True, "num_trees": num_trees}

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

        # 1. Ricostruiamo gli alberi: se è una chiave (str), li scarichiamo
        #    dallo storage condiviso; se sono già byte, li usiamo direttamente.
        if isinstance(serialized_trees_or_key, str):
            serialized_trees = load_bytes_from_shared_storage(
                serialized_trees_or_key, self.environment, self.worker_name
            )
            if serialized_trees is None:
                raise RuntimeError(
                    f"[{self.worker_name}] Impossibile scaricare il chunk di alberi "
                    f"dalla chiave '{serialized_trees_or_key}' nello storage condiviso."
                )
        else:
            serialized_trees = serialized_trees_or_key
        trees = pickle.loads(serialized_trees)
        print(f"[{self.worker_name}] Decodificati {len(trees)} alberi per il calcolo.")

        # 2. Gestione asimmetrica Centralizzato vs Federato
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

        if is_classifier and global_classes is not None:
            global_classes_arr = np.asarray(global_classes)
            n_global_classes = len(global_classes_arr)
            print(f"[{self.worker_name}] Avvio inferenza soft-voting (predict_proba) su {len(trees)} alberi "
                  f"({n_global_classes} classi globali)...")
            sub_predictions = []
            for tree in trees:
                # Un singolo albero, se addestrato su un campione bootstrap che per caso
                # non conteneva tutte le classi, espone tree.classes_ come sottoinsieme
                # di global_classes: rimappiamo le sue colonne di probabilità nello
                # spazio delle classi GLOBALE (0 per le classi non viste da quell'albero)
                # invece di assumere ciecamente che l'ordine coincida.
                raw_proba = tree.predict_proba(X_eval)
                aligned_proba = np.zeros((X_eval.shape[0], n_global_classes), dtype=np.float64)
                tree_classes = np.asarray(tree.classes_)
                col_positions = np.searchsorted(global_classes_arr, tree_classes)
                aligned_proba[:, col_positions] = raw_proba
                sub_predictions.append(aligned_proba)
        else:
            print(f"[{self.worker_name}] Avvio inferenza nativa lineare (hard predict) su {len(trees)} alberi...")
            sub_predictions = [tree.predict(X_eval) for tree in trees]

        print(f"[+] [{self.worker_name}] Calcolo predizioni completato per {len(trees)} alberi.")
        return pickle.dumps(sub_predictions)


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