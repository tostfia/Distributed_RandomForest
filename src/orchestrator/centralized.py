import pickle
import os
import gc
import ctypes
import socket
import time
import rpyc
import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from rpyc.utils.classic import obtain
import numpy as np
from sklearn.model_selection import train_test_split
import src.shared.utilities.datasplitter
from src.shared.config import SystemConfig
from src.shared.factory import DatasetDAOFactory
from src.orchestrator.BaseOrchestrator import BaseOrchestrator, env_timeout_seconds
from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader
from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
from src.shared.utilities.preprocessing import CICIDSPreprocessor
from src.shared.utilities.undersampling import undersample_majority_class
from src.dataset.checkpoint_dao import CheckpointDAOFactory
from src.shared.utilities.task_storage import (
    iter_task_parts_as_tree_lists,
    save_chunk_in_parts_to_shared_storage,
    save_bytes_to_shared_storage,
)

TEST_SIZE = 0.2
BUCKET_NAME = os.environ.get("DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket-759804778194-us-east-1-an")
# Stesso valore di run_baseline.py (TARGET_ROWS_PER_DAY): campionamento
# RIBILANCIATO per giorno di cattura invece di una sample_fraction uniforme
# sull'intero dataset (che farebbe dominare il campione dal giorno più
# grande -- vedi RawCSVDataLoader per la motivazione completa). Da questo
# valore dipende anche il ribilanciamento locale automatico dei due giorni
# "protetti" (Infiltration), attivo dentro RawCSVDataLoader ogni volta che
# target_rows_per_day non è None -- nessuna configurazione aggiuntiva
# richiesta qui per beneficiarne.
TARGET_ROWS_PER_DAY = 100_000
# Stessi valori di run_baseline.py: senza allinearli qui, il train
# distribuito e quello della baseline locale sarebbero addestrati su
# distribuzioni/feature-set diversi, invalidando il confronto delle
# METRICHE (quello sui tempi resterebbe comunque valido, essendo
# indipendente da questi due parametri).
UNDERSAMPLING_RATIO = 1.0
# Stesso valore di run_baseline.py: 15% del train ritagliato PRIMA
# dell'undersampling. Nella baseline serve a calibrare la soglia di
# decisione su un validation set con la vera distribuzione sbilanciata; qui
# NON viene usato per una soglia (il percorso distribuito non fa ancora
# quella calibrazione, vedi discussione) ma va comunque tolto dal train PER
# VOLUME: senza questo passo, l'undersampling lavorerebbe su un pool il 15%
# più grande di quello della baseline, e il train finale non sarebbe più
# lo stesso set di dati, solo un set con lo stesso RAPPORTO 1:1.
VALIDATION_SIZE_FOR_THRESHOLD = 0.15

# Timeout (in secondi) delle chiamate RPC sincrone verso i worker.
#
# PRIMA: due letterali 600 incastonati nelle chiamate a rpyc.connect (nel thread
# di dispatch dell'addestramento e in quello dell'inferenza). Terraform (ecs_task_definitions.tf) leggeva
# RPC_SYNC_TIMEOUT_SECONDS / RPC_INFERENCE_SYNC_TIMEOUT_SECONDS dal .env e le
# iniettava nella task definition ECS dell'orchestratore, ma il codice
# centralizzato non le leggeva: la configurazione c'era, era documentata, e non
# aveva alcun effetto. Solo federated.py le usava davvero.
#
# I DEFAULT RESTANO 600/600, non i 1800/900 di federated.py: così, quando le
# variabili non sono impostate — cioè in locale e in Docker Compose — il
# comportamento è identico byte per byte a quello precedente. Su AWS, dove
# Terraform (ecs_task_definitions.tf) le valorizza, il timeout diventa finalmente quello dichiarato nel
# .env, che è il punto di tutta questa configurazione.
RPC_SYNC_TIMEOUT_SECONDS = env_timeout_seconds("RPC_SYNC_TIMEOUT_SECONDS", 600)
RPC_INFERENCE_SYNC_TIMEOUT_SECONDS = env_timeout_seconds("RPC_INFERENCE_SYNC_TIMEOUT_SECONDS", 600)

class CentralizedOrchestrator(BaseOrchestrator):
    def __init__(self, orchestrator_name: str = None):
        self.cfg = SystemConfig()
        name = orchestrator_name or f"Orchestrator-Centralizzato-{socket.gethostname()}"

        self.current_job_id = None
        self.train_data_path = None
        self.test_data_path = None
        # SHARDING DINAMICO (12/9/2026): None = modalita' 'shared' (default,
        # comportamento identico a sempre - ogni worker scarica l'intero
        # dataset). Se valorizzata (lista di path, uno per shard), la
        # modalita' 'sharded' e' attiva per il job corrente: ogni worker
        # scarica SOLO la propria fetta, assegnata dinamicamente per
        # task_id (vedi _execute_training_step). Toggle via env var
        # CENTRALIZED_DATASET_MODE ('shared'|'sharded').
        self.train_data_shards = None
        self.chunk_sent_event = threading.Event()
        self._trees_cache = {}
        # Durata dell'ultima fase di preparazione dati (ETL). Serve agli scenari
        # di test per scomporre il tempo totale in "preparazione dati" +
        # "addestramento distribuito": la baseline locale misura t_seq sul solo
        # fit, quindi confrontarla con un totale che include l'ETL (30-40s su
        # AWS per via di S3) penalizzerebbe sistematicamente il cluster.
        # 0.0 quando l'ETL viene saltata grazie allo SHORT-CIRCUIT.
        self.last_etl_seconds = 0.0
        # Scomposizione del tempo di _execute_training_step, esposta perché il
        # confronto con la baseline locale sia onesto in entrambe le direzioni.
        # La baseline misura il solo fit di scikit-learn: sommarci sopra
        # trasferimenti S3, checkpoint e stima OOB — che la baseline non fa
        # affatto — penalizzerebbe il cluster per lavoro che non gli è stato
        # chiesto di confrontare.
        #
        #   last_dispatch_seconds     costruzione vera degli alberi: scoperta
        #                             dei worker, invio dei chunk via RPC e
        #                             attesa del loro completamento. È IL
        #                             numero da confrontare con T_seq/T_1node.
        #   last_aggregation_seconds  ricomposizione della foresta globale e
        #                             salvataggio del modello sullo storage.
        #   last_oob_seconds          RIMOSSA (12/9/2026): la stima OOB non
        #                             viene più calcolata in nessun caso -
        #                             restava un costo (fino a 150s+ nel
        #                             metodo sequenziale) per un dato che
        #                             nessun consumatore del sistema legge
        #                             mai (le accuracy_metrics finali vengono
        #                             sempre dall'inferenza reale sul test
        #                             set). Attributo mantenuto SEMPRE a 0.0
        #                             solo per compatibilità con lo schema
        #                             dei report esistenti (scalability.py
        #                             legge 'oob_estimation_seconds' via
        #                             getattr con default 0.0).
        #
        # Totale di _execute_training_step ~=
        #   last_etl_seconds + last_dispatch_seconds + last_aggregation_seconds
        self.last_dispatch_seconds = 0.0
        self.last_aggregation_seconds = 0.0
        self.last_oob_seconds = 0.0
        
        super().__init__(
            orchestrator_name=name,
            queue_name=self.cfg.sqs_centralized_queue
        )
        self.checkpoint_dao = CheckpointDAOFactory.get_dao(self.environment)

    def _resolve_dataset_type(self, payload: dict) -> str:
        """Determina il tipo di dataset basandosi sul payload inviato dal Client."""
        dataset_type = payload.get("dataset_type")
        if dataset_type:
            return str(dataset_type).strip().lower()
        return "real"
    
    def _prepare_data(self, payload: dict, base_seed: int, num_shards: int = None):
        t0 = time.perf_counter()
        job_id = payload.get("job_id", "unknown_job")
        dataset_path = payload.get("dataset_path")
        dataset_type = self._resolve_dataset_type(payload)
        hp = payload.get("hyperparameters", {})
        tree_type = hp.get("tree_type", "classifier")
        target_col = "Target" if  tree_type == "regressor" else "Label"

        splitter = src.shared.utilities.datasplitter.StratifiedDataSplitter(target_column=target_col, test_size=TEST_SIZE, random_state=base_seed)

        print(f"\n[{self.orchestrator_name}] Avvio ETL. Tipo: {dataset_type}")

        if dataset_type == "synthetic":
            loader = SyntheticDataLoader(task="regression" if tree_type == "regressor" else "classification", target_column=target_col)
            df_full = loader.load()

            if tree_type == "regressor":
                train_df, test_df = train_test_split(df_full, test_size=TEST_SIZE, random_state=base_seed)
            else:
                train_df, test_df = splitter.split(df_full)
            # Stessa pulizia già applicata al ramo 'real' (vedi 'del df_raw' /
            # 'del df_binarized' sotto): 'df_full' qui è 1.000.000 x 101 colonne
            # (~800MB+), e train_test_split/splitter.split restituiscono COPIE
            # (train_df/test_df), non view -- quindi df_full è ridondante subito
            # dopo lo split, ma prima di questa modifica restava referenziato
            # per tutta la durata di _prepare_data (upload S3 incluso, alcuni
            # minuti). Essendo un DataFrame pandas, può contenere riferimenti
            # interni ciclici che il reference counting di CPython non libera
            # immediatamente: 'del' esplicito + gc.collect() forzano il rilascio
            # prima che il dispatch del training (subito dopo, vedi
            # _execute_training_step) inizi ad accumulare memoria per gli
            # alberi -- riduce il fabbisogno di picco sull'Orchestratore.
            del df_full
            gc.collect()
        else:
            if not dataset_path: 
                raise ValueError("dataset_path mancante.")
            print(f"[DEBUG] dataset_path ricevuto = {repr(dataset_path)}")
            # sample_fraction=0.05 (uniforme) SOSTITUITO da
            # target_rows_per_day: stesso principio di run_baseline.py --
            # senza questo, il campione sarebbe dominato dal giorno di
            # cattura più grande (quasi metà del dataset da solo), e i due
            # giorni con l'attacco Infiltration (già raro anche al loro
            # interno) verrebbero diluiti ulteriormente da un campionamento
            # cieco al Label PRIMA che qualunque bilanciamento a valle possa
            # intervenire.
            loader = RawCSVDataLoader(
                data_url=dataset_path,
                dataset_seed=base_seed,
                target_rows_per_day=TARGET_ROWS_PER_DAY,
            )
            df_raw = loader.load()
            
            # Istanziamo il nuovo preprocessor modificato
            preprocessor = CICIDSPreprocessor(target_column=target_col)
            # ─── FASE 1: BINARIZZAZIONE SUL DATO INTERO ───
            df_binarized = preprocessor.binarize_target(df_raw)
            del df_raw
            # ─── FASE 2: SPLIT STRATIFICATO ADESSO SICURO ───
            print(f"[{self.orchestrator_name}] Esecuzione Split Stratificato...")
            train_df, test_df = splitter.split(df_binarized)
            del df_binarized

            # ─── FASE 3 & 4: PREPROCESAMENTO INDIPENDENTE (Metadata + NaN/inf) ───
            print(f"\n[{self.orchestrator_name}] === PREPROCESSING SUL TRAIN SET ===")
            train_df = preprocessor.process(train_df)
            
            print(f"\n[{self.orchestrator_name}] === PREPROCESSING SUL TEST SET ===")
            test_df = preprocessor.process(test_df)

            # ─── FASE 4b: SPLIT DEL VALIDATION SET (PRIMA dell'undersampling) ───
            # Stesso passo di run_baseline.py, stesso identico ordine (dopo
            # process(), prima di undersample_majority_class): 15% del train
            # tolto qui, per allineare il VOLUME del train che arriva
            # all'undersampling a quello della baseline -- vedi
            # VALIDATION_SIZE_FOR_THRESHOLD. Il validation stesso non è
            # ancora usato per calibrare una soglia in questo percorso
            # distribuito (nessuna logica di soglia F1-max/FPR-vincolata
            # qui), quindi viene scartato subito dopo lo split.
            if tree_type == "classifier":
                print(f"\n[{self.orchestrator_name}] === SPLIT VALIDATION SET "
                      f"({VALIDATION_SIZE_FOR_THRESHOLD*100:.0f}% del train, per allineamento volume) ===")
                validation_splitter = src.shared.utilities.datasplitter.StratifiedDataSplitter(
                    target_column=target_col, test_size=VALIDATION_SIZE_FOR_THRESHOLD, random_state=base_seed
                )
                train_df, _ = validation_splitter.split(train_df)

            # ─── FASE 5: UNDER-SAMPLING DELLA CLASSE MAGGIORITARIA (solo train) ───
            # Prima assente qui: il train distribuito restava alla distribuzione
            # naturale, mentre run_baseline.py addestra sempre su un train
            # bilanciato 1:1 -- due modelli addestrati su dati diversi, non
            # confrontabili sulle metriche. Il test set resta INTATTO (mai
            # sotto-campionato), stesso principio della baseline: altrimenti la
            # valutazione finale non misurerebbe più le prestazioni sulla
            # distribuzione reale.
            if tree_type == "classifier":
                print(f"\n[{self.orchestrator_name}] === UNDER-SAMPLING CLASSE MAGGIORITARIA (solo train) ===")
                train_df = undersample_majority_class(
                    train_df, target_column=target_col,
                    majority_class=0, minority_class=1,
                    ratio=UNDERSAMPLING_RATIO, random_state=base_seed,
                )

        # --- FEATURE SELECTION (Solo Real) ---
        if dataset_type == "real":
            # PRIMA qui si rifaceva un fit COMPLETO di CICIDSFeatureSelector
            # (un Random Forest + permutation importance da zero) sul lato
            # distribuito -- duplicando un lavoro che la baseline ha già
            # fatto, e per giunta rischiando di produrre un set di feature
            # DIVERSO da quello della baseline anche a parità di
            # iperparametri (nessuna garanzia che un fit indipendente
            # converga esattamente sulle stesse feature). Il tuning e la
            # feature selection restano ESCLUSIVAMENTE compiti della
            # baseline (run_baseline.py): il percorso distribuito si limita
            # a consumarne l'output già calcolato, esattamente come già
            # fa federatedWorker.py (_resolve_selected_features) -- stesso
            # meccanismo, stesso file, ora condiviso via
            # BaseOrchestrator.read_selected_features_from_config.
            feature_selezionate = self.read_selected_features_from_config(dataset_type)
            if feature_selezionate is not None:
                colonne_da_tenere = [c for c in feature_selezionate if c != target_col] + [target_col]
                print(f"[{self.orchestrator_name}] Applicazione spazio feature dalla baseline "
                      f"({len(feature_selezionate)} colonne).")
                train_df = train_df[colonne_da_tenere]
                test_df = test_df[colonne_da_tenere]
            else:
                print(f"[{self.orchestrator_name}] [ATTENZIONE] Nessuna feature selezionata "
                      f"disponibile da config_real.json: uso il set completo (69 feature circa). "
                      f"Esegui prima run_baseline.py per un confronto allineato alla baseline.")

        # --- SALVATAGGIO COORDINATO DAI DAO ---
        if self.environment == "aws":
            test_data_path = f"s3://{BUCKET_NAME}/distributed_tests/shared_test_{job_id}.csv"
        else:
            test_data_path = f"./.local_storage/shared_test_{job_id}.csv"

        print(f"\n[{self.orchestrator_name}] Delega salvataggio a DatasetDAOFactory...")
        try:
            dao = DatasetDAOFactory.get_dao(self.environment)
            # Il test set NON viene mai partizionato, indipendentemente dalla
            # modalita': l'inferenza divide gli ALBERI tra worker (ogni worker
            # valida l'intero test set con la propria fetta di alberi), non i
            # dati - vedi _execute_inference_step. Nessuna ridondanza N-way
            # da eliminare qui come invece accade per il training set.
            dao.save_dataset(path=test_data_path, df=test_df)

            if num_shards is not None and num_shards > 1:
                # SHARDING DINAMICO (12/9/2026): seed fisso per riproducibilita'
                # in entrambi i rami sotto. np.array_split copre l'INTERO
                # array anche con resti non divisibili esattamente (es.
                # 800000/3): ogni riga finisce in esattamente una fetta,
                # nessuna persa o duplicata - vero per entrambi i rami.
                print(f"[{self.orchestrator_name}] [SHARDING] Partizionamento train_df "
                      f"({train_df.shape[0]} righe) in {num_shards} shard...")
                rng = np.random.RandomState(base_seed)

                if tree_type == "classifier":
                    # STRATIFICATO (solo classificatore/reale, 12/9/2026):
                    # shuffle e split SEPARATI per classe, poi distribuiti
                    # proporzionalmente tra gli shard - invece di un unico
                    # shuffle globale. Garantisce che ogni shard riceva
                    # (quasi) esattamente la stessa proporzione di ciascuna
                    # classe presente in train_df (gia' vicina a 1:1 grazie
                    # all'undersampling a monte, vedi FASE 5 sopra), invece
                    # di affidarsi alla sola probabilita' di uno shuffle non
                    # stratificato. Il regressore (ramo else sotto) non ha
                    # un concetto di classe: resta con lo shuffle puro.
                    actual_target_shard = target_col if target_col in train_df.columns else (
                        "Target" if "Target" in train_df.columns else "Label"
                    )
                    print(f"[{self.orchestrator_name}] [SHARDING] Split stratificato per classe "
                          f"(target='{actual_target_shard}').")
                    class_values = train_df[actual_target_shard].to_numpy()
                    shard_indices = [np.array([], dtype=np.int64) for _ in range(num_shards)]
                    for cls in np.unique(class_values):
                        cls_positions = np.where(class_values == cls)[0]
                        cls_shuffled = rng.permutation(cls_positions)
                        cls_split = np.array_split(cls_shuffled, num_shards)
                        for i in range(num_shards):
                            shard_indices[i] = np.concatenate([shard_indices[i], cls_split[i]])
                    # Rimescola ogni shard dopo la concatenazione per classe,
                    # cosi' le righe non restano raggruppate per classe
                    # all'interno dello shard (irrilevante per il training
                    # dell'albero, solo per pulizia/uniformita').
                    for i in range(num_shards):
                        shard_indices[i] = rng.permutation(shard_indices[i])
                else:
                    # Regressore (sintetico): nessun concetto di classe da
                    # rispettare - shuffle globale puro, split sequenziale
                    # post-shuffle in num_shards fette.
                    shuffled_idx = rng.permutation(train_df.shape[0])
                    shard_indices = np.array_split(shuffled_idx, num_shards)

                if self.environment == "aws":
                    shard_paths = [
                        f"s3://{BUCKET_NAME}/distributed_trains/shared_train_{job_id}_shard_{i}.csv"
                        for i in range(num_shards)
                    ]
                else:
                    shard_paths = [
                        f"./.local_storage/shared_train_{job_id}_shard_{i}.csv"
                        for i in range(num_shards)
                    ]

                # Scrittura in PARALLELO, non sequenziale: stesso volume totale
                # di byte del file unico di oggi, ma N upload concorrenti
                # invece di uno solo - tiene il costo per-giro comparabile (o
                # migliore) anche dovendo riscrivere gli shard ad ogni round
                # di scaling (nessun riuso possibile tra round con
                # worker_count diversi: il numero di shard cambia, quindi
                # niente short-circuit qui, vedi _execute_training_step).
                def _write_shard(i):
                    shard_df = train_df.iloc[shard_indices[i]]
                    dao.save_dataset(path=shard_paths[i], df=shard_df)
                    return i

                with ThreadPoolExecutor(max_workers=num_shards) as executor:
                    futures = {executor.submit(_write_shard, i): i for i in range(num_shards)}
                    for future in as_completed(futures):
                        future.result()  # propaga eventuali eccezioni dei thread

                self.last_etl_seconds = time.perf_counter() - t0
                print(f"[DEBUG TIMING] _prepare_data (sharded, {num_shards} shard) completato in "
                      f"{self.last_etl_seconds:.2f}s")
                print(f"[{self.orchestrator_name}] [OK] {num_shards} shard di training + test set "
                      f"archiviati correttamente.")

                self.train_data_shards = shard_paths
                self.train_data_path = None  # non usato in modalita' sharded
            else:
                if self.environment == "aws":
                    train_data_path = f"s3://{BUCKET_NAME}/distributed_trains/shared_train_{job_id}.csv"
                else:
                    train_data_path = f"./.local_storage/shared_train_{job_id}.csv"
                dao.save_dataset(path=train_data_path, df=train_df)
                self.last_etl_seconds = time.perf_counter() - t0
                print(f"[DEBUG TIMING] _prepare_data completato in {self.last_etl_seconds:.2f}s")
                print(f"[{self.orchestrator_name}] [OK] Dataset di Train e Test archiviati correttamente.")

                # CACHE EFS (11/9/2026): scrittura best-effort, SOLO se
                # EFS_MOUNT_PATH e' impostata (vedi orchestrator_ec2.tf) - se la
                # variabile manca o la scrittura fallisce per qualunque motivo,
                # non deve MAI far fallire il job: S3 sopra e' gia' il
                # salvataggio canonico richiesto dalla traccia, questo e' solo
                # un'ottimizzazione di velocita' per i worker che leggeranno lo
                # stesso file (vedi dataset_dao.py per la logica di lettura/
                # fallback lato worker). Solo train_df: e' quello che ogni
                # worker scarica per il training (il collo di bottiglia
                # misurato), non test_df (letto una sola volta dall'orchestrator
                # stesso per l'inferenza, nessuna ridondanza da eliminare li').
                # SOLO modalita' 'shared': in modalita' 'sharded' ogni worker
                # legge una fetta DIVERSA, non c'e' ridondanza N-way da
                # eliminare con una cache condivisa nello stesso modo.
                efs_mount_path = os.environ.get("EFS_MOUNT_PATH", "").strip()
                if efs_mount_path:
                    try:
                        efs_cache_dir = os.path.join(efs_mount_path, "dataset_cache")
                        os.makedirs(efs_cache_dir, exist_ok=True)
                        efs_cache_path = os.path.join(efs_cache_dir, os.path.basename(train_data_path))
                        # Scrittura atomica (tmp + replace): stesso pattern gia'
                        # usato altrove nel progetto (es. BaseWorker._save_task_to_shared_storage)
                        # per evitare che un worker legga un file a meta' scritto.
                        tmp_path = f"{efs_cache_path}.tmp-{os.getpid()}"
                        train_df.to_csv(tmp_path, index=False)
                        os.replace(tmp_path, efs_cache_path)
                        print(f"[{self.orchestrator_name}] [EFS] Cache scritta anche su "
                              f"'{efs_cache_path}' (oltre a S3) per lettura veloce dai worker.")
                    except Exception as e_efs:
                        print(f"[{self.orchestrator_name}] [EFS] [WARN] Scrittura cache fallita "
                              f"({e_efs}) - i worker ricadranno su S3, nessun impatto sulla correttezza.")

                self.train_data_path = train_data_path
                self.train_data_shards = None

            # Stessa motivazione di 'del df_full' sopra: train_df/test_df sono
            # copie potenzialmente grandi (fino a ~800MB combinate per lo
            # scenario sintetico) che altrimenti resterebbero vive fino al
            # ritorno della funzione, a ridosso dell'inizio del dispatch di
            # training (vedi _execute_training_step, chiamato subito dopo).
            del train_df, test_df
            gc.collect()
        except Exception as e:
            raise IOError(f"[{self.orchestrator_name}] Errore critico nel salvataggio dei dataset tramite DAO: {e}")
        self.current_job_id = job_id
        self.test_data_path = test_data_path

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int) -> int:
        """
        Esegue lo step di addestramento distribuito centralizzato.
        Restituisce il numero REALE di alberi totali validati e salvati con successo.

        NOTA (12/9/2026): la stima Out-Of-Bag è stata rimossa interamente da
        questo metodo (era presente come funzionalità opzionale, poi col
        default per saltarla, ora rimossa del tutto). Le metriche di
        accuratezza finali (accuracy_metrics nei report) vengono sempre
        dall'inferenza reale su un test set separato, mai dall'OOB -
        calcolarla non serviva a nessun consumatore del sistema.
        """
        expected_job_id = payload.get("job_id", "unknown_job")

        # SHARDING DINAMICO (12/9/2026): toggle via env var, default 'shared'
        # (comportamento identico a sempre). In modalita' 'sharded', il
        # numero di shard = numero di worker rilevati IN QUESTO MOMENTO -
        # serve quindi conoscerli PRIMA di generare/scrivere il dataset,
        # a differenza della modalita' 'shared' dove l'ordine resta invariato
        # (ETL, poi scoperta worker piu' sotto, invariata).
        dataset_mode = os.environ.get("CENTRALIZED_DATASET_MODE", "shared").strip().lower()
        sharded_mode = dataset_mode == "sharded"
        early_num_workers = None
        if sharded_mode:
            print(f"[{self.orchestrator_name}] [SHARDING] Modalita' 'sharded' attiva - "
                  f"scopro i worker PRIMA dell'ETL per sapere in quante fette partizionare.")
            while True:
                early_workers = ServiceRegistry.get_available_workers(self.environment)
                if early_workers:
                    early_num_workers = len(early_workers)
                    print(f"[{self.orchestrator_name}] [SHARDING] {early_num_workers} worker rilevati "
                          f"-> il dataset verra' partizionato in altrettante fette.")
                    break
                print(f"[{self.orchestrator_name}] [SHARDING] Nessun worker disponibile per la scoperta "
                      f"anticipata. In attesa...")
                time.sleep(10)

        # 1. Preparazione dei dati (se non ancora pronti e non presenti su disco)
        if sharded_mode:
            # Short-circuit basato su STORAGE (12/9/2026), non solo in-memoria:
            # costruisce i path attesi per gli shard (stessa convenzione di
            # _prepare_data) e controlla se esistono GIA' su storage. Questo
            # copre DUE casi in un colpo solo:
            #   1) round successivi sullo stesso job con lo stesso numero di
            #      worker (stesso motivo della vecchia guardia in-memoria);
            #   2) FAILOVER dell'orchestratore: un nuovo standby che prende
            #      il comando e' un processo Python nuovo (self.train_data_shards
            #      = None per costruzione, vedi __init__) - senza un controllo
            #      su storage, pagherebbe sempre un re-sharding completo da
            #      zero anche se gli shard del leader morto sono ancora li'
            #      su S3, mentre la modalita' 'shared' lo evita gia' col suo
            #      short-circuit (vedi ramo elif sotto) - BUCO TROVATO E
            #      CHIUSO qui, non presente nella prima versione di oggi.
            if self.environment == "aws":
                expected_shards = [
                    f"s3://{BUCKET_NAME}/distributed_trains/shared_train_{expected_job_id}_shard_{i}.csv"
                    for i in range(early_num_workers)
                ]
                expected_test_sharded = f"s3://{BUCKET_NAME}/distributed_tests/shared_test_{expected_job_id}.csv"
            else:
                expected_shards = [
                    f"./.local_storage/shared_train_{expected_job_id}_shard_{i}.csv"
                    for i in range(early_num_workers)
                ]
                expected_test_sharded = f"./.local_storage/shared_test_{expected_job_id}.csv"

            dao_check = DatasetDAOFactory.get_dao(self.environment)
            # N chiamate exists() (head_object, non download): costo
            # trascurabile (decine di ms l'una) rispetto alle centinaia di
            # secondi di un re-sharding completo evitato nel caso migliore.
            all_shards_exist = (
                all(dao_check.exists(p) for p in expected_shards)
                and dao_check.exists(expected_test_sharded)
            )

            if all_shards_exist:
                print(f"[{self.orchestrator_name}] [SHARDING] [SHORT-CIRCUIT] {early_num_workers} shard "
                      f"già presenti su storage per questo job (round successivo, o ripresa dopo un "
                      f"failover dell'orchestratore) - nessuna rigenerazione.")
                self.train_data_shards = expected_shards
                self.train_data_path = None
                self.test_data_path = expected_test_sharded
                self.current_job_id = expected_job_id
                self.last_etl_seconds = 0.0
            else:
                self._prepare_data(payload, seed, num_shards=early_num_workers)
        elif self.train_data_path is None or self.current_job_id != expected_job_id:
            if self.environment == "aws":
                expected_train = f"s3://{BUCKET_NAME}/distributed_trains/shared_train_{expected_job_id}.csv"
                expected_test = f"s3://{BUCKET_NAME}/distributed_tests/shared_test_{expected_job_id}.csv"
            else:
                expected_train = f"./.local_storage/shared_train_{expected_job_id}.csv"
                expected_test = f"./.local_storage/shared_test_{expected_job_id}.csv"
            
            dao = DatasetDAOFactory.get_dao(self.environment)
            if dao.exists(expected_train) and dao.exists(expected_test):
                print(f"[{self.orchestrator_name}] [SHORT-CIRCUIT ETL] Dataset già presente nello storage condiviso. Salto la fase ETL.")
                self.last_etl_seconds = 0.0
                self.train_data_path = expected_train
                self.test_data_path = expected_test
                self.current_job_id = expected_job_id
                # Sicurezza contro stati stantii: se un job PRECEDENTE era in
                # modalita' sharded, self.train_data_shards potrebbe ancora
                # contenere path vecchi - azzerato esplicitamente qui, dato
                # che siamo nel ramo 'shared' (sharded_mode=False).
                self.train_data_shards = None
            else:
                self._prepare_data(payload, seed)
        checkpoint_trees_path = self._resolve_trees_checkpoint_path(self.current_job_id)
        if self.environment != "aws":
            os.makedirs("./.local_storage", exist_ok=True)
        all_trained_trees = []

        # Pulizia preventiva: se ripartiamo da zero per QUESTO job_id ma esiste
        # già un checkpoint fisico residuo (es. retry manuale con lo stesso id,
        # o rerun dopo una pulizia incompleta), lo scartiamo per evitare che
        # venga riletto per errore da un round successivo (parità con FederatedOrchestrator).
        if start_alberi == 0:
            # Rimuove parti incrementali E monolitico: ripartendo da zero non
            # deve sopravvivere nulla di un tentativo precedente sullo stesso id.
            self._purge_trees_checkpoint(self.current_job_id)
        self._trees_cache.pop(self.current_job_id, None) if start_alberi == 0 else None

        # ─── SINCRONIZZAZIONE STATO: SE ABBIAMO GIÀ ALBERI DA UN ROUND PRECEDENTE ───
        if start_alberi > 0:
            cached = self._trees_cache.get(self.current_job_id)
            if cached is not None and len(cached) == start_alberi:
                # Stessa istanza, stesso job: nessun fault, è solo il round successivo
                # nello stesso processo. Riusiamo la lista già in memoria, niente GET S3.
                print(f"\n[{self.orchestrator_name}] [STATE-SYNC] Continuazione round nella stessa istanza "
                      f"({start_alberi} alberi già in memoria). Nessun reload da storage necessario.")
                all_trained_trees = cached
            else:
                # Cache assente o non coerente con start_alberi: questa istanza non ha
                # memoria diretta del progresso richiesto. Può essere un riavvio dopo
                # crash, oppure un nuovo leader subentrato dopo un fault di un'altra
                # istanza. In entrambi i casi il checkpoint fisico su S3 (fonte di
                # verità condivisa) è l'unico modo sicuro per recuperare lo stato:
                # qui avviene il vero, garantito, recovery cross-istanza.
                print(f"\n[{self.orchestrator_name}] [FAILOVER-RESUME] Nessuna cache locale valida per "
                      f"start_alberi = {start_alberi}. Ripristino checkpoint fisico da storage condiviso...")
                if self._trees_checkpoint_exists(self.current_job_id):
                    try:
                        # Ricompone dalle parti incrementali, con fallback
                        # automatico sul formato monolitico precedente.
                        all_trained_trees = self._load_trees_checkpoint(self.current_job_id)

                        print(f"[{self.orchestrator_name}] [OK] Ripristinati con successo {len(all_trained_trees)} alberi reali dal checkpoint.")
                        # Allineiamo lo start effettivo alla dimensione dell'array caricato per robustezza
                        start_alberi = len(all_trained_trees)
                    except Exception as e_load:
                        print(f"[{self.orchestrator_name}] [ERROR] Checkpoint fisico corrotto: {e_load}. Ricalcolo da 0.")
                        start_alberi = 0
                        all_trained_trees = []
                else:
                    print(f"[{self.orchestrator_name}] [WARN] File di checkpoint fisico non trovato a {checkpoint_trees_path}. Riparto da zero.")
                    start_alberi = 0

        # FIX MEMORIA (vedi OOM globale osservato sul container orchestratore/
        # test-engine con RSS fino a ~3.9GB): 'all_trained_trees' non deve più
        # restare l'unica fonte di 'n_features_in_'/'classes_' fino alla fine
        # del round -- li estraiamo qui in modo incrementale (running_*) man
        # mano che i batch vengono confermati su disco, cosi' gli alberi già
        # persistiti possono essere sostituiti con None (vedi più sotto) senza
        # perdere l'informazione che serve per il manifesto finale.
        # Se stiamo riprendendo da un checkpoint fisico (FAILOVER-RESUME),
        # 'all_trained_trees' contiene già alberi REALI e già durevoli su
        # disco per definizione (li abbiamo appena letti da lì): estraiamo
        # subito i metadati e liberiamo anche questi, invece di lasciarli
        # materializzati per il resto del round.
        running_n_features = [None]
        running_classes = set()
        if all_trained_trees:
            first_real = next((t for t in all_trained_trees if t is not None), None)
            if first_real is not None:
                running_n_features[0] = int(first_real.n_features_in_)
            trees_with_classes_resumed = [t for t in all_trained_trees if t is not None and hasattr(t, "classes_")]
            if trees_with_classes_resumed:
                resumed_classes = np.unique(np.concatenate(
                    [np.asarray(t.classes_) for t in trees_with_classes_resumed]))
                running_classes.update(resumed_classes.tolist())
            for _idx in range(len(all_trained_trees)):
                all_trained_trees[_idx] = None
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
                
        total_step_trees = target_alberi - start_alberi
        print(f"\n [{self.orchestrator_name}] Distribuzione carico: {total_step_trees} alberi da generare...")

        hp = payload.get("hyperparameters", {})
        max_depth = hp.get("max_depth", None)
        tree_type = hp.get("tree_type", "classifier")
        # Fallback allineato ai default "corretti" di RandomForest{Classifier,Regressor}
        # se il manifesto non lo specifica esplicitamente.
        max_features = hp.get("max_features", "sqrt" if tree_type == "classifier" else 1 / 3)
        min_samples_split = hp.get("min_samples_split", 2)
        # class_weight ha senso solo in classificazione: il worker lo ignora comunque
        # per i regressori, ma evitiamo di forzarlo se il payload non lo prevede.
        class_weight = hp.get("class_weight", None)
        criterion = hp.get("criterion", None)
        # Inoltrati esplicitamente al worker: prima non venivano trasmessi
        # affatto e ogni albero usava i valori di boot del worker
        # (self.bootstrap / self.max_samples), rendendo di fatto inerte quanto
        # dichiarato nel manifesto della baseline. None = "non specificato",
        # e il worker mantiene i propri valori di boot (comportamento storico).
        bootstrap = hp.get("bootstrap", None)
        max_samples = hp.get("max_samples", None)
        print(f"[{self.orchestrator_name}] Iperparametri effettivi -> n_estimators(step)={total_step_trees}, "
              f"max_depth={max_depth}, max_features={max_features}, min_samples_split={min_samples_split}, "
              f"criterion={criterion}, bootstrap={bootstrap}, max_samples={max_samples}")

        # Azzerati a ogni step: se questo step non costruisce alberi (caso
        # limite sotto) i valori devono restare 0.0 e non conservare quelli
        # dello step precedente.
        self.last_dispatch_seconds = 0.0
        self.last_aggregation_seconds = 0.0
        self.last_oob_seconds = 0.0

        # Caso limite: già finito tutto ma eravamo crashati prima di consolidare
        if total_step_trees <= 0:
            print(f"[{self.orchestrator_name}] Tutti gli alberi richiesti ({len(all_trained_trees)}) sono già pronti in memoria.")
        else:
            print(f"\n [{self.orchestrator_name}] Distribuzione carico residuo: {total_step_trees} alberi da generare...")
            dispatch_start = time.perf_counter()
            while True:
                available_workers = ServiceRegistry.get_available_workers(self.environment)
                if available_workers:
                    print(f"[{self.orchestrator_name}] Worker rilevati: {list(available_workers.keys())}. Procedo...")
                    break
                
                print(f"[{self.orchestrator_name}] Nessun worker disponibile. In Attesa...")
                time.sleep(10)

            worker_names = list(available_workers.keys())
            num_workers = len(worker_names)
            source_info = self.train_data_path 

            # 3. CALCOLO DINAMICO DELLA DIMENSIONE DEL CHUNK
            CHUNK_SIZE = int(np.ceil(total_step_trees / num_workers))
            print(f"[{self.orchestrator_name}] Calcolo dinamico: {num_workers} worker rilevati -> CHUNK_SIZE impostata a {CHUNK_SIZE} alberi per task.")

            # 4. Configurazione della Coda di Sotto-Task locale
            task_queue = queue.Queue()
            sub_start = start_alberi
            task_id_counter = start_alberi + 1
            
            while sub_start < target_alberi:
                sub_end = min(sub_start + CHUNK_SIZE, target_alberi)
                # Ogni sotto-task associa un seed specifico calcolato sull'offset cumulativo
                task_seed = seed + sub_start
                # Usiamo sub_start come offset assoluto rispetto al seed iniziale del JOB
                task_queue.put((task_id_counter, sub_start, sub_end, task_seed))
                task_id_counter += 1
                sub_start = sub_end

            results_lock = threading.Lock()

            # Lock DEDICATO alla persistenza del checkpoint, separato da
            # results_lock. Prima l'upload su S3 avveniva dentro results_lock,
            # cioè dentro la stessa sezione critica che serve ad accodare gli
            # alberi ricevuti: ogni worker che finiva restava fermo ad aspettare
            # la fine dell'upload di un altro, non per calcolare ma solo per
            # poter registrare il proprio risultato. Era un punto di
            # serializzazione che cresceva col numero di worker, e falsava
            # proprio la misura di strong scaling.
            checkpoint_lock = threading.Lock()
            # Contatore monotono dell'ultimo snapshot effettivamente persistito.
            # Serve a due cose:
            #  1) impedire che uno snapshot più VECCHIO sovrascriva uno più
            #     recente — ora che la scrittura è fuori da results_lock, due
            #     thread possono arrivarci in ordine diverso da quello in cui
            #     hanno preso lo snapshot, e un checkpoint che regredisce
            #     sposterebbe INDIETRO il punto di ripartenza dopo un guasto;
            #  2) saltare le scritture già superate. Se quando un thread ottiene
            #     il lock risulta già persistito uno snapshot con più alberi, il
            #     suo è ridondante: il checkpoint resta comunque più avanti, la
            #     tolleranza ai guasti non peggiora e si risparmiano byte.
            # "parts" riparte dal numero di parti gia'su storage: scrivere di nuovo
            # dalla 0 sovrascriverebbe un delta valido con un altro delta.
            last_checkpointed = {"count": start_alberi,
                                 "parts": self._count_trees_checkpoint_parts(self.current_job_id)}

            active_worker_names = list(worker_names)

            # Reset dell'evento (già usato in fase di inferenza): qui serve a far sì
            # che i test di fault injection possano attendere in modo affidabile il
            # momento in cui il PRIMO task di training viene davvero inviato a un
            # worker, invece di limitarsi a un'attesa temporale fissa.
            self.chunk_sent_event.clear()

            # 5. Definizione della funzione consumatrice per i thread
            def worker_thread_consumer(w_name):
                w_info = available_workers[w_name]
                worker_conn = None
                try:
                    print(f" [RPC -> {w_name}] Apertura connessione su {w_info['host']}:{w_info['port']}...")
                    worker_conn = rpyc.connect(
                        w_info["host"],
                        w_info["port"],
                        config={
                            'allow_pickle': True,
                            'sync_request_timeout': RPC_SYNC_TIMEOUT_SECONDS,
                            'keepalive': True
                        }
                    )
                    with self.connessioni_lock:
                        self.connessioni_attive.append(worker_conn)
                    
                    # ─── Il thread resta attivo finché non raccogliamo la quota di alberi globale ───
                    while len(all_trained_trees) < target_alberi:
                        try:
                            # Timeout breve (2 secondi) per controllare periodicamente lo stato e non restare appesi
                            task_id, start_t, end_t, chunk_seed = task_queue.get(timeout=2)
                        except queue.Empty:
                            # Se la coda è momentaneamente vuota ma mancano alberi al target globale,
                            # un altro worker attivo potrebbe crashare a breve e rimettere un task in coda.
                            # Usciamo solo se l'addestramento è finito o se siamo l'ultimo worker attivo rimasto.
                            with results_lock:
                                total_attuali = len(all_trained_trees)
                                num_worker_attivi = len(active_worker_names)
                            
                            if total_attuali >= target_alberi or num_worker_attivi <= 1:
                                break
                            time.sleep(1)
                            continue

                        quota_chunk = end_t - start_t
                        print(f"[{self.orchestrator_name}-Thread] Assegnazione Task {task_id} ({quota_chunk} alberi: {start_t}-{end_t}) a {w_name}")
                        self._track_task(task_id=task_id, job_id=self.current_job_id, worker_name=w_name, status="PROCESSING")
                        try:
                            self.chunk_sent_event.set()

                            # SHARDING DINAMICO: se attivo, ogni task riceve la
                            # fetta corrispondente a 'task_id % numero di shard'
                            # invece del dataset intero fisso. Il task_id
                            # sopravvive INTATTO al riaccodamento in caso di
                            # guasto (vedi 'task_queue.put((task_id, ...))' più
                            # sotto nel blocco except): un worker che ne
                            # sostituisce un altro morto ricalcola lo stesso
                            # identico shard, nessuna modifica necessaria alla
                            # logica di fault tolerance esistente.
                            if self.train_data_shards:
                                task_source_info = self.train_data_shards[task_id % len(self.train_data_shards)]
                            else:
                                task_source_info = source_info

                            ack_raw = worker_conn.root.train_subset_forest(
                                source_info=task_source_info,
                                num_trees=quota_chunk,       
                                base_seed=chunk_seed,    
                                max_depth=max_depth,
                                tree_type=hp.get("tree_type"),
                                max_features=max_features,
                                min_samples_split=min_samples_split,
                                class_weight=class_weight,
                                criterion=criterion,
                                bootstrap=bootstrap,
                                max_samples=max_samples
                            )

                            # Il worker NON restituisce più il blob degli alberi
                            # (fino a 1+ GB su scenari di scalabilità) come valore
                            # di ritorno RPC: lo ha già persistito nello storage
                            # condiviso (S3/locale) prima di rispondere, e qui ci
                            # limitiamo a un piccolo ack + rilettura diretta dallo
                            # storage. Evita l'hang osservato quando RPyC deve
                            # trasportare un payload sincrono molto grande come
                            # valore di ritorno (vedi Scenario 2 - Scalabilità).
                            ack = obtain(ack_raw)
                            if not isinstance(ack, dict) or not ack.get("ack"):
                                raise RuntimeError(
                                    f"Risposta inattesa dal worker {w_name} per il task {task_id}: {ack!r}"
                                )

                            # FIX: l'Orchestratore ricomponeva l'INTERO task in un
                            # colpo solo (load_task_trees_from_shared_storage), con
                            # 'tree_reconstruction_lock' a serializzare la
                            # ricomposizione tra thread ma senza limite alla
                            # dimensione del singolo task -- con pochi worker
                            # attivi CHUNK_SIZE sale (total_step_trees / num_workers)
                            # e un singolo task può arrivare a pesare oltre 1GB con
                            # max_depth=None su dataset grandi (misurato: ~72MB per
                            # albero su Friedman#1 1M righe). Ora leggiamo il task
                            # UNA PARTE ALLA VOLTA (iter_task_parts_as_tree_lists) e
                            # persistiamo+liberiamo ogni parte subito, prima di
                            # caricare la successiva: il picco di ricomposizione
                            # scende alla dimensione di UN batch worker, costante
                            # indipendentemente da quanto è grande CHUNK_SIZE.
                            part_iter = iter_task_parts_as_tree_lists(
                                task_source_info, chunk_seed, quota_chunk,
                                self.environment, self.orchestrator_name
                            )
                            received_any_part = False
                            while True:
                                with self.tree_reconstruction_lock:
                                    try:
                                        part_trees = next(part_iter)
                                    except StopIteration:
                                        break
                                    except FileNotFoundError:
                                        if received_any_part:
                                            # Già ricevuta almeno una parte: il task
                                            # NON è "non ancora pronto", è
                                            # genuinamente incompleto (una parte
                                            # attesa dal manifest manca). Errore vero,
                                            # non un semplice "aspetta ancora".
                                            #
                                            # NOTA SU UN CASO LIMITE RESIDUO: qui sotto
                                            # (except Exception as e, più in basso) il
                                            # task viene riaccodato PER INTERO come
                                            # prima di questo fix -- ma a differenza di
                                            # prima, ora alcune delle sue parti
                                            # potrebbero essere GIÀ state persistite nel
                                            # checkpoint dell'Orchestratore (quelle lette
                                            # con successo prima di questa). Un retry
                                            # completo del task rigenererebbe quegli
                                            # stessi alberi da capo, causando un doppio
                                            # conteggio. Nella pratica questo scenario
                                            # richiede che una parte manchi DOPO che il
                                            # manifest (scritto per ultimo, a garanzia
                                            # che tutte le parti siano già su disco) è
                                            # stato trovato -- una vera corruzione/
                                            # cancellazione esterna, non una race del
                                            # normale percorso di scrittura. Rischio
                                            # accettato consapevolmente per la modalità
                                            # 'a batch' richiesta; un hardening completo
                                            # (retry solo delle parti mancanti, non
                                            # dell'intero task) richiederebbe propagare
                                            # l'informazione "quante parti già lette" nella
                                            # coda dei task, fuori scope per questo fix.
                                            raise
                                        raise RuntimeError(
                                            f"Worker {w_name}: task {task_id} confermato (ack) ma il blob "
                                            f"non è stato trovato nello storage condiviso."
                                        )
                                received_any_part = True

                                # SEZIONE CRITICA MINIMA: solo l'aggiornamento della
                                # lista condivisa e uno snapshot immutabile. L'upload
                                # su S3 e la scrittura su DynamoDB, che prima stavano
                                # qui dentro, sono stati spostati FUORI: tenerli nel
                                # lock significava che ogni worker che finiva restava
                                # bloccato dietro l'upload di un altro solo per poter
                                # registrare il proprio risultato.
                                with results_lock:
                                    all_trained_trees.extend(part_trees)
                                    current_total = len(all_trained_trees)
                                    # list(...) crea una copia: la serializzazione fuori
                                    # dal lock non deve poter vedere la lista mutare.
                                    snapshot = list(all_trained_trees)
                                part_trees = None  # non serve più: droppa il riferimento

                                # --- fuori da results_lock ---
                                with checkpoint_lock:
                                    if current_total > last_checkpointed["count"]:
                                        try:
                                            # Scrive SOLO gli alberi nuovi (alla parte 0
                                            # l'intero snapshot, per migrare dal formato
                                            # monolitico). Traffico totale: N invece di N*(W+1)/2.
                                            prev_checkpointed = last_checkpointed["count"]
                                            self._persist_trees_delta(
                                                self.current_job_id, snapshot,
                                                prev_checkpointed, last_checkpointed["parts"])
                                            last_checkpointed["count"] = current_total
                                            last_checkpointed["parts"] += 1

                                            # FIX MEMORIA: estraiamo i metadati leggeri
                                            # (n_features_in_ una sola volta, classes_ per
                                            # union) dal SOLO batch appena persistito, PRIMA
                                            # di liberarlo -- non da tutto 'all_trained_trees'
                                            # (che a questo punto può già contenere molte
                                            # posizioni azzerate da batch precedenti).
                                            newly_persisted = snapshot[prev_checkpointed:current_total]
                                            if running_n_features[0] is None and newly_persisted:
                                                running_n_features[0] = int(newly_persisted[0].n_features_in_)
                                            trees_with_classes_batch = [t for t in newly_persisted if hasattr(t, "classes_")]
                                            if trees_with_classes_batch:
                                                batch_classes = np.unique(np.concatenate(
                                                    [np.asarray(t.classes_) for t in trees_with_classes_batch]))
                                                running_classes.update(batch_classes.tolist())

                                            # Liberiamo gli alberi appena confermati su
                                            # disco: _persist_trees_delta (vedi
                                            # BaseOrchestrator.py) per part_index >= 1 usa
                                            # SOLO 'snapshot[already_persisted:]' -- non
                                            # tocca mai più il prefisso già scritto, quindi
                                            # può restare fatto di soli 'None' (stessa
                                            # LUNGHEZZA, così lo slicing per posizione resta
                                            # corretto per le scritture successive) senza
                                            # rompere nulla. Questo è ciò che teneva
                                            # l'orchestratore a ridosso di diversi GB di RAM
                                            # con alberi non potati su dataset grandi.
                                            with results_lock:
                                                for _idx in range(prev_checkpointed, current_total):
                                                    all_trained_trees[_idx] = None
                                            snapshot = None

                                            # FIX MEMORIA (parte 2): 'gc.collect()' da
                                            # solo non basta -- CPython/glibc spesso NON
                                            # restituisce al sistema operativo la memoria
                                            # liberata (la tiene in riserva per riusarla
                                            # internamente), quindi l'RSS visto da
                                            # 'docker stats'/cgroup può restare alto anche
                                            # quando dentro il processo non è rimasto
                                            # nulla di vivo. 'malloc_trim(0)' (glibc,
                                            # Linux) chiede esplicitamente all'allocatore
                                            # di restituire i blocchi liberi all'OS: è
                                            # quello che chiude il cerchio tra "l'ho
                                            # liberato in Python" e "il container vede
                                            # meno RAM usata". Innocuo se non c'è nulla da
                                            # restituire (no-op), quindi sicuro da
                                            # chiamare ad ogni batch persistito senza
                                            # doverlo controllare a monte.
                                            gc.collect()
                                            try:
                                                ctypes.CDLL("libc.so.6").malloc_trim(0)
                                            except Exception:
                                                pass  # piattaforme non-glibc (es. macOS): nessun problema, solo nessun effetto

                                            # La cache di istanza ora referenzia la STESSA
                                            # lista già alleggerita (non una copia piena):
                                            # la continuazione same-process (STATE-SYNC)
                                            # resta valida per il conteggio, senza tenere
                                            # in vita gli alberi già persistiti.
                                            self._trees_cache[self.current_job_id] = all_trained_trees
                                            print(f"   [RPC <- {w_name}] [CHECKPOINT FS OK] Parte di Task {task_id} archiviata. Progressivo in RAM/Storage: {current_total} alberi.")
                                        except Exception as e_fs:
                                            # last_checkpointed NON avanza: un writer successivo
                                            # deve poter riprovare a persistere lo stato.
                                            print(f"   [ERRORE FILE SYSTEM] Impossibile scrivere gli alberi parziali su file: {e_fs}")

                                        # Il contatore logico segue lo stesso ordine monotono del
                                        # checkpoint fisico, così i due non possono divergere.
                                        if hasattr(self, 'state_manager') and self.state_manager:
                                            try:
                                                self.state_manager.update_request_status(
                                                    job_id=self.current_job_id,
                                                    status="PROCESSING",
                                                    orchestrator_id=self.orchestrator_name,
                                                    retries=payload.get("retries", 0),
                                                    base_random_state=seed,
                                                    alberi_addestrati=current_total
                                                )
                                            except Exception as e_db:
                                                print(f"   [ERRORE] Impossibile inviare l'heartbeat di stato a DynamoDB: {e_db}")
                                    else:
                                        # Snapshot superato: sullo storage c'è già uno stato con
                                        # PIÙ alberi, quindi riscriverlo non aggiungerebbe nulla e
                                        # anzi farebbe REGREDIRE il punto di ripartenza.
                                        print(f"   [RPC <- {w_name}] [CHECKPOINT SKIP] Parte di Task {task_id}: già persistito uno "
                                              f"stato più avanzato ({last_checkpointed['count']} alberi >= {current_total}).")

                            current_total = len(all_trained_trees)

                            print(f"   [RPC <- {w_name}] Task {task_id} completato. Ricevuti {quota_chunk} alberi (in {last_checkpointed['parts']} parti totali).")
                            self._track_task(task_id=task_id, job_id=self.current_job_id, worker_name=w_name, status="COMPLETED")
                            task_queue.task_done()
                            
                        except Exception as e:

                            self._track_task(task_id=task_id, job_id=self.current_job_id, worker_name=w_name, status="FAILED")
                            print(f"   [ERRORE RPC] Fallimento o disconnessione del worker {w_name} durante il Task {task_id}: {e}")
                            # FAULT TOLERANCE REALE: Reinserimento immediato del chunk per la fault tolerance
                            task_queue.put((task_id, start_t, end_t, chunk_seed))
                            print(f"[{self.orchestrator_name}-Thread] Task {task_id} riaccodato con successo per il failover.")
                            
                            with results_lock:
                                if w_name in active_worker_names:
                                    active_worker_names.remove(w_name)
                            break  # Il canale RPC con questo worker è saltato, chiudiamo il thread relativo
                        
                except Exception as conn_err:
                    print(f"   [ERRORE CRITICO] Impossibile connettersi a {w_name}: {conn_err}")
                    with results_lock:
                        if w_name in active_worker_names:
                            active_worker_names.remove(w_name)
                finally:
                    if worker_conn:
                        with self.connessioni_lock:
                            if worker_conn in self.connessioni_attive:
                                self.connessioni_attive.remove(worker_conn)
                        try:
                            worker_conn.close()
                        except Exception:
                            pass

            # 6. Avvio dei thread
            # FIX: prima tutti i thread partivano in sequenza stretta, senza
            # nessuna pausa -- ogni thread, appena avviato, chiama subito
            # train_subset_forest sul worker, che a sua volta carica l'intero
            # dataset condiviso (source_info, fino a 1M righe) in _load_data.
            # Con N worker tutti avviati nello stesso istante, si ottengono N
            # caricamenti simultanei dello stesso CSV -- un picco di memoria
            # sincronizzato su tutti i container, causa più probabile degli
            # OOM quasi-simultanei osservati su quasi tutti i worker nello
            # scenario di scalabilità col dataset sintetico da 1M campioni.
            # Una piccola pausa tra un avvio e l'altro spalma questo picco nel
            # tempo invece di sincronizzarlo: costo totale trascurabile
            # rispetto al training (con 10 worker, meno di 3s), ma i
            # caricamenti si accavallano molto meno. Non elimina il problema
            # se il dataset è enorme o il mem_limit troppo stretto, ma riduce
            # sensibilmente la probabilità del crash sincronizzato visto nei
            # test. Vale SOLO per il primo task di ogni worker: dal secondo in
            # poi il dataset è già in cache locale (self._cached_X/_cached_y
            # in CentralizedWorker), quindi non ricarica nulla e la pausa non
            # si ripete.
            WORKER_START_STAGGER_SECONDS = 0.3
            threads = []
            for i, name in enumerate(worker_names):
                if i > 0:
                    time.sleep(WORKER_START_STAGGER_SECONDS)
                t = threading.Thread(target=worker_thread_consumer, args=(name,))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            # Fine della fase di costruzione degli alberi: da qui in poi è solo
            # ricomposizione e diagnostica. Questo è il tempo direttamente
            # confrontabile con T_seq/T_1node della baseline locale.
            self.last_dispatch_seconds = time.perf_counter() - dispatch_start
            print(f"[DEBUG TIMING] Costruzione distribuita degli alberi completata in "
                  f"{self.last_dispatch_seconds:.2f}s ({len(all_trained_trees)} alberi, "
                  f"{num_workers} worker).")

            # 7. Monitoraggio fallimento totale dello step
            if not task_queue.empty() and len(active_worker_names) == 0:
                print(f"   [{self.orchestrator_name}] Tutti i worker sono crashati. SQS gestirà il failover macro.")
                raise RuntimeError("Sotto-sistema Fault Tolerance interrotto: Nessun worker disponibile rimasto.")

            # Chiusura pulita delle connessioni
            print(f"[*] Pulizia risorse: chiusura di {len(self.connessioni_attive)} connessioni RPyC residue...")
            with self.connessioni_lock:
                for conn in self.connessioni_attive:
                    try: conn.close()
                    except Exception: pass

        # 8. Costruzione del manifesto del modello globale (streaming: nessun
        #    assemblaggio scikit-learn completo in RAM)
        if len(all_trained_trees) > 0:
            print(f"   [{self.orchestrator_name}] Costruzione del manifesto del modello globale "
                  f"(streaming, nessun assemblaggio scikit-learn completo in RAM)...")
            aggregation_start = time.perf_counter()
            try:
                n_features = running_n_features[0]
                if n_features is None:
                    # Fallback di sicurezza: non dovrebbe succedere (ogni batch
                    # persistito con successo aggiorna running_n_features), ma se
                    # per qualche motivo un albero fosse rimasto materializzato
                    # (es. un salvataggio fallito, mai liberato) lo recuperiamo
                    # da lì come ultima spiaggia.
                    first_real_tree = next((t for t in all_trained_trees if t is not None), None)
                    n_features = int(first_real_tree.n_features_in_) if first_real_tree is not None else None
                n_estimators = len(all_trained_trees)

                classes_list = None
                n_classes = None
                if tree_type == "classifier":
                    # Classi accumulate in modo incrementale (running_classes)
                    # man mano che ogni batch veniva persistito e liberato,
                    # invece di rileggerle qui da 'all_trained_trees' (che a
                    # questo punto contiene quasi solo None -- vedi fix
                    # memoria più sopra nel ciclo di dispatch).
                    if running_classes:
                        detected_classes = np.array(sorted(running_classes), dtype=np.int64)
                    else:
                        print(f"   [{self.orchestrator_name}] [WARN] Nessun albero espone 'classes_'. Fallback su {{0, 1}}.")
                        detected_classes = np.array([0, 1])
                    classes_list = detected_classes.astype(np.int64).tolist()
                    n_classes = len(classes_list)

                # Manifesto leggero invece del modello scikit-learn intero: gli
                # alberi restano dove sono già (le parti scritte incrementalmente
                # da _persist_trees_delta durante il dispatch, poco sopra) —
                # evita di pickle-are ~7 GiB di alberi già vivi in RAM (che
                # raddoppierebbero temporaneamente il picco: causa dell'OOM
                # osservato empiricamente proprio a questo punto, con 10
                # worker/100 alberi). L'inferenza e la stima OOB rileggono le
                # parti in streaming (_iter_checkpoint_trees/
                # _iter_checkpoint_tree_ranges in BaseOrchestrator.py) invece
                # di ricostruire l'oggetto RandomForest completo.
                manifest = {
                    "n_estimators": n_estimators,
                    "n_features_in_": int(n_features),
                    "n_outputs_": 1,
                    "tree_type": tree_type,
                    "classes_": classes_list,
                    "n_classes_": n_classes,
                    "job_id": self.current_job_id,
                }

                model_path = self._resolve_model_path(self.current_job_id)
                self.checkpoint_dao.save(model_path, manifest)

                print(f"   [{self.orchestrator_name}] Manifesto del modello salvato con successo in "
                      f"'{model_path}' ({n_estimators} alberi, referenziati dalle parti già persistite).")

                self.last_aggregation_seconds = time.perf_counter() - aggregation_start
                print(f"[DEBUG TIMING] Costruzione manifesto e salvataggio: "
                      f"{self.last_aggregation_seconds:.2f}s.")

                # Libera esplicitamente la foresta materializzata dalla RAM: da
                # qui in poi (stima OOB) si rilegge in streaming dallo storage,
                # non serve più tenerla viva. Include anche la cache cross-round
                # (self._trees_cache), che altrimenti manterrebbe un riferimento
                # vivo indipendentemente da questa variabile locale — sicuro da
                # rimuovere qui perché il training di QUESTO job è concluso,
                # nessun round futuro potrà mai più averne bisogno.
                self._trees_cache.pop(self.current_job_id, None)
                n_trees_for_report = len(all_trained_trees)
                del all_trained_trees
                gc.collect()

                # NOTA (12/9/2026): la stima OOB è stata rimossa interamente
                # (era qui, con un fallback sequenziale + un percorso
                # distribuito sperimentale). last_oob_seconds resta 0.0 dal
                # reset a inizio metodo - nessun ricalcolo necessario.
                print(f"[DEBUG TIMING] Riepilogo _execute_training_step -> "
                      f"ETL {self.last_etl_seconds:.2f}s | costruzione alberi "
                      f"{self.last_dispatch_seconds:.2f}s | aggregazione "
                      f"{self.last_aggregation_seconds:.2f}s | OOB {self.last_oob_seconds:.2f}s")

                # ─── MODIFICA 3: Restituiamo la dimensione REALE degli alberi salvati ───
                return n_trees_for_report
                
            except Exception as e:
                print(f"   [ERRORE AGGREGAZIONE] Fallimento durante la costruzione del manifesto: {e}")
                traceback.print_exc()
                return len(all_trained_trees)

        print(f"   [{self.orchestrator_name}] Nessun albero collezionato.")
        # ─── Ritorna 0 se non è stato possibile generare o caricare nulla ───
        return 0
    
    def _execute_inference_step(self, payload: dict) -> dict:
        """
        Esegue l'inferenza distribuita centralizzata in modalità Fault-Tolerant
        sfruttando una task queue concorrente per riallocare dinamicamente i blocchi in caso di crash.
        """
        job_id = payload.get("job_id")
        hp = payload.get("hyperparameters", {})
        tree_type = hp.get("tree_type", "classifier")
        target_col = "Target" if tree_type == "regressor" else "Label"
        dataset_type = self._resolve_dataset_type(payload)

        print(f"\n[{self.orchestrator_name}] === AVVIO INFERENZA DISTRIBUITA CENTRALIZZATA FAULT-TOLERANT ===")
        inference_start_time = time.perf_counter()
        model_path = self._resolve_model_path(job_id)

        # 1. RISOLUZIONE DINAMICA FILE MODELLO (.pkl) E TESTING SET (.csv) IN BASE ALL'AMBIENTE
        if self.environment == "aws":
            self.test_data_path = f"s3://{BUCKET_NAME}/distributed_tests/shared_test_{job_id}.csv"
        else:
            self.test_data_path = f"./.local_storage/shared_test_{job_id}.csv"

        print(f"[{self.orchestrator_name}] [AUTO-RESOLVE] Modello: {model_path} | Test Data: {self.test_data_path}")

        # 2. CARICAMENTO DEL MANIFESTO DEL MODELLO (non più l'oggetto scikit-learn
        #    completo: gli alberi restano sulle parti già persistite durante il
        #    training, letti in streaming più sotto — evita di deserializzare
        #    l'intera foresta (~7 GiB in questo scenario) in un colpo solo,
        #    prima ancora di iniziare a smistare i chunk ai worker)
        if not self.checkpoint_dao.exists(model_path):
            raise FileNotFoundError(f"Modello globale non trovato in '{model_path}'.")
        print(f"[{self.orchestrator_name}] Caricamento del manifesto del modello da {model_path}...")
        manifest = self.checkpoint_dao.load(model_path)
        total_trees = manifest["n_estimators"]
        print(f"[{self.orchestrator_name}] Manifesto caricato. Numero totale di alberi: {total_trees}")

        # Spazio di classi GLOBALE (calcolato in fase di training su TUTTI gli
        # alberi, salvato nel manifesto): serve ai worker per allineare le
        # colonne di predict_proba di ogni singolo albero, anche quando un
        # albero non ha visto tutte le classi nel proprio campione bootstrap.
        global_classes = manifest.get("classes_") if tree_type == "classifier" else None

        # 3. CARICAMENTO E PREPARAZIONE DEL DATASET DI TEST TRAMITE DAO
        print(f"[{self.orchestrator_name}] Caricamento Testing Set persistito via DAO: {self.test_data_path}")
        dao = DatasetDAOFactory.get_dao(self.environment)
        test_df = dao.load_dataset(self.test_data_path)

        print(f"[{self.orchestrator_name}] Preparazione della matrice di test (Shape: {test_df.shape})...")
        actual_target = target_col if target_col in test_df.columns else ("Target" if "Target" in test_df.columns else "Label")
        if actual_target != target_col:
            print(f"[{self.orchestrator_name}] [WARN] Colonna target attesa '{target_col}' non trovata nel test set. "
            f"Uso '{actual_target}' come fallback.")
        X_test = test_df.drop(columns=[actual_target]).to_numpy(dtype=np.float64)
        y_test = test_df[actual_target].to_numpy()
        serialized_X_test = pickle.dumps(X_test)
        X_test_key = f"inference_testset/{job_id}.pkl"
        save_bytes_to_shared_storage(X_test_key, serialized_X_test, self.environment, self.orchestrator_name)

        # 4. SCOPERTA WORKER E INIZIALIZZAZIONE STRUTTURE FAULT-TOLERANT
        available_workers = ServiceRegistry.get_available_workers(self.environment)
        if not available_workers:
            raise RuntimeError("Nessun worker disponibile nel Service Registry per l'inferenza.")

        worker_names = list(available_workers.keys())
        num_workers = len(worker_names)
        print(f"[{self.orchestrator_name}] Worker pronti per l'inferenza: {num_workers} -> {worker_names}")

        # Calcolo dinamico granulare della dimensione del chunk di alberi
        CHUNK_SIZE = int(np.ceil(total_trees /num_workers))
        print(f"[{self.orchestrator_name}] CHUNK_SIZE di inferenza impostata a {CHUNK_SIZE} alberi per task.")

        # Popolamento della coda thread-safe dei sotto-task di inferenza. I
        # blocchi vengono letti IN STREAMING dalle parti già persistite
        # durante il training (_iter_checkpoint_tree_ranges in
        # BaseOrchestrator.py), non affettati da una lista 'all_trees' già
        # interamente materializzata: al più ~2 parti vive in RAM per volta,
        # mai l'intera foresta.
        task_queue = queue.Queue()
        task_id_counter = 0
        predictions_chunks = self._load_inference_checkpoint(job_id)  # Tentativo di ripristino da checkpoint
        already_done_ranges = {start for start, _ in predictions_chunks}
        results_lock = threading.Lock()
       
        active_worker_names = list(worker_names)
        self.chunk_sent_event.clear()   # <-- reset, così ogni run è pulita
        for tree_start, tree_end, chunk_estimators in self._iter_checkpoint_tree_ranges(job_id, CHUNK_SIZE):
            if tree_start not in already_done_ranges:
                # FIX: prima l'intero chunk (fino a CHUNK_SIZE alberi, oltre
                # 1GB con pochi worker attivi) veniva serializzato e scritto
                # come UN blob unico -- il worker doveva poi scaricarlo e
                # deserializzarlo tutto insieme (vedi
                # exposed_predict_subset_forest in BaseWorker.py), tenendo
                # contemporaneamente in RAM sia i byte grezzi sia gli oggetti
                # albero appena decodificati: causa dell'OOM osservato sui
                # worker in fase di inferenza dopo aver ridotto NUM_WORKERS
                # (stesso identico problema già risolto lato Orchestratore
                # per la ricomposizione dei task di training, qui speculare
                # sul lato worker). Scriviamo ora il chunk a piccole parti
                # (stesso pattern manifest+parti del training): il worker
                # legge, predice e libera una parte alla volta.
                chunk_key_prefix = f"inference_chunks/{job_id}/chunk_{tree_start}_{tree_end}"
                save_chunk_in_parts_to_shared_storage(
                    chunk_key_prefix, chunk_estimators, self.environment, self.orchestrator_name
                )
                task_queue.put((task_id_counter, tree_start, tree_end, chunk_key_prefix))
                task_id_counter += 1
            else: 
                print(f"[SHORT-CIRCUIT] Chunk {tree_start}-{tree_end} già completato. Skip.")


        # Strutture dati condivise protette da Lock per i thread consumatori
        MAX_RETRIES_PER_TASK = 3  # Numero massimo di tentativi per ogni sotto-task prima di considerarlo fallito
        task_retries = {}  # Dizionario per tracciare i tentativi per ogni task_id

        failed_tasks = set()
        # 5. DEFINIZIONE DEL CONSUMATORE CONCORRENTE PER L'INFERENZA VIA RPC
        def inference_worker_consumer(w_name):
            rounds_done = 0
            w_info = available_workers[w_name]
            worker_conn = None
            try:
                print(f" [RPC INF -> {w_name}] Apertura connessione su {w_info['host']}:{w_info['port']}...")
                worker_conn = rpyc.connect(
                    w_info["host"],
                    w_info["port"],
                    config={
                        'allow_pickle': True,
                        'sync_request_timeout': RPC_INFERENCE_SYNC_TIMEOUT_SECONDS,
                        'keepalive': True
                    }
                )
                with self.connessioni_lock:
                    self.connessioni_attive.append(worker_conn)
                
                while True:
                    try:
                        task_id, start_idx, end_idx, chunk_key_prefix = task_queue.get(timeout=2)
                        rounds_done += 1
                    except queue.Empty:
                        break

                    quota_alberi = end_idx - start_idx
                    print(f"[{self.orchestrator_name}-InfThread] Assegnazione Task {task_id} ({quota_alberi} alberi: {start_idx}-{end_idx}) a {w_name}")
                    self._track_task(task_id=task_id, job_id=job_id, worker_name=w_name, status="PROCESSING")
                    try:
                        self.chunk_sent_event.set()
                        
                        # Invocazione remota sul metodo esposto dal BaseWorker:
                        # 'chunk_key_prefix' non è più la chiave di UN blob, ma il
                        # prefisso di un manifest+parti (vedi
                        # save_chunk_in_parts_to_shared_storage sopra e
                        # iter_chunk_parts_from_shared_storage lato worker): il
                        # worker legge, predice e libera una parte alla volta,
                        # invece di scaricare e deserializzare l'intero chunk in
                        # un colpo solo.
                        raw_response = worker_conn.root.predict_subset_forest(
                            chunk_key_prefix,
                            X_test_key,
                            tree_type,
                            global_classes
                        )
                        sub_predictions = pickle.loads(obtain(raw_response))
                        
                        with results_lock:
                            # Tracciamo start_idx per poter riordinare sequenzialmente i blocchi alla fine
                            predictions_chunks.append((start_idx, sub_predictions))
                            inference_cp_path = self._get_inference_checkpoint_path(job_id)
                            try:
                               self.checkpoint_dao.save(inference_cp_path, predictions_chunks)
                               print(f"   [RPC INF <- {w_name}] [CHECKPOINT INFERENZA OK] Task {task_id} archiviato. Progressivo in RAM/Storage: {len(predictions_chunks)} chunk.")
                            except Exception as e_fs:
                                print(f"   [ERRORE FILE SYSTEM] Impossibile scrivere i chunk di inferenza parziali su file: {e_fs}")
                            
                        print(f"   [RPC INF <- {w_name}] Task {task_id} completato con successo.")
                        self._track_task(task_id=task_id, job_id=job_id, worker_name=w_name, status="COMPLETED")
                        task_queue.task_done()
                        
                    except Exception as e:
                        print(f"   [ERRORE RPC INFERENZA] Fallimento del worker {w_name} sul Task {task_id}: {e}")
                        retries = task_retries.get(task_id, 0) + 1
                        task_retries[task_id] = retries
                        if retries > MAX_RETRIES_PER_TASK:
                            # Segnaliamo il fallimento permanente invece di loopar all'infinito
                            print(f"[FATAL] Task {task_id} ha superato il limite di {MAX_RETRIES_PER_TASK} retry. Abort.")
                            self._track_task(task_id=task_id, job_id=job_id, worker_name=w_name, status="FAILED")
                            failed_tasks.add(task_id)
                            task_queue.task_done()
                        else:
                            # FAILOVER: Inserimento immediato del task interrotto nuovamente in coda
                            self._track_task(task_id=task_id, job_id=job_id, worker_name=w_name, status="REQUEUED")
                            task_queue.put((task_id, start_idx, end_idx, chunk_key_prefix))
                            print(f"[{self.orchestrator_name}-InfThread] Task {task_id} riaccodato per il failover.")
                        
                        with results_lock:
                            if w_name in active_worker_names:
                                active_worker_names.remove(w_name)
                        break  # Interruzione del loop per questo canale RPC corrotto
                print(f"[{w_name}] ha completato {rounds_done} round")       
            except Exception as conn_err:
                print(f"   [ERRORE CONNESSIOINE INFERENZA] Impossibile raggiungere il worker {w_name}: {conn_err}")
                with results_lock:
                    if w_name in active_worker_names:
                        active_worker_names.remove(w_name)
            finally:
                if worker_conn:
                    with self.connessioni_lock:
                        if worker_conn in self.connessioni_attive:
                            self.connessioni_attive.remove(worker_conn)
                    try:
                        worker_conn.close()
                    except Exception:
                        pass
         
        # 6. AVVIO MULTI-THREADING E SINCRONIZZAZIONE DEI CONSUMATORI
        rpc_start_time = time.perf_counter()
        threads = []
        for name in worker_names:
            t = threading.Thread(target=inference_worker_consumer, args=(name,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        
        try:
            if failed_tasks:
                raise RuntimeError(f"Inferenza parziale: {len(failed_tasks)} chunk non completati.")
            if not task_queue.empty():
                raise RuntimeError("Task in coda orfani: tutti i worker sono crashati.")
        finally:
            with self.connessioni_lock:
                for conn in self.connessioni_attive:
                    try: conn.close()
                    except Exception: pass

        rpc_inference_time = time.perf_counter() - rpc_start_time

        # 7. ORDINAMENTO SEQUENZIALE E COMPOSIZIONE DELLA MATRICE DELLE PREDIZIONI
        print(f"[{self.orchestrator_name}] Collezionamento predizioni completato. Ricomposizione matrice in corso...")
        predictions_chunks.sort(key=lambda x: x[0])
        
        all_worker_predictions = []
        for _, sub_preds in predictions_chunks:
            all_worker_predictions.extend(sub_preds)

        predictions_matrix = np.array(all_worker_predictions)
        print(f"[{self.orchestrator_name}] Matrice complessiva delle predizioni rigenerata: {predictions_matrix.shape}")
        
        total_inference_time = time.perf_counter() - inference_start_time

        # Soglia di decisione calibrata dalla baseline (vedi
        # VALIDATION_SIZE_FOR_THRESHOLD/decision_threshold in run_baseline.py):
        # letta da config_<dataset_type>.json invece di ricadere sull'argmax
        # implicito (soglia 0.50) di _aggregate_forest_predictions. None se il
        # manifesto non la contiene ancora (fallback automatico al comportamento
        # precedente, vedi read_decision_threshold_from_config).
        decision_threshold = (
            self.read_decision_threshold_from_config(dataset_type)
            if tree_type == "classifier" else None
        )

        # 8. AGGREGAZIONE: SOFT VOTING (media delle probabilità per-albero) per la
        # classificazione, a decision_threshold se disponibile (altrimenti argmax,
        # invariato), MEDIA per la regressione, seguita dal calcolo delle metriche
        # sulla predizione finale.
        final_predictions, y_probs = self._aggregate_forest_predictions(
            predictions_matrix=predictions_matrix,
            tree_type=tree_type,
            global_classes=global_classes,
            decision_threshold=decision_threshold,
        )
        metrics = self.calculate_metrics(
            final_predictions=final_predictions,
            y_test=y_test,
            tree_type=tree_type,
            y_probs=y_probs
        )
        self._save_metrics(job_id, "inference", {
            "job_id": job_id, "mode": "centralized", "phase": "inference",
            "tree_type": tree_type, "testing_set_size": X_test.shape[0],
            "timings": {"total_inference_time": total_inference_time, "rpc_inference_time": rpc_inference_time},
            "metrics": metrics
        })
        if hasattr(self, 'state_manager') and self.state_manager:
            try:
                self.state_manager.update_request_status(
                    job_id=job_id,
                    status="COMPLETED",
                    orchestrator_id=self.orchestrator_name,
                    alberi_addestrati=total_trees,
                )
            except Exception as e_db:
                print(f"   [ERRORE] Impossibile scrivere lo stato COMPLETED su DynamoDB/local: {e_db}")

        # Esposto esplicitamente (prima mancava): chi chiama questo metodo — inclusa
        # la suite di test locale — deve poter leggere le metriche reali dal valore di
        # ritorno, invece di affidarsi a un monkey-patch interno fragile.
        return {
            "status": "SUCCESS" if not failed_tasks else "PARTIAL",
            "testing_set_size": int(X_test.shape[0]),
            "total_inference_time": total_inference_time,
            "rpc_inference_time": rpc_inference_time,
            "metrics": metrics
        }

    def _save_checkpoint(self, job_id: str, current_alberi: int, retries: int, base_random_state: int, alberi_reali: list = None):
        """
        Estende il checkpoint della classe base aggiungendo il salvataggio FISICO
        degli alberi (specifico del calcolo centralizzato).
        """
        # 1. Chiamiamo la classe base per aggiornare DynamoDB (evita duplicazione di codice)
        super()._save_checkpoint(job_id, current_alberi, retries, base_random_state)
        
        # 2. Se ci sono alberi fisici da blindare su disco/S3, lo facciamo qui
        if alberi_reali is not None and len(alberi_reali) > 0:
            try:
                # Sostituzione integrale dello stato: si azzera e si riscrive come
                # parte 0. Percorso oggi mai esercitato — BaseOrchestrator chiama
                # _save_checkpoint senza 'alberi_reali' — ma va tenuto coerente
                # col formato a parti, altrimenti reintrodurrebbe un monolitico.
                self._purge_trees_checkpoint(job_id)
                self._persist_trees_delta(job_id, alberi_reali, 0, 0)
                print(f"[{self.orchestrator_name}] [CENTRALIZED-CHECKPOINT-FISICO] {len(alberi_reali)} alberi salvati in storage.")
            except Exception as e:
                print(f"[{self.orchestrator_name}] [ERRORE STORAGE] Fallito salvataggio fisico degli alberi: {e}")

    def _clean_checkpoint(self, job_id: str):
        """
        Override del metodo di pulizia per rimuovere il file pickle parziale.

        BUG CORRETTO (7/9/2026): questo metodo cancellava SEMPRE le parti del
        checkpoint alberi (_purge_trees_checkpoint) subito dopo il
        completamento di un job riuscito -- corretto PRIMA dell'introduzione
        del manifesto leggero (vedi _execute_training_step), quando il
        modello finale era un pickle scikit-learn autosufficiente e quelle
        parti erano davvero solo stato temporaneo di resume tra i round.

        Dopo il manifesto leggero, il modello NON contiene più gli alberi:
        salva solo metadati e RIFERISCE le parti già persistite su storage
        (vedi il commento "referenziati dalle parti già persistite" al
        momento del salvataggio). Cancellarle qui distrugge l'unica copia
        reale del modello subito dopo averlo "salvato" -- bug osservato
        empiricamente il 7/9/2026: ogni inferenza su un job addestrato con
        questo formato falliva con "ricevuta shape (0,)", perché tutte le
        parti erano già state rimosse nello stesso istante in cui il
        training terminava.

        Le parti ora sopravvivono al completamento del job, esattamente come
        saved_models/model_{job_id}.pkl -- la pulizia esplicita di un modello
        non più necessario resta una scelta dell'utente (es. teardown.sh
        --purge-models), mai automatica a fine training.
        """
        super()._clean_checkpoint(job_id)
        self._trees_cache.pop(job_id, None)
 
        inference_cp = self._get_inference_checkpoint_path(job_id)
        try:
            self.checkpoint_dao.delete(inference_cp)
        except Exception as e:
            print(f"[{self.orchestrator_name}] [CLEAN WARN] Impossibile cancellare {inference_cp}: {e}")
    
    def _resolve_trees_checkpoint_path(self, job_id: str) -> str:
        
        if self.environment == "aws":
            return f"s3://{BUCKET_NAME}/checkpoints/checkpoint_trees_{job_id}.pkl"
        return f"./.local_storage/checkpoint_trees_{job_id}.pkl"
    
    def _resolve_model_path(self, job_id: str) -> str:
        """Path del modello globale aggregato, in una sotto-cartella dedicata alla
        modalità centralizzata per evitare collisioni col modello federato in caso
        di job_id riutilizzati tra le due modalità."""
        if self.environment == "aws":
            return f"s3://{BUCKET_NAME}/saved_models/centralized/model_{job_id}.pkl"
        return os.path.join("./saved_models", f"model_{job_id}.pkl")
    
    
    def _get_inference_checkpoint_path(self, job_id: str) -> str:
        if self.environment == "aws":
            return f"s3://{BUCKET_NAME}/checkpoints/inference_chunks_{job_id}.pkl"
        return f"./.local_storage/inference_chunks_{job_id}.pkl"
    
    
    def _load_inference_checkpoint(self, job_id: str):
        path = self._get_inference_checkpoint_path(job_id)
        if self.checkpoint_dao.exists(path):
            try:
                chunks = self.checkpoint_dao.load(path)
                print(f"[{self.orchestrator_name}] [LOAD CHECKPOINT INFERENZA] Caricati {len(chunks)} chunk di inferenza dal checkpoint.")
                return chunks
            except Exception as e:
                print(f"[{self.orchestrator_name}] [LOAD CHECKPOINT INFERENZA] Errore nel caricamento del checkpoint: {e}")
        return []
    
if __name__ == "__main__":
    print("[BOOT] Avvio del nodo Orchestratore Centralizzato...")
    orchestrator = CentralizedOrchestrator()
    orchestrator.start()