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
BUCKET_NAME = os.environ.get("DATASETS_BUCKET_NAME", "rf-distributed-datasets-383056860320-us-east-1")
# Stesso valore di run_baseline.py (TARGET_ROWS_PER_DAY): campionamento
# RIBILANCIATO per giorno di cattura invece di una sample_fraction uniforme
# sull'intero dataset (che farebbe dominare il campione dal giorno più
# grande -- vedi RawCSVDataLoader per la motivazione completa). Da questo
# valore dipende anche il ribilanciamento locale automatico dei due giorni
# "protetti" (Infiltration), attivo dentro RawCSVDataLoader ogni volta che
# target_rows_per_day non è None -- nessuna configurazione aggiuntiva
# richiesta qui per beneficiarne.
TARGET_ROWS_PER_DAY = 200_000
# SHARDING FISSO REALE: solo per tree_type == "classifier" (dataset reale, CICIDS).
# A differenza del dataset sintetico, qui il numero di shard è FISSO e indipendente dai 
# worker attivi. Questo permette di riusare gli stessi file shard tra configurazioni 
# diverse (es. test di scalabilità con 3/5/7/10 worker), abbattendo i tempi ETL.
# Il valore 20 è calcolato raddoppiando il numero massimo di worker previsti (10):
# garantisce che la configurazione più grande riceva un numero di shard perfettamente 
# divisibile (2 per worker), evitando sbilanciamenti nel dataset.
REAL_FIXED_SHARDS = 20
# Stessi valori di run_baseline.py: senza allinearli qui, il train
# distribuito e quello della baseline locale sarebbero addestrati su
# distribuzioni/feature-set diversi, invalidando il confronto delle
# METRICHE (quello sui tempi resterebbe comunque valido, essendo
# indipendente da questi due parametri).
UNDERSAMPLING_RATIO = 1.0
# Allineato a run_baseline.py: 15% del train ritagliato PRIMA dell'undersampling.
# Anche se qui non viene usato per calibrare la soglia, va rimosso PER VOLUME: 
# garantisce che l'undersampling lavori su un pool di partenza identico a 
# quello della baseline, producendo lo stesso esatto set di dati.
VALIDATION_SIZE_FOR_THRESHOLD = 0.15


RPC_SYNC_TIMEOUT_SECONDS = env_timeout_seconds("RPC_SYNC_TIMEOUT_SECONDS", 600)
RPC_INFERENCE_SYNC_TIMEOUT_SECONDS = env_timeout_seconds("RPC_INFERENCE_SYNC_TIMEOUT_SECONDS", 600)

class CentralizedOrchestrator(BaseOrchestrator):
    def __init__(self, orchestrator_name: str = None):
        self.cfg = SystemConfig()
        name = orchestrator_name or f"Orchestrator-Centralizzato-{socket.gethostname()}"

        self.current_job_id = None
        self.train_data_path = None
        self.test_data_path = None
       # SHARDING DINAMICO: 
        # - None: modalità 'shared' (default, ogni worker scarica l'intero dataset). 
        # - Lista di path: modalità 'sharded', ogni worker scarica SOLO la propria fetta, 
        #   assegnata dinamicamente per task_id (vedi _execute_training_step). 
        # Toggle via variabile d'ambiente CENTRALIZED_DATASET_MODE ('shared'|'sharded').
        self.train_data_shards = None
        # CACHE DELL'INTERMEDIO: salva train_df/test_df dopo l'ETL pesante (download, 
        # binarizzazione, split, preprocessing, undersampling) ma PRIMA dello sharding finale.
        # Round successivi con identici base_seed/dataset_type/tree_type riusano questa cache:
        # se cambia solo num_shards (es. test di scalabilità), viene rieseguito solo lo 
        # shuffle e la scrittura finale, abbattendo i tempi.
        self._cached_prepared_key = None
        self._cached_prepared_train_df = None
        self._cached_prepared_test_df = None
        self.chunk_sent_event = threading.Event()
        self._trees_cache = {}
        # Durata dell'ultima fase di preparazione dati (ETL). Serve agli scenari
        # di test per scomporre il tempo totale in "preparazione dati" +
        # "addestramento distribuito": la baseline locale misura t_seq sul solo
        # fit, quindi confrontarla con un totale che include l'ETL (30-40s su
        # AWS per via di S3) penalizzerebbe sistematicamente il cluster.
        # 0.0 quando l'ETL viene saltata grazie allo SHORT-CIRCUIT.
        self.last_etl_seconds = 0.0
        # Scomposizione del tempo di _execute_training_step per garantire un confronto
        # equo con la baseline locale (che misura il solo fit di scikit-learn escludendo 
        # l'overhead di rete e I/O che penalizzerebbe il cluster).
        #
        #   last_dispatch_seconds      costruzione vera degli alberi: scoperta worker, 
        #                              invio chunk via RPC e attesa. È il numero da 
        #                              confrontare con T_seq/T_1node.
        #   last_aggregation_seconds   ricomposizione della foresta e salvataggio.
        #   last_oob_seconds           attributo legacy (stima OOB rimossa), mantenuto 
        #                              sempre a 0.0 per retrocompatibilità con lo schema 
        #                              dei report (scalability.py).
        #
        # Totale _execute_training_step ~= last_etl_seconds + last_dispatch_seconds + last_aggregation_seconds
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
        target_col = "Target" if tree_type == "regressor" else "Label"

        prepared_key = (dataset_type, tree_type, base_seed, dataset_path)
        if (self._cached_prepared_key == prepared_key
                and self._cached_prepared_train_df is not None
                and self._cached_prepared_test_df is not None):
            print(f"[{self.orchestrator_name}] [PREPARE-CACHE] Parametri identici al round "
                  f"precedente (dataset_type={dataset_type}, seed={base_seed}) - riuso "
                  f"train_df/test_df gia' pronti, salto l'intera pipeline ETL pesante.")
            train_df = self._cached_prepared_train_df
            test_df = self._cached_prepared_test_df
      
            self._save_prepared_data(train_df, test_df, job_id, base_seed, num_shards,
                                      tree_type, target_col, prepared_key, t0,
                                      from_cache=True)
            return

        splitter = src.shared.utilities.datasplitter.StratifiedDataSplitter(target_column=target_col, test_size=TEST_SIZE, random_state=base_seed)

        print(f"\n[{self.orchestrator_name}] Avvio ETL. Tipo: {dataset_type}")

        if dataset_type == "synthetic":
            loader = SyntheticDataLoader(task="regression" if tree_type == "regressor" else "classification", target_column=target_col)
            df_full = loader.load()

            if tree_type == "regressor":
                train_df, test_df = train_test_split(df_full, test_size=TEST_SIZE, random_state=base_seed)
            else:
                train_df, test_df = splitter.split(df_full)
            del df_full
            gc.collect()
        else:
            if not dataset_path: 
                raise ValueError("dataset_path mancante.")
            print(f"[DEBUG] dataset_path ricevuto = {repr(dataset_path)}")
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
            # Allineato a run_baseline.py: il 15% del train viene ritagliato e scartato 
            # per garantire che l'undersampling successivo lavori sullo stesso identico 
            # volume di dati della baseline. Qui non viene effettuata alcuna calibrazione 
            # della soglia.
            if tree_type == "classifier":
                print(f"\n[{self.orchestrator_name}] === SPLIT VALIDATION SET "
                      f"({VALIDATION_SIZE_FOR_THRESHOLD*100:.0f}% del train, per allineamento volume) ===")
                validation_splitter = src.shared.utilities.datasplitter.StratifiedDataSplitter(
                    target_column=target_col, test_size=VALIDATION_SIZE_FOR_THRESHOLD, random_state=base_seed
                )
                train_df, _ = validation_splitter.split(train_df)

           # ─── FASE 5: UNDER-SAMPLING DELLA CLASSE MAGGIORITARIA (solo train) ───
            # Bilancia il train set (1:1) per allinearlo a run_baseline.py, garantendo 
            # la confrontabilità delle metriche. Il test set resta INTATTO (mai 
            # sotto-campionato) per valutare le prestazioni sulla distribuzione reale.
            if tree_type == "classifier":
                print(f"\n[{self.orchestrator_name}] === UNDER-SAMPLING CLASSE MAGGIORITARIA (solo train) ===")
                train_df = undersample_majority_class(
                    train_df, target_column=target_col,
                    majority_class=0, minority_class=1,
                    ratio=UNDERSAMPLING_RATIO, random_state=base_seed,
                )

        # --- FEATURE SELECTION (Solo Dataset Reale) ---
        if dataset_type == "real":
        # Il tuning e la feature selection sono compiti esclusivi della baseline 
        # (run_baseline.py). Il percorso distribuito si limita a consumare l'output 
        # già calcolato via BaseOrchestrator.read_selected_features_from_config. 
        # Questo evita lavoro duplicato (permutation importance) e garantisce che 
        # cluster e baseline poggino sull'identico set di feature.
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

        self._save_prepared_data(train_df, test_df, job_id, base_seed, num_shards,
                                  tree_type, target_col, prepared_key, t0, from_cache=False)

    def _save_prepared_data(self, train_df, test_df, job_id: str, base_seed: int,
                             num_shards, tree_type: str, target_col: str,
                             prepared_key: tuple, t0: float, from_cache: bool):
        """Salva train_df/test_df (via DAO, con sharding opzionale) e, se
        questa e' una computazione fresca (from_cache=False), aggiorna la
        cache dell'intermedio per i round successivi con parametri identici.
        """
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
            # Seed fisso per garantire riproducibilità in entrambi i rami. 
            # Si utilizza np.array_split perché copre l'intero array senza perdere 
            # o duplicare righe, gestendo automaticamente anche i resti non divisibili 
            # esattamente (es. 800000/3).
                print(f"[{self.orchestrator_name}] [SHARDING] Partizionamento train_df "
                      f"({train_df.shape[0]} righe) in {num_shards} shard...")
                rng = np.random.RandomState(base_seed)

                if tree_type == "classifier":
                # SPLIT STRATIFICATO: shuffle e split eseguiti separatamente per classe e poi 
                # distribuiti proporzionalmente tra gli shard. 
                # Garantisce che ogni shard riceva l'esatta proporzione di ciascuna classe 
                # presente nel train_df (già ribilanciato 1:1), senza affidarsi alla sola 
                # probabilità di uno shuffle globale.
                # Il regressore (ramo else sotto) non avendo classi ricade su uno shuffle puro.
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

            # Scrittura in PARALLELO (N upload concorrenti invece di uno sequenziale).
            # A parità di volume totale di byte, l'uso del multithreading abbatte i tempi di I/O.
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
                      f"{self.last_etl_seconds:.2f}s{' [da cache]' if from_cache else ''}")
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
                print(f"[DEBUG TIMING] _prepare_data completato in {self.last_etl_seconds:.2f}s"
                      f"{' [da cache]' if from_cache else ''}")
                print(f"[{self.orchestrator_name}] [OK] Dataset di Train e Test archiviati correttamente.")

                # CACHE EFS: Scrittura best-effort attivata SOLO se EFS_MOUNT_PATH è impostata 
                # (vedi orchestrator_ec2.tf). Se fallisce o manca, il job prosegue in sicurezza 
                # usando S3 (salvataggio canonico). È un'ottimizzazione per la lettura dai worker 
                # (vedi dataset_dao.py).
                #
                # - Solo train_df: è l'unico file scaricato da più worker. test_df viene 
                #   letto una sola volta dall'orchestratore in inferenza (nessuna ridondanza).
                # - Solo modalità 'shared': in 'sharded' ogni worker legge una fetta diversa, 
                #   rendendo inutile una cache condivisa.
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

            # CACHE DELL'INTERMEDIO: aggiornata solo in caso di computazione fresca 
            # (se from_cache=True, gli oggetti sono già in cache e riassegnarli è inutile).
            # Mantiene train_df e test_df vivi in memoria sull'orchestratore per riutilizzarli 
            # in round futuri con parametri identici. Il costo in RAM (un'unica copia extra) 
            # è ampiamente giustificato dal risparmio di tempo sull'intera pipeline ETL.
            if not from_cache:
                self._cached_prepared_key = prepared_key
                self._cached_prepared_train_df = train_df
                self._cached_prepared_test_df = test_df
            gc.collect()
        except Exception as e:
            raise IOError(f"[{self.orchestrator_name}] Errore critico nel salvataggio dei dataset tramite DAO: {e}")
        self.current_job_id = job_id
        self.test_data_path = test_data_path

    def _execute_training_step(self, payload: dict, start_alberi: int, target_alberi: int, seed: int) -> int:

        #Esegue lo step di addestramento distribuito centralizzato.
        #Restituisce il numero REALE di alberi totali validati e salvati con successo.
        
        expected_job_id = payload.get("job_id", "unknown_job")
        # Estratto anticipatamente perché il tree_type è necessario PRIMA dell'ETL 
        # per determinare la strategia di sharding (fisso vs dinamico).
        hp = payload.get("hyperparameters", {})
        tree_type = hp.get("tree_type", "classifier")

        # SHARDING: gestito tramite variabile d'ambiente CENTRALIZED_DATASET_MODE (default 'shared').
        # In modalità 'sharded' è strettamente necessario scoprire i worker attivi PRIMA 
        # dell'ETL per poter determinare in quante fette partizionare il dataset.
        dataset_mode = os.environ.get("CENTRALIZED_DATASET_MODE", "shared").strip().lower()
        sharded_mode = dataset_mode == "sharded"
        early_num_workers = None
        effective_num_shards = None
        if sharded_mode:
            print(f"[{self.orchestrator_name}] [SHARDING] Modalita' 'sharded' attiva - "
                  f"scopro i worker PRIMA dell'ETL per sapere in quante fette partizionare.")
            while True:
                early_workers = ServiceRegistry.get_available_workers(self.environment)
                if early_workers:
                    early_num_workers = len(early_workers)
                    print(f"[{self.orchestrator_name}] [SHARDING] {early_num_workers} worker rilevati.")
                    break
                print(f"[{self.orchestrator_name}] [SHARDING] Nessun worker disponibile per la scoperta "
                      f"anticipata. In attesa...")
                time.sleep(10)

            # NUMERO DI SHARD:
            # - Classificatore (reale): fisso (REAL_FIXED_SHARDS). Permette il riuso dei 
            #   file generati tra round con un numero diverso di worker.
            # - Regressore (sintetico): dinamico (pari al numero di worker). Poiché l'ETL 
            #   sintetico è quasi istantaneo (< 1s), uno schema fisso aggiungerebbe solo 
            #   complessità a scapito di una partizione statistica più naturale.
            if tree_type == "classifier":
                effective_num_shards = REAL_FIXED_SHARDS
                print(f"[{self.orchestrator_name}] [SHARDING] Classificatore/reale -> "
                      f"{REAL_FIXED_SHARDS} shard FISSI (indipendenti dal numero di worker "
                      f"di questo giro), per permettere il riuso tra round diversi.")
            else:
                effective_num_shards = early_num_workers
                print(f"[{self.orchestrator_name}] [SHARDING] Regressore/sintetico -> "
                      f"{early_num_workers} shard (dinamico, pari al numero di worker).")

        # 1. Preparazione dei dati (se non ancora pronti e non presenti su disco)
        if sharded_mode:
           # SHORT-CIRCUIT SU STORAGE: verifica la presenza degli shard direttamente su 
            # disco/S3 (stessa convenzione di _prepare_data) per evitare un re-sharding 
            # completo. Copre due scenari:
            # 1) Round successivi dello stesso job.
            # 2) Failover dell'orchestratore: il nuovo standby è un processo pulito 
            #    (self.train_data_shards = None al boot), ma rileva e riusa gli shard 
            #    già generati e salvati su S3 dal leader caduto.
            if self.environment == "aws":
                expected_shards = [
                    f"s3://{BUCKET_NAME}/distributed_trains/shared_train_{expected_job_id}_shard_{i}.csv"
                    for i in range(effective_num_shards)
                ]
                expected_test_sharded = f"s3://{BUCKET_NAME}/distributed_tests/shared_test_{expected_job_id}.csv"
            else:
                expected_shards = [
                    f"./.local_storage/shared_train_{expected_job_id}_shard_{i}.csv"
                    for i in range(effective_num_shards)
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
                print(f"[{self.orchestrator_name}] [SHARDING] [SHORT-CIRCUIT] {effective_num_shards} shard "
                      f"già presenti su storage per questo job (round successivo, o ripresa dopo un "
                      f"failover dell'orchestratore) - nessuna rigenerazione.")
                self.train_data_shards = expected_shards
                self.train_data_path = None
                self.test_data_path = expected_test_sharded
                self.current_job_id = expected_job_id
                self.last_etl_seconds = 0.0
            else:
                self._prepare_data(payload, seed, num_shards=effective_num_shards)
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
            # Previene conflitti di stato: azzera esplicitamente self.train_data_shards 
            # nel ramo 'shared'. Senza questo reset, l'orchestratore manterrebbe in 
            # memoria i path obsoleti di un eventuale job precedente eseguito in 
            # modalità 'sharded'.
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

        # GESTIONE DELLA MEMORIA (Prevenzione OOM): 
        # L'estrazione dei metadati (n_features_in_, classes_) viene eseguita 
        # in modo incrementale man mano che i batch vengono confermati su disco, 
        # evitando di mantenere l'intera lista 'all_trained_trees' in memoria fino alla fine 
        # del round. 
        # In caso di resume da checkpoint fisico (failover), i metadati vengono 
        # estratti immediatamente e gli oggetti alberiformi vengono rilasciati 
        # per liberare RAM.
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
        # Fallback ai parametri di default nativi di RandomForest (Classifier/Regressor) 
        # nel caso in cui non siano esplicitamente definiti nel manifesto.
        max_features = hp.get("max_features", "sqrt" if tree_type == "classifier" else 1 / 3)
        min_samples_split = hp.get("min_samples_split", 2)
        # class_weight ha senso solo in classificazione: il worker lo ignora comunque
        # per i regressori, ma evitiamo di forzarlo se il payload non lo prevede.
        class_weight = hp.get("class_weight", None)
        criterion = hp.get("criterion", None)
        # Parametri bootstrap e max_samples inoltrati esplicitamente al worker. 
        # Se impostati a None, il worker mantiene i propri valori predefiniti. 
        # Questo assicura che le impostazioni dichiarate nel manifesto della baseline 
        # vengano correttamente applicate anziché essere sovrascritte dai default locali.
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

        # Edge case: training completato ma orchestratore crashato prima del consolidamento finale.
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

            # Lock dedicato alla persistenza del checkpoint, disaccoppiato da results_lock.
            # Evita che l'I/O di rete (upload su S3) blocchi la sezione critica adibita 
            # alla sola registrazione dei risultati. Questo previene colli di bottiglia 
            # di serializzazione all'aumentare dei worker e garantisce misurazioni 
            # accurate dello strong scaling.
            checkpoint_lock = threading.Lock()
            # Contatore monotono dell'ultimo snapshot effettivamente persistito.
            # Assicura due garanzie critiche:
            # 1) Previene race condition: poiché la scrittura asincrona fuori dal lock 
            #    può completarsi fuori ordine, il contatore evita che uno snapshot obsoleto 
            #    sovrascriva uno più recente, impedendo regressioni nello stato di recovery.
            # 2) Evita scritture ridondanti: se è già stato persistito uno stato con un 
            #    numero maggiore di alberi, lo snapshot corrente viene saltato, riducendo 
            #    il I/O inutile senza compromettere la fault tolerance.
            # 
            # Nota: L'indicizzazione delle parti riparte dal numero di file già presenti 
            # su storage per evitare di sovrascrivere delta di checkpoint validi.
            last_checkpointed = {"count": start_alberi,
                                 "parts": self._count_trees_checkpoint_parts(self.current_job_id)}

            active_worker_names = list(worker_names)

            # Reset dell'evento di sincronizzazione (utilizzato anche in inferenza).
            # Consente ai test di fault injection di attendere in modo deterministico 
            # l'invio effettivo del primo task di training a un worker, evitando sleep 
            # a tempo fisso.
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
                    
                    # Il thread resta attivo finché non raccogliamo la quota di alberi globale
                    while len(all_trained_trees) < target_alberi:
                        try:
                            # Timeout breve (2 secondi) per controllare periodicamente lo stato e non restare appesi
                            task_id, start_t, end_t, chunk_seed = task_queue.get(timeout=2)
                        except queue.Empty:
                        # Se la coda è temporaneamente vuota ma il target globale di alberi non è raggiunto,
                        # i worker non devono terminare prematuramente (per gestire eventuali crash o task reinseriti).
                        # FIX (punto 6): la terminazione è consentita SOLO a training completato. Prima si
                        # usciva anche quando questo era l'ultimo worker attivo rimasto (num_worker_attivi
                        # <= 1) -- ma se in seguito un suo task fosse stato riaccodato per un fallimento,
                        # nessun thread sarebbe rimasto a consumarlo: il controllo di sicurezza esistente
                        # subito dopo (raise RuntimeError) scatta solo con active_worker_names VUOTO, non
                        # con un solo worker rimasto ma uscito prematuramente dal polling.
                            with results_lock:
                                total_attuali = len(all_trained_trees)

                            if total_attuali >= target_alberi:
                                break
                            time.sleep(1)
                            continue

                        quota_chunk = end_t - start_t
                        print(f"[{self.orchestrator_name}-Thread] Assegnazione Task {task_id} ({quota_chunk} alberi: {start_t}-{end_t}) a {w_name}")
                        self._track_task(task_id=task_id, job_id=self.current_job_id, worker_name=w_name, status="PROCESSING")
                        try:
                            self.chunk_sent_event.set()

                            # NOTA DI ARCHITETTURA RPYC: le liste Python passate come argomenti RPC 
                            # vengono trasmesse per riferimento (netref) e non per valore, generando 
                            # chiamate remote sincrone indesiderate a ogni confronto lato worker e 
                            # causando EOFError in caso di instabilità della connessione.
                            # Per evitare questo, le liste di path S3 vengono serializzate come stringhe 
                            # uniche delimitate da '|' (carattere sicuro poiché assente nei path S3) 
                            # e decodificate puntualmente lato worker in CentralizedWorker._load_data.
                            if self.train_data_shards:
                                _n_shards = len(self.train_data_shards)
                                _assigned_shards = [
                                    self.train_data_shards[j] for j in range(_n_shards)
                                    if j % num_workers == task_id % num_workers
                                ]
                                task_source_info = "|".join(_assigned_shards)
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
                                max_samples=max_samples,
                                job_id=self.current_job_id
                            )

                            # I worker persistono il blob degli alberi (che può raggiungere dimensioni notevoli) 
                            # direttamente nello storage condiviso prima di rispondere. L'orchestratore 
                            # riceve un ACK leggero e ricarica i dati dallo storage, evitando il trasporto 
                            # di payload pesanti tramite chiamate RPC sincrone (prevenendo blocchi di rete in RPyC).
                            ack = obtain(ack_raw)
                            if not isinstance(ack, dict) or not ack.get("ack"):
                                raise RuntimeError(
                                    f"Risposta inattesa dal worker {w_name} per il task {task_id}: {ack!r}"
                                )

                            # GESTIONE MEMORIA E CONSISTENZA STORAGE:
                            # 1. Caricamento incrementale dei task: per prevenire picchi critici di memoria 
                            #    (che possono superare 1GB con task massicci o alberi profondi), i task vengono 
                            #    letti e processati iterativamente parte per parte (iter_task_parts_as_tree_lists). 
                            #    Ciascuna parte viene persistita e rilasciata prima di caricare la successiva, 
                            #    mantenendo il picco di RAM costante e indipendente dalla dimensione del chunk.
                            # 2. Allineamento delle chiavi di storage: la derivazione del prefisso di storage 
                            #    gestisce uniformemente sia file singoli che liste multi-shard, garantendo 
                            #    una corrispondenza esatta tra orchestratore e worker per evitare disallineamenti 
                            #    nelle chiavi di lettura/scrittura su storage condiviso.
                            storage_key_for_reread = f"shared_train_{self.current_job_id}.csv"
                            part_iter = iter_task_parts_as_tree_lists(
                                storage_key_for_reread, chunk_seed, quota_chunk,
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
                                        # Rilevamento di task incompleto: se almeno una parte è stata ricevuta 
                                        # ma il task risulta globalmente carente rispetto al manifesto, si tratta 
                                        # di un errore effettivo (es. corruzione o cancellazione anomala su storage) 
                                        # e non di una semplice attesa di sincronizzazione.
                                        # 
                                        # NOTA SUL COMPORTAMENTO DI RETRY: in caso di eccezione durante il recupero, 
                                        # il task viene riaccodato per intero. Poiché alcune parti potrebbero essere 
                                        # state già registrate prima del fallimento, un retry completo potrebbe 
                                        # teoricamente rigenerare gli stessi alberi (rischio di doppio conteggio). 
                                        # Tale scenario presuppone una corruzione del file system/storage successiva 
                                        # alla scrittura del manifesto ed è un rischio accettato in questa fase; 
                                        # un hardening completo richiederebbe il tracciamento granulare delle parti 
                                        # già elaborate all'interno della coda dei task.
                                            raise
                                        raise RuntimeError(
                                            f"Worker {w_name}: task {task_id} confermato (ack) ma il blob "
                                            f"non è stato trovato nello storage condiviso."
                                        )
                                received_any_part = True

                               # SEZIONE CRITICA MINIMA: la sezione protetta si limita all'aggiornamento 
                                # della lista condivisa e alla creazione di uno snapshot immutabile. 
                                # Le operazioni di I/O (upload su S3 e scrittura su DynamoDB) sono eseguite 
                                # all'esterno del lock per evitare che i worker si blocchino a vicenda 
                                # in attesa della rete, prevenendo colli di bottiglia nella registrazione 
                                # dei risultati.
                                with results_lock:
                                    all_trained_trees.extend(part_trees)
                                    current_total = len(all_trained_trees)
                                    # list(...) crea una copia difensiva: garantisce l'immutabilità dello snapshot 
                                    # durante la serializzazione eseguita all'esterno del lock.
                                    snapshot = list(all_trained_trees)
                                # Rilascia esplicitamente il riferimento per consentire il recupero 
                                # della memoria (Garbage Collection) dopo il salvataggio della parte corrente.
                                part_trees = None  

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

                                            # GESTIONE DELLA MEMORIA: estrazione incrementale dei metadati (n_features_in_ 
                                            # e classes_) direttamente dal batch corrente prima del suo rilascio, evitando 
                                            # l'ispezione dell'intera collezione 'all_trained_trees' (la quale può contenere 
                                            # placeholder nulli derivanti da batch precedentemente liberati).
                                            newly_persisted = snapshot[prev_checkpointed:current_total]
                                            if running_n_features[0] is None and newly_persisted:
                                                running_n_features[0] = int(newly_persisted[0].n_features_in_)
                                            trees_with_classes_batch = [t for t in newly_persisted if hasattr(t, "classes_")]
                                            if trees_with_classes_batch:
                                                batch_classes = np.unique(np.concatenate(
                                                    [np.asarray(t.classes_) for t in trees_with_classes_batch]))
                                                running_classes.update(batch_classes.tolist())

                                            # GESTIONE DELLA MEMORIA: gli alberi già persistiti su disco vengono sostituiti 
                                            # da placeholder 'None' nella collezione in memoria. 
                                            # Poiché il metodo _persist_trees_delta utilizza uno slicing posizionale 
                                            # (snapshot[already_persisted:]) per i delta successivi, preservare la lunghezza 
                                            # originaria della lista tramite i valori nulli garantisce la correttezza degli 
                                            # indici, prevenendo al contempo un consumo eccessivo di RAM su dataset estesi 
                                            # con alberi non potati.
                                            with results_lock:
                                                for _idx in range(prev_checkpointed, current_total):
                                                    all_trained_trees[_idx] = None
                                            snapshot = None

                                            # GESTIONE DELLA MEMORIA (Rilascio dell'RSS verso il sistema operativo):
                                            # gc.collect() da solo non è sufficiente, poiché glibc tende a trattenere 
                                            # la memoria liberata nelle proprie riserve interne, lasciando l'RSS del 
                                            # container (visibile via Docker/cgroup) artificialmente alto.
                                            # La chiamata a malloc_trim(0) forza l'allocatore C a restituire i blocchi 
                                            # liberi direttamente al sistema operativo. L'operazione è sicura e idempotente 
                                            # (agisce come no-op se non ci sono blocchi da rilasciare), risultando 
                                            # ideale per l'invocazione sistematica dopo il completamento di ogni batch.
                                            gc.collect()
                                            try:
                                                ctypes.CDLL("libc.so.6").malloc_trim(0)
                                            except Exception:
                                                pass  # piattaforme non-glibc (es. macOS)

                                            # La cache di istanza referenzia direttamente la lista alleggerita 
                                            # (evitando duplicazioni di memoria): la sincronizzazione intra-processo 
                                            # (STATE-SYNC) mantiene la coerenza del conteggio senza preservare in RAM 
                                            # i payload degli alberi già persistiti.
                                            self._trees_cache[self.current_job_id] = all_trained_trees
                                            print(f"   [RPC <- {w_name}] [CHECKPOINT FS OK] Parte di Task {task_id} archiviata. Progressivo in RAM/Storage: {current_total} alberi.")

                                            # FIX (punto 5): l'heartbeat va scritto SOLO se il
                                            # checkpoint fisico e' andato a buon fine -- spostato
                                            # DENTRO il try, altrimenti DynamoDB dichiarerebbe
                                            # alberi_addestrati=current_total anche quando
                                            # _persist_trees_delta e' appena fallito (vedi except
                                            # sotto), cioe' piu' alberi di quelli davvero
                                            # recuperabili dal checkpoint in caso di failover.
                                            # last_checkpointed["count"] (appena aggiornato sopra)
                                            # e' la fonte di verita' di quanto e' STATO persistito,
                                            # non di quanto si STAVA per persistere.
                                            if hasattr(self, 'state_manager') and self.state_manager:
                                                try:
                                                    self.state_manager.update_request_status(
                                                        job_id=self.current_job_id,
                                                        status="PROCESSING",
                                                        orchestrator_id=self.orchestrator_name,
                                                        retries=payload.get("retries", 0),
                                                        base_random_state=seed,
                                                        alberi_addestrati=last_checkpointed["count"]
                                                    )
                                                except Exception as e_db:
                                                    print(f"   [ERRORE] Impossibile inviare l'heartbeat di stato a DynamoDB: {e_db}")
                                        except Exception as e_fs:
                                            # last_checkpointed NON avanza: un writer successivo
                                            # deve poter riprovare a persistere lo stato. L'heartbeat
                                            # DynamoDB viene SALTATO in questo caso (vedi sopra):
                                            # current_total non e' mai stato davvero persistito su
                                            # storage, quindi non va dichiarato come tale.
                                            print(f"   [ERRORE FILE SYSTEM] Impossibile scrivere gli alberi parziali su file: {e_fs}")
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
            # GESTIONE DELLA MEMORIA (Staggered Startup): l'avvio dei thread worker 
            # è deliberatamente scaglionato tramite una breve pausa.
            # Questo previene richieste simultanee di caricamento del dataset condiviso 
            # da parte dei worker, evitando picchi di memoria concorrenti su tutti i 
            # container (che potrebbero innescare OOM sincronizzati). L'overhead 
            # temporale è trascurabile e impatta unicamente il primo task; i sotto-task 
            # successivi utilizzano la cache locale del worker (self._cached_X/_cached_y) 
            # senza innescare ulteriori ricaricamenti.
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


                print(f"[DEBUG TIMING] Riepilogo _execute_training_step -> "
                      f"ETL {self.last_etl_seconds:.2f}s | costruzione alberi "
                      f"{self.last_dispatch_seconds:.2f}s | aggregazione "
                      f"{self.last_aggregation_seconds:.2f}s | OOB {self.last_oob_seconds:.2f}s")

                # Restituiamo la dimensione REALE degli alberi salvati
                return n_trees_for_report
                
            except Exception as e:
                print(f"   [ERRORE AGGREGAZIONE] Fallimento durante la costruzione del manifesto: {e}")
                traceback.print_exc()
                return len(all_trained_trees)

        print(f"   [{self.orchestrator_name}] Nessun albero collezionato.")
        # Ritorna 0 se non è stato possibile generare o caricare nulla 
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
        
        # FAULT TOLERANCE (fix): un chunk di alberi mancante non deve abortire
        # l'intera inferenza. Con l'aggregazione a soft voting usata qui (media
        # delle predizioni per-albero), una foresta con meno alberi resta un
        # ensemble valido, solo con meno alberi che votano — esattamente
        # l'"esito parziale" richiesto dalla traccia. PRIMA di questo fix, il
        # blocco sollevava un'eccezione non appena 'failed_tasks' o
        # 'task_queue' non erano vuoti, uscendo dalla funzione PRIMA di
        # raggiungere l'aggregazione e il return con "status": "PARTIAL" più
        # sotto: quel ramo restava di fatto irraggiungibile in ogni caso di
        # guasto. Ci fermiamo ora solo se non è stato raccolto NESSUN chunk:
        # lì non c'è foresta da aggregare, quindi resta un fallimento vero.
        if not predictions_chunks:
            with self.connessioni_lock:
                for conn in self.connessioni_attive:
                    try: conn.close()
                    except Exception: pass
            raise RuntimeError("Inferenza fallita: nessun worker ha completato un chunk (0 alberi disponibili).")

        if failed_tasks or not task_queue.empty():
            orphaned = task_queue.qsize()
            print(f"[{self.orchestrator_name}] [WARN] Inferenza PARZIALE: {len(failed_tasks)} chunk falliti "
                  f"definitivamente, {orphaned} mai tentati. Procedo con i {len(predictions_chunks)} chunk "
                  f"raccolti (foresta ridotta, ma ensemble valido).")

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
        prediction_sample = self._sample_predictions(y_test, final_predictions, tree_type)
        self._save_metrics(job_id, "inference", {
            "job_id": job_id, "mode": "centralized", "phase": "inference",
            "tree_type": tree_type, "testing_set_size": X_test.shape[0],
            "timings": {"total_inference_time": total_inference_time, "rpc_inference_time": rpc_inference_time},
            "metrics": metrics,
            "prediction_sample": prediction_sample,
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


        return {
            "status": "SUCCESS" if (not failed_tasks and task_queue.empty()) else "PARTIAL",
            "testing_set_size": int(X_test.shape[0]),
            "total_inference_time": total_inference_time,
            "rpc_inference_time": rpc_inference_time,
            "metrics": metrics,
            "prediction_sample": prediction_sample,
        }

    def _save_checkpoint(self, job_id: str, current_alberi: int, retries: int, base_random_state: int):
        """
        Estende il checkpoint della classe base (metadati logici su DynamoDB).

        FIX (punto 2): rimosso il parametro 'alberi_reali' e il ramo che lo
        gestiva -- era codice morto (BaseOrchestrator._save_checkpoint non
        accetta 'alberi_reali' e nessun chiamante lo passava mai; il
        salvataggio fisico degli alberi passa SEMPRE da _persist_trees_delta
        nel dispatch incrementale, vedi _execute_training_step). Tenerlo in
        vita rischiava di essere riattivato per errore: farlo era già stata
        la causa di un OOM osservato in produzione quando lo stesso pattern
        era stato usato lato federato (tripla serializzazione dell'intera
        foresta), poi rimosso da lì per lo stesso motivo (vedi federated.py).
        """
        super()._save_checkpoint(job_id, current_alberi, retries, base_random_state)

    def _clean_checkpoint(self, job_id: str):

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