from abc import ABC, abstractmethod
import json
import os
import re
import shutil
import boto3
from botocore.exceptions import ClientError

# Cartella dei manifesti prodotti da run_baseline(): sono la fonte di verità
# condivisa fra baseline locale, client e cluster (vedi
# main.py::load_hyperparameters_from_config e SyntheticDataLoader, che da qui
# ricava la ricetta del dataset).
BASELINE_MANIFEST_DIR = "outputs_baseline"


class BaseTestScenario(ABC):
    """Classe base astratta per tutti gli scenari di test."""

    # Cache A LIVELLO DI CLASSE (condivisa tra TUTTE le istanze di scenario
    # create nello stesso processo dell'engine, es. quando si sceglie 'all').
    # Chiave: firma dei parametri che determinano il CONTENUTO del dataset
    # preprocessato (path, tipo, tipo di albero, seed). Valore: (train_path,
    # test_path) del primo scenario che li ha prodotti.
    #
    # NOTA: questo NON tocca in alcun modo il codice dell'orchestratore. Sfrutta
    # solo il fatto che l'orchestratore già cerca da sé, prima di rifare l'ETL,
    # un file 'shared_train_{job_id}.csv'/'shared_test_{job_id}.csv' — se quel
    # file esiste già quando lo scenario chiama _execute_training_step, il suo
    # normale SHORT-CIRCUIT ETL interno lo trova e salta il preprocessing da
    # solo. Qui ci limitiamo a "precopiare" un dataset già pronto (prodotto da
    # uno scenario precedente con parametri IDENTICI) nel path che
    # l'orchestratore si aspetterà per il prossimo job_id.
    _dataset_cache: dict = {}

    def __init__(self, config: dict, orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        # Cache di istanza: _resolve_hyperparameters() viene chiamata più volte
        # nello stesso scenario (per il payload e per il numero di alberi) e
        # senza cache ristamperebbe la stessa diagnostica a ogni invocazione.
        self._resolved_hp = None

    @abstractmethod
    def run(self) -> dict:
        """Esegue lo scenario e restituisce un dizionario con i risultati/metriche."""
        pass

    # ------------------------------------------------------------------ #
    # Iperparametri: fonte di verità condivisa con la baseline            #
    # ------------------------------------------------------------------ #

    def _local_hyperparameters(self) -> dict:
        """
        Blocco iperparametri definito in test_config.json, scelto in base a
        'selected_task'. Usato come fallback e quando l'override esplicito è
        attivo (vedi _resolve_hyperparameters).
        """
        if self.config.get("selected_task") == "classifier":
            return dict(self.config.get("hyperparameters_class", {}) or {})
        return dict(self.config.get("hyperparameters_regre", {}) or {})

    def _resolve_hyperparameters(self) -> dict:
        """
        Iperparametri del job da eseguire.

        FONTE DI VERITÀ: il manifesto della baseline
        ('outputs_baseline/config_synthetic.json' oppure 'config_real.json'),
        lo stesso file già letto dal client
        (main.py::load_hyperparameters_from_config) e — per la ricetta del
        dataset — da SyntheticDataLoader.

        PERCHÉ NON test_config.json: la baseline locale addestra con gli
        iperparametri del manifesto. Se gli scenari usassero i propri
        ('hyperparameters_class'/'hyperparameters_regre'), il confronto
        baseline-cluster richiesto dalla traccia metterebbe a paragone due
        modelli DIVERSI — ad esempio 40 alberi a profondità illimitata contro
        100 alberi a profondità 15. È una differenza che non produce alcun
        errore e si nota solo confrontando i log a mano.

        OVERRIDE ESPLICITO: impostando "use_local_hyperparameters": true in
        test_config.json si tornano a usare i valori locali. Serve per gli
        esperimenti deliberatamente diversi dalla baseline (es. più alberi per
        dare al cluster abbastanza lavoro da ammortizzare l'overhead nei test
        di scalabilità), ma va dichiarato: così la scelta è visibile nel
        config e nei log, invece di essere un disallineamento silenzioso.

        FALLBACK: se il manifesto manca o è illeggibile si usano comunque i
        valori locali, così il comportamento resta quello storico su una
        macchina dove run_baseline() non è mai stata eseguita.
        """
        if self._resolved_hp is not None:
            return dict(self._resolved_hp)

        local_hp = self._local_hyperparameters()

        if self.config.get("use_local_hyperparameters"):
            print("[TEST CONFIG] Override attivo ('use_local_hyperparameters'): uso gli "
                  "iperparametri di test_config.json. ATTENZIONE: tempi e metriche di questo "
                  "run NON sono confrontabili con la baseline locale, che addestra con quelli "
                  "del manifesto.")
            self._resolved_hp = local_hp
            return dict(local_hp)

        dataset_type = self.config.get("dataset_type", "real")
        manifest_name = "config_synthetic.json" if dataset_type == "synthetic" else "config_real.json"
        manifest_path = os.path.join(BASELINE_MANIFEST_DIR, manifest_name)

        if not os.path.exists(manifest_path):
            print(f"[TEST CONFIG] [WARN] Manifesto '{manifest_path}' non trovato: uso gli "
                  f"iperparametri di test_config.json. Il confronto con la baseline non sarà "
                  f"valido finché non si esegue run_baseline() per questo dataset.")
            self._resolved_hp = local_hp
            return dict(local_hp)

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_hp = (json.load(f) or {}).get("hyperparameters", {}) or {}
        except (json.JSONDecodeError, OSError) as e:
            print(f"[TEST CONFIG] [WARN] Manifesto '{manifest_path}' illeggibile ({e}): uso gli "
                  f"iperparametri di test_config.json.")
            self._resolved_hp = local_hp
            return dict(local_hp)

        if not manifest_hp:
            print(f"[TEST CONFIG] [WARN] Sezione 'hyperparameters' assente o vuota in "
                  f"'{manifest_path}': uso gli iperparametri di test_config.json.")
            self._resolved_hp = local_hp
            return dict(local_hp)

        # Coerenza del task: un manifesto generato per un tipo di albero diverso
        # da quello selezionato nei test renderebbe il confronto privo di senso
        # (es. baseline di classificazione contro cluster di regressione).
        # Il manifesto sintetico è UNO SOLO per entrambi i task, quindi la
        # situazione si verifica banalmente lanciando la baseline per un task e
        # i test per l'altro: meglio gridarlo che lasciarlo passare.
        manifest_tree_type = manifest_hp.get("tree_type")
        selected_task = self.config.get("selected_task")
        if manifest_tree_type and selected_task and manifest_tree_type != selected_task:
            print(f"[TEST CONFIG] [ATTENZIONE] Il manifesto '{manifest_path}' è stato generato per "
                  f"tree_type='{manifest_tree_type}', ma test_config.json dichiara "
                  f"selected_task='{selected_task}'. Rilancia run_baseline() per il task corretto: "
                  f"altrimenti baseline e cluster addestrano modelli di tipo diverso.")

        print(f"[TEST CONFIG] Iperparametri letti dal manifesto della baseline '{manifest_path}' -> "
              f"n_estimators={manifest_hp.get('n_estimators')}, max_depth={manifest_hp.get('max_depth')}, "
              f"max_features={manifest_hp.get('max_features')}, criterion={manifest_hp.get('criterion')}, "
              f"bootstrap={manifest_hp.get('bootstrap')}, max_samples={manifest_hp.get('max_samples')}")

        self._resolved_hp = dict(manifest_hp)
        return dict(self._resolved_hp)

    def _resolve_federated_partitioning(self) -> dict:
        """
        Strategia/alpha di partizionamento e allocazione alberi dichiarati nel
        manifesto della baseline ('federated_partitioning', vedi
        run_baseline.py e main.py::load_federated_partitioning) — STESSA fonte
        di verità già usata da _resolve_hyperparameters.

        Irrilevante per il centralizzato (non partiziona mai i dati) e per il
        dataset sintetico (nessun manifesto di partizionamento). In quei casi,
        e in generale se il manifesto manca/è illeggibile, ritorna il default
        sicuro {"strategy": "iid", "alpha": None, "tree_allocation": "proportional"}.

        Senza questo, i payload costruiti direttamente dagli scenari (che
        bypassano TrainingRequest/InferenceRequest e quindi main.py) non
        porterebbero mai questi campi: federated.py userebbe comunque i propri
        fallback sicuri e NON andrebbe in errore, ma le metriche/i log
        etichetterebbero sempre "iid"/"proportional" anche quando gli shard
        sul disco sono stati provisionati con Dirichlet/equal — un
        disallineamento silenzioso tra ciò che è stato davvero testato e ciò
        che viene riportato.
        """
        default = {"strategy": "iid", "alpha": None, "tree_allocation": "proportional"}

        dataset_type = self.config.get("dataset_type", "real")
        if dataset_type != "real":
            # Il dataset sintetico non ha shard fisici di dimensione variabile
            # (ogni worker genera la stessa n_samples): 'proportional' finirebbe
            # comunque per ricadere su un'allocazione equa dopo una probe RPC a
            # vuoto (vedi WARN "Nessuna dimensione di shard rilevata" in
            # federated.py._allocate_tree_quotas). Dichiariamo 'equal' esplicitamente
            # così quella probe inutile viene saltata del tutto (vedi
            # tree_allocation_strategy == "equal" in
            # FederatedOrchestrator._execute_training_step).
            return {"strategy": "iid", "alpha": None, "tree_allocation": "equal"}

        manifest_path = os.path.join(BASELINE_MANIFEST_DIR, "config_real.json")
        if not os.path.exists(manifest_path):
            return default
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f) or {}
        except (json.JSONDecodeError, OSError):
            return default

        partitioning = manifest.get("federated_partitioning") or {}
        return {
            "strategy": partitioning.get("strategy", "iid"),
            "alpha": partitioning.get("alpha"),
            "tree_allocation": partitioning.get("tree_allocation", "proportional"),
        }

    def _augment_payload_with_partitioning(self, payload: dict) -> dict:
        """
        Da chiamare all'interno di ogni _build_payload() PRIMA di ritornare il
        dizionario, per i soli scenari in modalità federata: aggiunge i tre
        campi piatti che federated.py legge davvero (partition_strategy,
        partition_alpha, tree_allocation_strategy — vedi
        FederatedOrchestrator._execute_training_step/_execute_inference_step),
        speculare a come main.py li ricava con load_federated_partitioning()
        prima di costruire una TrainingRequest/InferenceRequest.
        """
        info = self._resolve_federated_partitioning()
        payload["partition_strategy"] = info["strategy"]
        payload["partition_alpha"] = info["alpha"]
        payload["tree_allocation_strategy"] = info["tree_allocation"]
        return payload

    def _pick_worker_index_with_real_work(self, environment: str, default_index: int = 1) -> int:
        """
        Determina, per la modalità FEDERATA, quale worker conviene colpire per
        simulare un crash realistico: interroga ogni worker registrato per la
        dimensione del proprio shard locale (RPC exposed_get_local_shard_size,
        vedi FederatedWorker) e ritorna l'indice (1-based, stesso schema di
        naming Worker-Locale-NN / worker-service-N) di quello con lo shard
        PIÙ GRANDE.

        Perché non "sempre worker 1" (comportamento storico): con
        l'allocazione proporzionale (FederatedOrchestrator._allocate_tree_quotas,
        strategy="proportional", il default), un worker con uno shard piccolo
        o vuoto — tipico con partizionamento Dirichlet ad alpha basso — può
        ricevere pochissimi alberi o addirittura zero in un dato round.
        Ucciderlo non eserciterebbe alcuna redistribuzione di lavoro reale: il
        test dichiarerebbe SUCCESS senza aver davvero testato il path di
        fault-tolerance, proprio negli scenari non-IID più interessanti.

        Euristica: lo shard più grande è, sotto allocazione proporzionale,
        quasi certamente quello con la quota di alberi più alta — la scelta
        più sicura senza dover intercettare lo stato interno
        dell'orchestratore (che vive in un altro processo/task).

        Ritorna default_index (comportamento storico) se non è possibile
        determinarlo: nessun worker raggiungibile, RPC fallite, o
        ServiceRegistry/rpyc non disponibili in questo contesto.
        """
        try:
            import rpyc
            from src.shared.binding.serviceregistry import ServiceRegistry
        except ImportError:
            return default_index

        try:
            available_workers = ServiceRegistry.get_available_workers(environment)
        except Exception as e:
            print(f"[TEST WARN] Impossibile leggere ServiceRegistry per la selezione del worker "
                  f"da colpire ({e}): ricado sull'indice di default ({default_index}).")
            return default_index

        if not available_workers:
            return default_index

        best_index = default_index
        best_size = -1
        for worker_name, w_info in available_workers.items():
            conn = None
            try:
                conn = rpyc.connect(
                    w_info["host"], w_info["port"],
                    config={"allow_pickle": True, "sync_request_timeout": 15}
                )
                size = int(conn.root.exposed_get_local_shard_size())
            except Exception:
                continue
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

            match = re.search(r"(\d+)\s*$", worker_name)
            if not match:
                continue
            idx = int(match.group(1))

            if size > best_size:
                best_size = size
                best_index = idx

        if best_size < 0:
            print(f"[TEST WARN] Nessun worker ha risposto alla probe della dimensione shard: "
                  f"ricado sull'indice di default ({default_index}).")
            return default_index

        print(f"[TEST] Worker selezionato per la simulazione del crash: indice {best_index} "
              f"(shard più grande osservato tra quelli raggiungibili: {best_size} campioni).")
        return best_index

    def _resolve_target_trees(self, default: int = 100) -> int:
        """
        Numero di alberi che lo scenario deve far costruire.

        Deriva dallo STESSO blocco restituito da _resolve_hyperparameters
        invece di essere riletto per conto proprio da test_config.json: in
        caso contrario si potrebbe chiedere al cluster di costruire 100 alberi
        mentre il payload inviato ai worker ne dichiara 40.
        """
        try:
            return int(self._resolve_hyperparameters().get("n_estimators", default))
        except (TypeError, ValueError):
            return default

    def _dataset_signature(self, payload: dict, seed: int) -> tuple:
        hp = payload.get("hyperparameters", {})
        return (
            payload.get("dataset_path", ""),
            payload.get("dataset_type", ""),
            hp.get("tree_type", "classifier"),
            seed,
        )

    def _reuse_dataset_if_available(self, payload: dict, seed: int = 123):
        """
        Da chiamare SEMPRE subito prima di orchestrator._execute_training_step().

        Se uno scenario precedente in questa sessione ha già preparato un
        dataset con parametri di ETL identici (stesso dataset_path, tipo,
        tipo di albero e seed), rende disponibili quei file già pronti nel
        path che l'orchestratore si aspetta per il job_id corrente. L'orchestratore
        troverà così il file già presente e salterà da solo l'intera fase di
        preprocessing (che può costare 130-260s), grazie al suo meccanismo di
        SHORT-CIRCUIT ETL già esistente — nessuna modifica al suo codice.

        Se invece è la prima volta che si vede questa combinazione di
        parametri, registra semplicemente il path che AVRÀ questo job_id una
        volta completata la sua ETL, così il PROSSIMO scenario con parametri
        identici potrà riusarlo.

        Due implementazioni parallele in base allo storage dell'orchestratore:
        - 'local': copia i file su filesystem locale (bind mount).
        - 'aws': copia gli oggetti S3 lato server con CopyObject — i byte
          (anche 150-260 MB) non transitano mai per il container del test
          engine, il trasferimento avviene interamente dentro S3 in pochi
          secondi. Stessa identica cache a livello di firma dei parametri.
        """
        environment = getattr(self.orchestrator, "environment", "local")
        job_id = payload.get("job_id")
        if not job_id:
            return

        signature = self._dataset_signature(payload, seed)

        if environment == "aws":
            self._reuse_dataset_aws(signature, job_id, seed)
            return

        if environment != "local":
            # Ambiente non riconosciuto: nessuna scorciatoia nota, l'ETL
            # verrà comunque eseguita normalmente da _execute_training_step.
            return

        expected_train = f"./.local_storage/shared_train_{job_id}.csv"
        expected_test = f"./.local_storage/shared_test_{job_id}.csv"

        cached = BaseTestScenario._dataset_cache.get(signature)
        if cached:
            ref_train, ref_test = cached
            if os.path.exists(ref_train) and os.path.exists(ref_test):
                try:
                    os.makedirs("./.local_storage", exist_ok=True)
                    shutil.copyfile(ref_train, expected_train)
                    shutil.copyfile(ref_test, expected_test)
                    print(f"[TEST CACHE] Dataset riusato da uno scenario precedente con "
                          f"parametri ETL identici (seed={seed}): ETL saltata per questo job.")
                except Exception as e:
                    print(f"[TEST CACHE WARN] Copia del dataset in cache fallita, "
                          f"procedo con ETL normale: {e}")
                return

        # Prima apparizione di questa combinazione di parametri: registriamo
        # SOLO il path futuro. Il file non esiste ancora (l'ETL vera e propria
        # avverrà dentro _execute_training_step subito dopo questa chiamata),
        # ma da qui in avanti qualunque altro scenario con la stessa firma lo
        # troverà già pronto sul disco al momento in cui gli servirà.
        BaseTestScenario._dataset_cache[signature] = (expected_train, expected_test)

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple:
        """Scompone un URI 's3://bucket/key/...' in (bucket, key)."""
        without_scheme = uri[len("s3://"):]
        bucket, _, key = without_scheme.partition("/")
        return bucket, key

    def _reuse_dataset_aws(self, signature: tuple, job_id: str, seed: int):
        """
        Equivalente AWS di _reuse_dataset_if_available. Invece di shutil.copyfile
        su bind mount locale (che su S3 non esiste), usa S3 CopyObject: un
        comando che dice a S3 "duplica questo oggetto in un altro path",
        eseguito interamente lato server — i byte del dataset non passano mai
        per il container del test engine, a differenza di un
        download+upload. Stesso identico path atteso dall'orchestratore
        (vedi CentralizedOrchestrator._prepare_data / _execute_training_step:
        's3://{bucket}/distributed_trains/shared_train_{job_id}.csv' e
        analogo per il test set), quindi anche qui nessuna modifica al codice
        dell'orchestratore: trova l'oggetto già lì e attiva da solo il suo
        SHORT-CIRCUIT ETL.
        """
        bucket_name = os.environ.get(
            "DATASETS_BUCKET_NAME", "my-cluster-datasets-bucket-759804778194-us-east-1-an"
        )
        expected_train = f"s3://{bucket_name}/distributed_trains/shared_train_{job_id}.csv"
        expected_test = f"s3://{bucket_name}/distributed_tests/shared_test_{job_id}.csv"

        cached = BaseTestScenario._dataset_cache.get(signature)
        if cached:
            ref_train, ref_test = cached
            s3_client = boto3.client("s3")
            try:
                ref_train_bucket, ref_train_key = self._parse_s3_uri(ref_train)
                ref_test_bucket, ref_test_key = self._parse_s3_uri(ref_test)
                exp_train_bucket, exp_train_key = self._parse_s3_uri(expected_train)
                exp_test_bucket, exp_test_key = self._parse_s3_uri(expected_test)

                # Verifica esistenza degli oggetti sorgente PRIMA di copiare: se
                # lo scenario precedente che li ha registrati è fallito a metà
                # ETL (o è ancora in corso), l'oggetto potrebbe non essere mai
                # stato scritto — meglio ricadere sull'ETL normale che
                # propagare un CopyObject fallito e bloccare lo scenario.
                s3_client.head_object(Bucket=ref_train_bucket, Key=ref_train_key)
                s3_client.head_object(Bucket=ref_test_bucket, Key=ref_test_key)

                s3_client.copy_object(
                    Bucket=exp_train_bucket, Key=exp_train_key,
                    CopySource={"Bucket": ref_train_bucket, "Key": ref_train_key},
                )
                s3_client.copy_object(
                    Bucket=exp_test_bucket, Key=exp_test_key,
                    CopySource={"Bucket": ref_test_bucket, "Key": ref_test_key},
                )
                print(f"[TEST CACHE AWS] Dataset riusato da uno scenario precedente con "
                      f"parametri ETL identici (seed={seed}): copia server-side S3 "
                      f"invece di ETL completa. ETL saltata per questo job.")
            except ClientError as e:
                print(f"[TEST CACHE AWS WARN] Copia S3 del dataset in cache fallita "
                      f"({e}), procedo con ETL normale.")
            return

        # Prima apparizione di questa combinazione di parametri su AWS:
        # registriamo SOLO il path S3 futuro. L'oggetto non esiste ancora
        # (l'ETL vera e propria avverrà dentro _execute_training_step subito
        # dopo questa chiamata, con l'upload su S3 che già vediamo nel log),
        # ma da qui in avanti qualunque altro scenario con la stessa firma lo
        # troverà già pronto su S3 al momento in cui gli servirà.
        BaseTestScenario._dataset_cache[signature] = (expected_train, expected_test)

    def _mark_job_finished(self, job_id: str, alberi_addestrati: int = 0):
        """
        Da chiamare DOPO ogni invocazione diretta di _execute_training_step o
        _execute_inference_step (che, a differenza di _process_job, non finalizzano
        mai lo stato del job). Senza questa chiamata il job resta per sempre in
        "PROCESSING": una sessione di test futura lo troverebbe via
        _perform_active_recovery() e lo "riprenderebbe" da capo con iperparametri
        di default, sprecando lavoro dei worker su un job estraneo e ritardando
        il test reale. Fallisce in modo silenzioso (solo un warning) per non far
        crashare lo scenario se lo state_manager non è disponibile o l'update fallisce.
        """
        state_manager = getattr(self.orchestrator, "state_manager", None)
        if not state_manager:
            return
        try:
            state_manager.update_request_status(
                job_id=job_id,
                status="COMPLETED",
                orchestrator_id=getattr(self.orchestrator, "orchestrator_name", "TEST-SCENARIO"),
                retries=0,
                base_random_state=0,
                alberi_addestrati=alberi_addestrati,
            )
        except Exception as e:
            print(f"[TEST WARN] Impossibile finalizzare lo stato del job {job_id[:8]}: {e}")



    def _resolve_dataset_shape(self) -> dict:
        """
        Parametri di GENERAZIONE del dataset sintetico (n_samples, n_features,
        noise, n_informative_reg) — SIBLING di 'hyperparameters' nel manifesto,
        non al suo interno: _resolve_hyperparameters() li ignora di proposito
        (restituisce solo gli iperparametri sklearn confrontabili con la
        baseline). Senza questo metodo separato, n_samples del manifesto non
        arriva MAI ai worker: SyntheticDataLoader lato worker riceve solo il
        sotto-dizionario 'hyperparameters' e ricade sui propri default.
        """
        if self.config.get("dataset_type", "real") != "synthetic":
            return {}
        manifest_path = os.path.join(BASELINE_MANIFEST_DIR, "config_synthetic.json")
        if not os.path.exists(manifest_path):
            return {}
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f) or {}
        except (json.JSONDecodeError, OSError):
            return {}
        return {
            "n_samples": manifest.get("n_samples"),
            "n_features": manifest.get("n_features"),
            "noise": manifest.get("noise"),
            "n_informative_reg": manifest.get("n_informative_reg"),
        }