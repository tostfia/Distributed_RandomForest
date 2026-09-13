import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from src.dataset.dataset_dao_factory import DatasetDAOFactory
from src.worker.BaseWorker import BaseWorker


class CentralizedWorker(BaseWorker):
    
    """Worker per la gestione dell'addestramento in modalità centralizzata."""

    def __init__(
        self,
        worker_name: str,
        queue_name: str,
        tree_class_reference: type,
        target_column: str,
        max_samples: float = None,
        bootstrap: bool = True,
        tree_type: str = "classifier"
    ):
        super().__init__(
            worker_name=worker_name,
            queue_name=queue_name,
            tree_class_reference=tree_class_reference,
            max_samples=max_samples,
            bootstrap=bootstrap,
        )
        self.target_column = target_column
        self.tree_type = tree_type
        
        self.dao = DatasetDAOFactory.get_dao()
        
        print(
            f"[CentralizedWorker] Inizializzato in ambiente: {self.environment.upper()} "
            f"con DAO: {type(self.dao).__name__}"
        )

        self._cached_source = None
        self._cached_content_length = None
        self._cached_X = None
        self._cached_y = None
    
    def is_regression(self) -> bool:
        return self.tree_type == "regressor"
    
    def _get_tree_class(self) -> type:
        """Restituisce la classe corretta in base al tipo di task rilevato."""
        if self.is_regression():
            return DecisionTreeRegressor
        else:
            return DecisionTreeClassifier

    def _load_data(self, source_info: str) -> tuple[np.ndarray, np.ndarray]:
        
        """Carica il dataset centralizzato delegando al DAO e lo trasforma in matrici NumPy.
        Args:
            source_info (str): URL S3 o path locale. Se contiene '|', è
            l'unione di PIÙ shard (13/9/2026, sharding fisso per il reale:
            quando ci sono meno worker attivi degli shard totali, ogni
            worker riceve più shard assegnati, codificati in un'unica
            stringa delimitata dall'Orchestratore - MAI una lista Python:
            RPyC trasmette le liste per riferimento/netref, non per valore,
            causando EOFError se il worker prova a usarle dopo che la
            connessione originale si è chiusa. Vedi bugfix in centralized.py,
            dispatch loop, stesso giorno.
        """
        is_multi = "|" in source_info
        # Decodifica UNA volta qui: entrambi i punti sotto (calcolo cache e
        # caricamento vero e proprio) iterano su questa lista LOCALE, mai
        # sulla stringa 'source_info' carattere per carattere.
        shard_list = source_info.split("|") if is_multi else None

        # CACHE SU CONTENUTO invece che su URL esatto (11/9/2026): la cache
        # precedente confrontava 'source_info' (l'URL S3 completo) - ma ogni
        # job genera un URL diverso (es. 'shared_train_test_scal_3_....csv'
        # vs 'shared_train_test_scal_5_....csv'), anche quando il CONTENUTO
        # è identico (stesso seed/parametri, vedi "[TEST CACHE AWS] Dataset
        # riusato..." nei log dell'orchestratore). Il confronto sull'URL
        # falliva quindi sempre tra una configurazione di scaling e l'altra,
        # forzando un ri-download completo (~66-100MB, 24-35s MISURATI
        # empiricamente l'11/9/2026) anche quando lo stesso identico
        # container aveva già quei dati in memoria dal giro precedente.
        #
        # get_content_length() costa un head_object (solo metadata, nessun
        # trasferimento del contenuto) invece di un get_object completo: il
        # confronto qui sotto costa quindi decine di ms anche in caso di
        # cache MISS, contro i 24-35s di un download - il costo aggiuntivo
        # nel caso peggiore (contenuto diverso, serve comunque scaricare) è
        # trascurabile rispetto al beneficio nel caso migliore (contenuto
        # identico, download evitato del tutto).
        #
        # CASO LISTA (13/9/2026): somma dei content_length di TUTTI gli shard
        # assegnati - stesso principio (confronto economico, nessun
        # download per il solo controllo), sommato invece che singolo. Una
        # collisione tra combinazioni diverse di shard con la STESSA somma
        # è statisticamente trascurabile per questo uso (cache di velocità,
        # non di correttezza) e comunque mai osservata empiricamente.
        try:
            if is_multi:
                content_length = sum(self.dao.get_content_length(p) for p in shard_list)
            else:
                content_length = self.dao.get_content_length(source_info)
        except Exception as e:
            # Se il controllo leggero fallisce per qualunque motivo (rete,
            # permessi, DAO locale senza supporto), non deve bloccare il
            # training: si procede come se fosse un cache miss, esattamente
            # il comportamento di prima di questa modifica.
            print(f"[CentralizedWorker] [WARN] get_content_length fallita ({e}), "
                  f"procedo senza cache su contenuto (comportamento pre-fix).")
            content_length = None

        if (
            content_length is not None
            and self._cached_content_length == content_length
            and self._cached_X is not None
            and self._cached_y is not None
        ):
            print(f"[CentralizedWorker] Dati già in cache (stessa dimensione totale: "
                  f"{content_length} byte) - nessun ri-download nonostante l'URL/lista "
                  f"diversa da quella del job precedente.")
            self._cached_source = source_info  # aggiornato per coerenza/debug
            return self._cached_X, self._cached_y

        if self._cached_source == source_info and self._cached_X is not None and self._cached_y is not None:
            print("[CentralizedWorker] Utilizzo dei dati già caricati in cache.")
            return self._cached_X, self._cached_y

        if is_multi:
            print(f"[CentralizedWorker] Richiesta di caricamento dati tramite DAO da "
                  f"{len(shard_list)} shard: {shard_list}")
            # Shard scaricati e concatenati in un unico DataFrame prima
            # dell'estrazione X/y sotto: stessa logica di risoluzione target/
            # feature applicata UNA volta al risultato unito, non ripetuta
            # per shard (tutti gli shard condividono lo stesso schema di
            # colonne per costruzione, essendo partizioni dello stesso
            # dataset già processato dall'Orchestratore).
            dfs = [self.dao.load_dataset(p) for p in shard_list]
            df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
            del dfs
        else:
            print(f"[CentralizedWorker] Richiesta di caricamento dati tramite DAO da: {source_info}")
            df: pd.DataFrame = self.dao.load_dataset(source_info)

        if self.is_regression():
            self.target_column = "Target"
        else:
            self.target_column = "Label"

        actual_target = self.target_column if self.target_column in df.columns else (
            "Target" if "Target" in df.columns else "Label"
        )
        if actual_target not in df.columns:
            raise ValueError(
                f"Colonna target '{actual_target}' non trovata nel dataset. "
                f"Colonne disponibili: {df.columns.tolist()}"
            )
        
        feature_cols = [c for c in df.columns if c != actual_target]
        
        y_df = df[actual_target]
        X_df = df[feature_cols]

        # FIX MEMORIA (vedi OOM osservato con 10 worker che caricano
        # simultaneamente l'intero dataset centralizzato da 1M righe):
        # 1) float32 invece di float64 per X -- dimezza il picco di RAM
        #    (1M x 100 x 4 byte invece di 8) SENZA perdita di precisione
        #    reale: sklearn.tree lavora internamente in float32
        #    (tree._tree.DTYPE) e a fit-time avrebbe comunque ricopiato/
        #    convertito X in float32, tenendo per un istante ENTRAMBE le
        #    copie in RAM. Costruirlo già in float32 elimina questa
        #    doppia copia invece di limitarsi ad approssimare i dati.
        # 2) 'del df' subito dopo aver estratto X/y: 'df' e le sue view
        #    (X_df/y_df) restano altrimenti vive fino al return della
        #    funzione, quindi per tutta la costruzione di X/y convivono in
        #    RAM sia il DataFrame originale sia gli array numpy appena
        #    copiati -- un picco transitorio di 2-3x la dimensione finale
        #    dei dati, proprio nell'istante più delicato (10 worker che
        #    lo fanno tutti insieme).
        X = X_df.to_numpy(dtype=np.float32)
        if y_df.dtype == 'object' or y_df.nunique() > 20:
            y = y_df.to_numpy(dtype=np.float64)
            self.tree_type = "regressor"
        else:
            y = y_df.to_numpy(dtype=np.int64)
            self.tree_type = "classifier"

        del df, X_df, y_df

        self._cached_source = source_info
        self._cached_content_length = content_length
        self._cached_X = X
        self._cached_y = y
        print(
            f"[CentralizedWorker] Dati caricati con successo: "
            f"X shape = {X.shape}, y shape = {y.shape}"
        )
        
        return self._cached_X, self._cached_y