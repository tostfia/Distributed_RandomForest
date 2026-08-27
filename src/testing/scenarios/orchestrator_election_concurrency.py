"""
Scenario di test: Elezione del Leader sotto Concorrenza Reale.

A differenza di OrchestratorFailoverScenario (orchestrator_fault.py), che
verifica la LIVENESS del meccanismo di leader election (un failover avviene
e il job riparte), questo scenario verifica la sua SAFETY: la mutua
esclusione del lock di leadership (tabella OrchestratorLocks, vedi
AwsStateManager.acquire_global_lock / MockStateManager.acquire_global_lock)
sotto vera concorrenza, non in sequenza.

Il test di failover esistente avvia Leader e Standby IN SEQUENZA (prima
l'uno, poi l'altro con un ritardo): non esercita mai il caso in cui più
candidati chiamano acquire_global_lock() nello stesso istante. Questo
scenario colma quel gap testando direttamente la primitiva di lock, senza
istanziare Orchestratori completi (nessun dataset/ETL/worker/RPC coinvolto):
N thread "candidati" sincronizzati con un threading.Barrier competono per lo
stesso lock_key su più round, e per ognuno si verifica che vinca UNO E UN
SOLO candidato.

Funziona identicamente in locale e su AWS perché si appoggia esclusivamente
a StateManagerInterface (self.orchestrator.state_manager), già istanziato
da TestEngine per l'ambiente corrente:
  - locale: MockStateManager -> MockDynamoDB.try_acquire_lock, che ottiene
    l'atomicità con un file lock reale (fcntl.flock, LOCK_EX) attorno alla
    read-modify-write (vedi dynamodb_mock.py::_locked_read_modify_write) —
    atomico anche fra thread diversi dello stesso processo, non solo fra
    processi separati.
  - AWS: AwsStateManager -> AwsDynamoDB.try_acquire_lock, che ottiene
    l'atomicità con una ConditionExpression valutata lato server DynamoDB
    (attribute_not_exists(lock_key) OR expires_at < :now). Il confronto
    ':now' è calcolato lato CLIENT (vedi dynamodb_aws.py): la correttezza
    di QUESTO specifico test non dipende dal clock (i candidati competono
    nello stesso istante, non su TTL già scaduti), ma è la stessa
    dipendenza discussa in relazione a proposito del recovery del leader.

Usa una lock_key dedicata (_ELECTION_TEST_LOCK_KEY), diversa da
'global_orchestrator_leader_lock' usata in produzione: non deve interferire
con l'orchestratore reale eventualmente in esecuzione nello stesso processo
(self.orchestrator, già istanziato da TestEngine anche per gli altri
scenari).
"""
import threading
import time

from src.testing.scenarios.base import BaseTestScenario

_ELECTION_TEST_LOCK_KEY = "test_election_concurrency_lock"


def _merge_aws_overrides(config: dict, key: str, environment: str) -> dict:
    """
    Logica analoga a orchestrator_fault.py::_merge_aws_overrides, con UNA
    differenza deliberata: qui l'override viene applicato SOLO quando
    environment == "aws". La versione originale in orchestrator_fault.py non
    fa questo controllo e unisce sempre il blocco 'aws.suggested_overrides',
    anche in esecuzione locale — innocuo per quello scenario (i valori AWS
    sono più permissivi di quelli locali, quindi al più si attende più del
    necessario), ma qui andrebbe nella direzione opposta: 'num_rounds' AWS
    (5) è più BASSO di quello locale (20), quindi senza questo controllo un
    run locale userebbe silenziosamente una copertura ridotta pensata per
    contenere i costi DynamoDB reali nel Learner Lab.
    """
    merged = dict(config.get(key, {}) or {})
    if environment == "aws" and (config.get("aws", {}) or {}).get("suggested_overrides", {}).get(key):
        overrides = config["aws"]["suggested_overrides"][key]
        merged.update({k: v for k, v in overrides.items() if not k.startswith("_")})
    return merged


class OrchestratorElectionConcurrencyScenario(BaseTestScenario):
    """
    Verifica la mutua esclusione del lock di leadership sotto concorrenza
    reale (proprietà di SAFETY), a complemento di OrchestratorFailoverScenario
    (che verifica LIVENESS/recovery ma avvia i contendenti in sequenza, non
    sotto concorrenza vera). Non coinvolge dataset, ETL, worker o job: opera
    solo sulla primitiva di lock esposta da StateManagerInterface.
    """

    def run(self) -> dict:
        state_manager = self.orchestrator.state_manager
        environment = getattr(self.orchestrator, "environment", "local")

        cfg = _merge_aws_overrides(self.config, "orchestrator_election_concurrency", environment)
        num_rounds = int(cfg.get("num_rounds", 20))
        num_candidates = int(cfg.get("num_candidates", 5))
        lock_ttl_seconds = int(cfg.get("lock_ttl_seconds", 5))

        backend_desc = (
            "AWS DynamoDB (ConditionExpression lato server)"
            if environment == "aws"
            else "Mock locale su file (fcntl.flock, LOCK_EX)"
        )

        print(f"\n--- [TEST] Elezione del Leader sotto Concorrenza Reale "
              f"({num_candidates} candidati x {num_rounds} round, backend: {backend_desc}) ---")

        start_time = time.perf_counter()
        violations = []
        rounds_ok = 0

        for round_index in range(num_rounds):
            barrier = threading.Barrier(num_candidates)
            results = {}
            results_lock = threading.Lock()

            def _candidate(owner_name: str):
                try:
                    # Tutti i candidati attendono qui: partono verso
                    # acquire_global_lock nello stesso istante, non in
                    # sequenza con ritardi artificiali (a differenza del
                    # ramo locale di OrchestratorFailoverScenario, che avvia
                    # Leader e Standby con un time.sleep(2) in mezzo).
                    barrier.wait(timeout=10)
                except threading.BrokenBarrierError:
                    with results_lock:
                        results[owner_name] = False
                    return
                try:
                    acquired = state_manager.acquire_global_lock(
                        _ELECTION_TEST_LOCK_KEY, owner_name, ttl=lock_ttl_seconds
                    )
                except Exception as e:
                    print(f"[TEST WARN] Round {round_index}: errore da acquire_global_lock "
                          f"per '{owner_name}': {e}")
                    acquired = False
                with results_lock:
                    results[owner_name] = acquired

            owners = [
                f"{self.orchestrator.orchestrator_name}-ElectionTest-r{round_index}-c{i}"
                for i in range(num_candidates)
            ]
            threads = [
                threading.Thread(target=_candidate, args=(o,), name=f"candidate-{i}")
                for i, o in enumerate(owners)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

            winners = [o for o, acquired in results.items() if acquired]

            if len(winners) == 1:
                rounds_ok += 1
                # Rilascio esplicito: prepara un round successivo pulito
                # senza dover attendere il TTL naturale. Se il rilascio
                # fallisse (non dovrebbe: il possessore coincide con
                # 'winners[0]'), il round successivo lo scoprirebbe da sé
                # (0 vincitori finché il TTL non scade) — non è quindi un
                # fallimento silenzioso.
                try:
                    released = state_manager.release_global_lock(_ELECTION_TEST_LOCK_KEY, winners[0])
                    if not released:
                        print(f"[TEST WARN] Round {round_index}: rilascio del lock da parte del "
                              f"vincitore '{winners[0]}' non confermato.")
                except Exception as e:
                    print(f"[TEST WARN] Round {round_index}: eccezione durante il rilascio: {e}")
            else:
                violation = {
                    "round": round_index,
                    "winners_count": len(winners),
                    "winners": winners,
                }
                violations.append(violation)
                print(f"[TEST VIOLATION] Round {round_index}: {len(winners)} vincitori invece di 1 "
                      f"({winners if winners else 'nessuno'}).")
                # Stato incerto dopo una violazione (0 o più di 1 vincitori):
                # forziamo la rimozione dell'eventuale entry residua, così il
                # round successivo riparte da lock libero invece di
                # propagare l'anomalia ai round seguenti.
                for o in winners:
                    try:
                        state_manager.release_global_lock(_ELECTION_TEST_LOCK_KEY, o)
                    except Exception:
                        pass

        duration = time.perf_counter() - start_time
        status = "SUCCESS" if not violations else "FAILED"

        if status == "SUCCESS":
            print(f"\n[TEST PASSED] Mutua esclusione verificata su tutti i {num_rounds} round "
                  f"({num_candidates} candidati/round, backend: {backend_desc}).")
        else:
            print(f"\n[TEST FAILED] Mutua esclusione violata in {len(violations)}/{num_rounds} round.")

        return {
            "scenario_description": "Verifica la mutua esclusione del lock di leadership "
                                     "(OrchestratorLocks) sotto concorrenza reale: N candidati "
                                     "sincronizzati con threading.Barrier competono sullo stesso "
                                     "lock_key in ogni round. Complementare a "
                                     "OrchestratorFailoverScenario, che verifica liveness/recovery "
                                     "ma avvia i contendenti in sequenza, non sotto vera concorrenza.",
            "status": status,
            "backend": backend_desc,
            "environment": environment,
            "num_rounds": num_rounds,
            "num_candidates_per_round": num_candidates,
            "lock_ttl_seconds": lock_ttl_seconds,
            "rounds_with_exactly_one_winner": rounds_ok,
            "violations": violations,
            "duration_seconds": round(duration, 2),
        }