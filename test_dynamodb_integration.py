"""
Smoke test end-to-end per la migrazione a DynamoDB.

USO:
    python test_dynamodb_integration.py

Va lanciato UNA VOLTA con il tuo .env impostato su ambiente "local"
(userà il MockDynamoDB su file), e UNA VOLTA con l'ambiente impostato
su "aws" (userà le tabelle reali create in console). Stesso identico
script, stesso identico output atteso in entrambi i casi: è proprio
questo il punto di avere un'unica interfaccia.

Non usa alcun framework di test (pytest ecc.) per restare autocontenuto:
stampa PASS/FAIL riga per riga e un riepilogo finale. Tutti i dati di
test usano prefissi TEST_ e vengono ripuliti a fine esecuzione, quindi
è sicuro lanciarlo anche contro le tabelle reali.
"""

import time
import uuid

from src.shared.mock_aws.dynamodb.dynamodb_factory import DynamoDBFactory
from src.shared.factory import get_aws_services
from src.shared.config import SystemConfig

from src.shared.binding.serviceregistry import ServiceRegistry
from src.shared.binding.taskregistry import TaskRegistry

cfg = SystemConfig()
RESULTS = []


def check(label: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    RESULTS.append((label, condition))
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def section(title: str):
    print(f"\n=== {title} ===")


def main():
    print(f"Ambiente rilevato (cfg.env): '{cfg.env}'")
    db = DynamoDBFactory.get_db(cfg.env)

    # ------------------------------------------------------------------
    section("0. Interfaccia del client DB")
    for method in ("put_item", "get_item", "delete_item", "scan_table",
                   "put_item_if_not_exists", "query_by_index"):
        check(f"il client espone {method}()", hasattr(db, method))

    # ------------------------------------------------------------------
    section("1. CRUD di base su workers_registry")
    test_worker = f"TEST_worker_{uuid.uuid4().hex[:8]}"
    db.put_item("workers_registry", test_worker, {
        "host": "127.0.0.1", "port": 9999, "status": "AVAILABLE",
        "last_heartbeat": int(time.time()),
    })
    read_back = db.get_item("workers_registry", test_worker)
    check("put_item + get_item su workers_registry", read_back.get("Item", {}).get("status") == "AVAILABLE")

    deleted = db.delete_item("workers_registry", test_worker)
    check("delete_item su workers_registry", deleted)
    check("l'item non esiste più dopo la delete", db.get_item("workers_registry", test_worker) == {})

    # ------------------------------------------------------------------
    section("2. Conditional write (put_item_if_not_exists)")
    test_orch = f"TEST_orch_{uuid.uuid4().hex[:8]}"
    first_write = db.put_item_if_not_exists("orchestrators_registry", test_orch, {"status": "AVAILABLE"})
    second_write = db.put_item_if_not_exists("orchestrators_registry", test_orch, {"status": "AVAILABLE"})
    check("prima scrittura condizionale riesce", first_write is True)
    check("seconda scrittura condizionale sulla stessa chiave fallisce", second_write is False)
    db.delete_item("orchestrators_registry", test_orch)

    # ------------------------------------------------------------------
    section("3. ServiceRegistry (worker end-to-end)")
    ServiceRegistry.register_worker(test_worker, "127.0.0.1", 9999)
    available = ServiceRegistry.get_available_workers(cfg.env)
    check("il worker registrato compare tra i disponibili", test_worker in available)

    ServiceRegistry.update_worker_heartbeat(test_worker)
    refreshed = ServiceRegistry.get_available_workers(cfg.env)
    check("l'heartbeat aggiornato resta entro il timeout", test_worker in refreshed)

    ServiceRegistry.deregister_worker(test_worker)
    after_deregister = ServiceRegistry.get_available_workers(cfg.env)
    check("il worker deregistrato sparisce dai disponibili", test_worker not in after_deregister)

    # ------------------------------------------------------------------
    section("4. WorkerTasks + GSI (job_id-index, worker_name-index)")
    test_job = f"TEST_job_{uuid.uuid4().hex[:8]}"
    test_worker_2 = f"TEST_worker_{uuid.uuid4().hex[:8]}"
    task_ids = [f"{test_job}_1", f"{test_job}_2"]

    db.put_item("WorkerTasks", task_ids[0], {
        "job_id": test_job, "worker_name": test_worker_2, "status": "ASSIGNED",
        "updated_at": int(time.time()),
    })
    db.put_item("WorkerTasks", task_ids[1], {
        "job_id": test_job, "worker_name": test_worker_2, "status": "COMPLETED",
        "updated_at": int(time.time()),
    })

    # Su AWS reale i GSI sono eventually consistent: una piccola attesa
    # evita falsi negativi subito dopo la scrittura.
    if cfg.env != "local":
        time.sleep(2)

    by_job = TaskRegistry.get_tasks_by_job(test_job)
    check("get_tasks_by_job trova i 2 task inseriti", len(by_job) == 2, f"trovati: {len(by_job)}")

    by_worker = TaskRegistry.get_tasks_by_worker(test_worker_2)
    check("get_tasks_by_worker trova gli stessi 2 task", len(by_worker) == 2, f"trovati: {len(by_worker)}")

    # Pulizia
    db.delete_item("WorkerTasks", task_ids[0])
    db.delete_item("WorkerTasks", task_ids[1])

    # ------------------------------------------------------------------
    section("Riepilogo")
    passed = sum(1 for _, ok in RESULTS if ok)
    total = len(RESULTS)
    print(f"{passed}/{total} test superati.")
    if passed != total:
        print("Controlla i FAIL sopra prima di lanciare training/inferenza reali.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()