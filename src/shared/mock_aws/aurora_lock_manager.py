import psycopg2
import os
from datetime import datetime, timedelta

class AuroraLockManager:
    """
    In locale usa PostgreSQL, su AWS usa Aurora PostgreSQL serverless, 
    l'interfaccia è identica cambiano le variabili d'ambiente
    """

    TASK_TIMEOUT_SECONDS = 120

    def __init__(self):
        self.conn_param = {
            "host": os.environ.get("AURORA_HOST", "localhost"),
            "port": int(os.environ.get("AURORA_PORT",5432)),
            "dbname": os.environ.get("AURORA_DB","rf_locks"),
            "user":     os.environ.get("AURORA_USER", "admin"),
            "password": os.environ.get("AURORA_PASSWORD", "password"),
        }
    
    def _get_conn(self):
        return psycopg2.connect(**self.conn_params)
    
    def register_tasks(self, job_id: str, batches: list[dict]):
        """
        Registra tutti i batch di un job come PENDING all'inizio del training.
        batches = [{"task_id": "...", "start": 0, "target": 20}, ...]
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                for b in batches:
                    cur.execute("""
                        INSERT INTO task_registry
                            (task_id, job_id, start_alberi, target_alberi, status)
                        VALUES (%s, %s, %s, %s, 'PENDING')
                        ON CONFLICT (task_id) DO NOTHING
                    """, (b["task_id"], job_id, b["start"], b["target"]))

    def try_lock_task(self, task_id: str, worker_name: str) -> bool:
        """
        Tenta di acquisire il lock su un task PENDING.
        Usa SELECT FOR UPDATE per garantire atomicità — impossibile con DynamoDB.
        Ritorna True se il lock è stato acquisito, False se già preso.
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                # NOWAIT → fallisce subito se qualcun altro ha il lock
                cur.execute("""
                    SELECT task_id FROM task_registry
                    WHERE task_id = %s AND status = 'PENDING'
                    FOR UPDATE NOWAIT
                """, (task_id,))

                row = cur.fetchone()
                if not row:
                    return False    
                cur.execute("""
                    UPDATE task_registry
                    SET status = 'LOCKED',
                        worker_name = %s,
                        assigned_at = NOW(),
                        updated_at = NOW()
                    WHERE task_id = %s
                """, (worker_name, task_id))
                return True   
    
    def heartbeat_task(self, task_id: str):
        """Il worker chiama questo ogni 30s per segnalare che è ancora vivo."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE task_registry
                    SET updated_at = NOW()
                    WHERE task_id = %s AND status = 'LOCKED'
                """, (task_id,))

    def complete_task(self, task_id: str):
        """Il worker chiama questo quando ha finito il batch."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE task_registry
                    SET status = 'DONE', updated_at = NOW()
                    WHERE task_id = %s
                """, (task_id,))

    def recover_orphan_tasks(self, job_id: str) -> list[dict]:
        """
        L'orchestratore chiama questo periodicamente.
        Trova task LOCKED da più di TASK_TIMEOUT_SECONDS → worker morto → rimette PENDING.
        """
        threshold = datetime.now() - timedelta(seconds=self.TASK_TIMEOUT_SECONDS)

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                # Trova e rimette in PENDING in una sola operazione atomica
                cur.execute("""
                    UPDATE task_registry
                    SET status = 'PENDING',
                        worker_name = NULL,
                        updated_at = NOW()
                    WHERE job_id = %s
                      AND status = 'LOCKED'
                      AND updated_at < %s
                    RETURNING task_id, start_alberi, target_alberi
                """, (job_id, threshold))

                orphans = cur.fetchall()
                if orphans:
                    print(f"[AuroraLock] Recuperati {len(orphans)} task orfani per job {job_id[:8]}")
                return [{"task_id": r[0], "start": r[1], "target": r[2]} for r in orphans]

    def get_pending_tasks(self, job_id: str) -> list[dict]:
        """Ritorna tutti i task PENDING di un job — usato dall'orchestratore per redistribuire."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT task_id, start_alberi, target_alberi
                    FROM task_registry
                    WHERE job_id = %s AND status = 'PENDING'
                """, (job_id,))
                return [{"task_id": r[0], "start": r[1], "target": r[2]} for r in cur.fetchall()]
        

        