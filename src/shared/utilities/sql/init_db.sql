CREATE TABLE task_registry (
    task_id        VARCHAR(64) PRIMARY KEY,  -- job_id + batch_index
    job_id         VARCHAR(64) NOT NULL,
    worker_name    VARCHAR(64),
    start_alberi   INT NOT NULL,
    target_alberi  INT NOT NULL,
    status         VARCHAR(16) DEFAULT 'PENDING',  -- PENDING, LOCKED, DONE
    assigned_at    TIMESTAMP,
    updated_at     TIMESTAMP DEFAULT NOW()
);