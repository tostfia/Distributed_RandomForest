import os
import sys
import time
import signal
import subprocess
from dotenv import load_dotenv

load_dotenv()

class InfrastructureManager:
    def __init__(self, mode, topology, exec, total_trees=20):
        # mode: "local" o "aws" (corrisponde a ENV_MODE nel sistema)
        # topology: "centralized" o "federated" (corrisponde a TRAINING_MODE nel sistema)
        self.mode = mode  
        self.exec = exec          
        self.topology = topology    
        self.total_trees = total_trees
        self.active_processes = {'orchestrator': None, 'workers': []}
        self.log_files = []

    def _run_cmd(self, cmd, env=None):
        try:
            # Passiamo l'ambiente modificato se presente (es. per impostare variabili contestuali al volo)
            current_env = {**os.environ, **(env or {})}
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, env=current_env)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Comando fallito: {cmd}\nErrore: {e.stderr}")
            return None

    def deploy(self, num_workers):
        print(f"\n[+] DEPLOY: Avvio [{self.topology.upper()}] con {num_workers} Worker su [{self.mode.upper()}] tramite [{self.exec.upper()}]...")
        
        env_context = {
            "ENV_MODE": self.mode,
            "TRAINING_MODE": self.topology,
            "NUM_WORKERS": str(num_workers),
            "PYTHONPATH": os.getcwd()
        }

        if self.exec == "docker":
            self._run_cmd("docker-compose down --remove-orphans")
            cmd_scale = f"docker-compose up -d --scale worker={num_workers} orchestrator"
            self._run_cmd(cmd_scale, env=env_context)
            time.sleep(6) 
        else:
            orch_module = "src.master.orchestrator.main"
            worker_module = "src.worker.centralizedWorker" if self.topology == "centralized" else "src.worker.federatedWorker"
            
            env_vars = {**os.environ, **env_context}
            
            # NOTA: Usiamo sys.executable con "-u" (unbuffered) e NON reindirizziamo su file log.
            # In questo modo vedrai le stampe reali dell'orchestratore sul tuo terminale!
            print("\n--- [LOG IN DIRETTA ORCHESTRATORE] ---")
            self.active_processes['orchestrator'] = subprocess.Popen(
                [sys.executable, "-u", "-m", orch_module],
                env=env_vars,
                preexec_fn=os.setsid if sys.platform != "win32" else None
            )
            time.sleep(3) 

            print("--- [LOG IN DIRETTA WORKERS] ---")
            self.active_processes['workers'] = []
            base_port = 18811  # La porta base per RPyC (cambiala se ne usi un'altra di default)

            for i in range(num_workers):
                worker_name = f"Worker-{i+1}"
                worker_port = str(base_port + i)
                # Rileviamo se usare classifier o regressor dalle impostazioni dell'infra
                tree_type = getattr(self, "tree_type", "classifier") 

                print(f"[+] Launching {worker_name} su porta {worker_port} ({tree_type})...")
                
                env_vars = {
                    **os.environ, 
                    "PYTHONPATH": os.getcwd(),
                    "ENV_MODE": self.mode,
                    "TRAINING_MODE": self.topology
                }
                
                
                p = subprocess.Popen([
                    sys.executable, "-u", "-m", "src.worker.main",  
                    worker_name,
                    worker_port,
                    tree_type
                ], env=env_vars)
                
                self.active_processes['workers'].append(p)
            time.sleep(2)
            print("---------------------------------------\n")

    def teardown(self):
        print("[-] TEARDOWN: Pulizia infrastruttura...")
        if self.exec == "docker":
            self._run_cmd("docker-compose down -v")
        else:
            for p in self.active_processes['workers']:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM) if sys.platform != "win32" else p.terminate()
                except (ProcessLookupError, AttributeError): pass
            self.active_processes['workers'] = []

            p_orch = self.active_processes['orchestrator']
            if p_orch:
                try:
                    os.killpg(os.getpgid(p_orch.pid), signal.SIGTERM) if sys.platform != "win32" else p_orch.terminate()
                except (ProcessLookupError, AttributeError): pass
                self.active_processes['orchestrator'] = None

            for f in self.log_files:
                if not f.closed: f.close()
            self.log_files = []

            lock_path = "./.local_storage/global_orchestrator_leader_lock.json"
            if os.path.exists(lock_path):
                try: os.remove(lock_path)
                except Exception: pass
        print("[v] Pulizia completata.")