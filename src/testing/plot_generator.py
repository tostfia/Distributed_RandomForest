"""
Generatore automatico dei grafici per la relazione finale.

Legge i report JSON prodotti da TestEngine._print_final_summary() e il manifesto
.pkl prodotto da run_baseline(), e produce i PNG in ./plots.

PRINCIPIO DI ROBUSTEZZA (requisito esplicito del progetto):
    Nessun metodo di questa classe puo' far fallire l'esecuzione. Ogni accesso a
    file e' preceduto da os.path.exists(), ogni accesso a dizionario passa da
    dict.get(), e ogni generatore e' eseguito dentro un wrapper che intercetta
    qualunque eccezione. Se i dati per un grafico non ci sono (perche' quel test
    non e' mai stato eseguito, o e' stato SKIPPED), il grafico viene saltato con
    un [WARN] e la generazione prosegue.

PRIORITA' DELLE SORGENTI:
    test_reports/aws/  ->  test_reports/docker/  ->  test_reports/local/
    Il primo ambiente che contiene dati utilizzabili diventa l'ambiente
    "primario" per i grafici a singolo ambiente; gli altri restano comunque
    caricati e vengono usati per i confronti cross-ambiente.

Uso:
    python -m src.testing.plot_generator
oppure, dall'engine:
    PlotGenerator().generate_all_plots()
"""

import glob
import json
import os
import pickle
import traceback

import matplotlib

# Backend non interattivo: obbligatorio dentro container Docker/ECS, dove non
# esiste alcun display. Va impostato PRIMA di importare pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Palette e stile: una sola definizione, riusata da tutti i grafici, cosi' la
# relazione risulta visivamente coerente da un capitolo all'altro.
# ---------------------------------------------------------------------------
PALETTE = {
    "primary":    "#1F4E79",   # blu profondo  - misura principale
    "secondary":  "#C0504D",   # rosso mattone - misura di confronto
    "accent":     "#E8A33D",   # ambra         - evidenziazioni
    "success":    "#4F8F5B",   # verde         - riferimenti positivi
    "neutral":    "#7F7F7F",   # grigio        - curve ideali / teoriche
    "light":      "#AFC7DC",
}

# Colori delle fasi nella scomposizione dei tempi (grafico di Amdahl).
PHASE_COLORS = {
    "etl_seconds":            "#C0504D",
    "training_only_seconds":  "#1F4E79",
    "aggregation_seconds":    "#E8A33D",
    "oob_estimation_seconds": "#4F8F5B",
    "unaccounted_seconds":    "#BFBFBF",
}

PHASE_LABELS = {
    "etl_seconds":            "ETL / preparazione dati (seriale)",
    "training_only_seconds":  "Costruzione alberi (parallelo)",
    "aggregation_seconds":    "Aggregazione modello (seriale)",
    "oob_estimation_seconds": "Stima OOB (seriale)",
    "unaccounted_seconds":    "Overhead non attribuito",
}


class PlotGenerator:
    """Costruisce i grafici PNG per la relazione a partire dai report di test."""

    ENV_PRIORITY = ("aws", "docker", "local")

    # Scenari che possono contenere metriche di qualita' del modello, in ordine
    # di affidabilita' (vedi _find_accuracy_metrics).
    METRIC_SOURCES = ("performance_and_metrics", "scalability",
                      "network_simulation", "inference_worker_fault")

    ENV_LABELS = {
        "aws":    "AWS ECS Fargate",
        "docker": "Docker Compose (locale)",
        "local":  "Bare-metal (locale)",
    }

    def __init__(
        self,
        reports_root: str = "test_reports",
        plots_dir: str = "plots",
        baseline_pkl: str = os.path.join("outputs_baseline", "baseline_random_forest_completa.pkl"),
        dpi: int = 160,
    ):
        self.reports_root = reports_root
        self.plots_dir = plots_dir
        self.baseline_pkl_path = baseline_pkl
        self.dpi = dpi

        # Una voce per file di report: {env, mode, path, file, mtime,
        # scenarios, fingerprint}. Vedi _load_all_reports.
        self.runs = []
        self.primary_env = None
        self.primary_mode = None

        self.baseline = None          # contenuto del .pkl, se leggibile
        self.generated = []           # path dei PNG effettivamente prodotti
        self.skipped = []             # (nome_grafico, motivo)

        self._apply_style()
        self._ensure_plots_dir()
        self._load_all_reports()
        self._load_baseline()

    # =======================================================================
    # SETUP
    # =======================================================================

    def _apply_style(self):
        """Stile accademico sobrio: griglia leggera, niente cornici superflue."""
        sns.set_theme(style="whitegrid", context="paper")
        plt.rcParams.update({
            "figure.dpi": 110,
            "savefig.dpi": self.dpi,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
            "axes.labelsize": 12,
            "axes.labelpad": 8,
            "axes.edgecolor": "#4D4D4D",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#D9D9D9",
            "figure.autolayout": False,
        })

    def _ensure_plots_dir(self):
        try:
            os.makedirs(self.plots_dir, exist_ok=True)
        except OSError as e:
            print(f"[WARN] Impossibile creare la cartella '{self.plots_dir}': {e}")

    # =======================================================================
    # CARICAMENTO DATI
    # =======================================================================

    @staticmethod
    def _infer_mode_from_filename(filename: str) -> str:
        """
        Deduce la modalita' di addestramento dal nome del file.

        TestEngine oggi salva sempre 'test_report_<scenario>.json', senza la
        modalita': in quel caso restituiamo 'unknown' e i grafici a singola
        modalita' funzionano lo stesso. Se invece il nome contiene
        'centralized'/'federated' (o il file sta in una sottocartella con quel
        nome), il confronto Centralizzato vs Federato diventa possibile.
        """
        lowered = filename.lower()
        if "federated" in lowered or "federato" in lowered:
            return "federated"
        if "centralized" in lowered or "centralizzato" in lowered:
            return "centralized"
        return "unknown"

    def _load_all_reports(self):
        """
        Carica i JSON di test_reports/<env>[/<sottocartella>]/*.json.

        UN FILE = UNA RUN. Questo e' il punto piu' importante di tutta la
        classe: ogni file di report e' il prodotto di UNA sessione di test, con
        un suo dataset, un suo numero di alberi e una sua configurazione. File
        diversi nella stessa cartella sono, quasi sempre, ESPERIMENTI DIVERSI
        eseguiti in momenti diversi.

        Fondere gli scenari di piu' file in un unico dizionario (come faceva la
        prima versione) produce grafici formalmente corretti e sostanzialmente
        falsi: si finisce per confrontare il tempo di un job da 30 alberi sul
        dataset reale con quello di un job da 100 alberi sul sintetico, e a
        concludere che "il crash del worker rende il sistema piu' veloce".

        Per questo ogni run resta separata, e ogni grafico comparativo attinge
        da UNA sola run.
        """
        if not os.path.isdir(self.reports_root):
            print(f"[WARN] Cartella dei report '{self.reports_root}' inesistente: "
                  f"nessun grafico basato sui test potra' essere generato.")
            return

        for env in self.ENV_PRIORITY:
            env_dir = os.path.join(self.reports_root, env)
            if not os.path.isdir(env_dir):
                continue

            # Ricerca ricorsiva: gestisce sia test_reports/local/*.json sia
            # test_reports/local/federated/*.json (organizzazione consigliata
            # per poter confrontare le due modalita').
            paths = sorted(glob.glob(os.path.join(env_dir, "**", "*.json"), recursive=True))
            for path in paths:
                payload = self._read_json(path)
                if not isinstance(payload, dict) or not payload:
                    continue

                scenarios = {k: v for k, v in payload.items() if isinstance(v, dict)}
                if not scenarios:
                    print(f"[WARN] '{path}' non contiene alcuno scenario valido: lo salto.")
                    continue

                rel = os.path.relpath(path, env_dir)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    mtime = 0.0

                self.runs.append({
                    "env": env,
                    "mode": self._infer_mode_from_filename(rel),
                    "path": path,
                    "file": os.path.basename(path),
                    "mtime": mtime,
                    "scenarios": scenarios,
                    "fingerprint": self._fingerprint(scenarios),
                })

        if not self.runs:
            print("[WARN] Nessun report utilizzabile trovato in "
                  f"'{self.reports_root}/{{aws,docker,local}}'.")
            return

        self._print_provenance()

        primary = self._pick_run(["performance_and_metrics", "scalability"])
        if primary:
            self.primary_env = primary["env"]
            self.primary_mode = primary["mode"]

    @staticmethod
    def _fingerprint(scenarios: dict) -> dict:
        """
        Riassume la configurazione di una run: serve a riconoscere a colpo
        d'occhio (e a stampare) che due file NON descrivono lo stesso
        esperimento. Chiavi: numero di alberi, dimensione del test set, tipo di
        task.
        """
        trees, test_sizes, tasks = set(), set(), set()
        for block in scenarios.values():
            value = block.get("trees_built")
            if isinstance(value, (int, float)) and value > 0:
                trees.add(int(value))
            for metrics in (block.get("model_accuracy_metrics"), block.get("accuracy_metrics")):
                if isinstance(metrics, dict):
                    size = metrics.get("testing_set_size")
                    if isinstance(size, (int, float)) and size > 0:
                        test_sizes.add(int(size))
                    tasks.add("regressione" if "mse" in metrics or "r2" in metrics
                              else "classificazione")
        return {"trees": sorted(trees), "test_sizes": sorted(test_sizes),
                "tasks": sorted(tasks)}

    def _print_provenance(self):
        """
        Tabella di provenienza: quale file, quale ambiente, quanti scenari, con
        quale configurazione. E' la prima cosa da guardare quando un grafico
        sembra assurdo — di solito la risposta e' che due righe di questa
        tabella non parlano dello stesso esperimento.
        """
        print("\n[PLOT] Report rilevati (una riga = una sessione di test):")
        print(f"        {'AMBIENTE':<8} {'FILE':<44} {'SCEN.':>5}  CONFIGURAZIONE")
        for run in sorted(self.runs, key=lambda r: (self.ENV_PRIORITY.index(r["env"]), r["file"])):
            fp = run["fingerprint"]
            descr = []
            if fp["trees"]:
                descr.append("alberi: " + "/".join(str(t) for t in fp["trees"]))
            if fp["test_sizes"]:
                descr.append("test set: " + "/".join(f"{s:,}".replace(",", ".")
                                                    for s in fp["test_sizes"]))
            if fp["tasks"]:
                descr.append("/".join(fp["tasks"]))
            print(f"        {run['env']:<8} {run['file'][:44]:<44} "
                  f"{len(run['scenarios']):>5}  {' | '.join(descr) if descr else 'n/d'}")

        # Avviso esplicito quando nello stesso ambiente convivono run con
        # configurazioni diverse: e' il caso in cui i confronti fra file
        # sarebbero privi di senso, ed e' esattamente cio' che questa classe
        # ora si rifiuta di fare.
        for env in self.ENV_PRIORITY:
            runs = [r for r in self.runs if r["env"] == env]
            trees = {t for r in runs for t in r["fingerprint"]["trees"]}
            sizes = {s for r in runs for s in r["fingerprint"]["test_sizes"]}
            if len(runs) > 1 and (len(trees) > 1 or len(sizes) > 1):
                print(f"[WARN] In 'test_reports/{env}' convivono report con carichi o dataset "
                      f"DIVERSI (alberi: {sorted(trees)}, test set: {sorted(sizes)}). "
                      f"Sono esperimenti distinti: ogni grafico usera' una sola run e non "
                      f"mettera' mai a confronto file diversi. Per una relazione pulita, "
                      f"svuota la cartella e rilancia una volta sola con SCENARIO=all.")

    def _read_json(self, path: str):
        if not os.path.exists(path):
            print(f"[WARN] File '{path}' non trovato: lo salto.")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[WARN] File '{path}' non e' JSON valido ({e}): lo salto.")
        except OSError as e:
            print(f"[WARN] Impossibile leggere '{path}' ({e}): lo salto.")
        return None

    def _load_baseline(self):
        """
        Carica il manifesto .pkl della baseline locale.

        Il file contiene un oggetto sklearn: se la versione di scikit-learn
        installata qui e' incompatibile con quella che lo ha prodotto, il
        pickle.load() puo' fallire. E' un caso previsto, non un errore fatale:
        si perdono solo i grafici che dipendono dalla baseline.
        """
        path = self.baseline_pkl_path
        if not os.path.exists(path):
            print(f"[WARN] Baseline '{path}' non trovata: i grafici che la usano "
                  f"(feature importance, confronto ML, riferimenti T_seq) saranno saltati.")
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self.baseline = data
                shape = (data.get("dataset_shape") or {})
                print(f"[PLOT] Baseline caricata da '{path}' "
                      f"(task: {shape.get('tree_type', 'n/d')}, "
                      f"dataset: {shape.get('dataset_type', 'n/d')}).")
            else:
                print(f"[WARN] '{path}' non contiene il dizionario atteso: lo ignoro.")
        except Exception as e:  # pickle puo' sollevare praticamente qualsiasi cosa
            print(f"[WARN] Impossibile deserializzare '{path}' ({type(e).__name__}: {e}). "
                  f"I grafici basati sulla baseline saranno saltati.")

    # =======================================================================
    # ACCESSO AI DATI (tutto via .get(), mai indicizzazione diretta)
    # =======================================================================

    @staticmethod
    def _scenarios(run) -> dict:
        """Gli scenari di una run (dizionario vuoto se la run e' None)."""
        return (run or {}).get("scenarios", {})

    def _scenario(self, name: str, run=None):
        """Uno scenario specifico dentro una run, o None se assente/non valido."""
        data = self._scenarios(run).get(name)
        return data if isinstance(data, dict) else None

    def _pick_run(self, required_names, mode=None, prefer="env"):
        """
        Sceglie LA run (cioe' il file, cioe' la sessione di test) da cui
        prendere i dati per UN grafico.

        Criteri, in ordine:
          1. priorita' d'ambiente aws -> docker -> local, ma solo fra le run che
             contengono almeno uno degli scenari richiesti (se su AWS manca lo
             scenario di scalabilita', la curva viene presa da Docker e
             l'ambiente effettivo finisce nel sottotitolo della figura);
          2. copertura: a parita' d'ambiente vince la run che contiene piu'
             scenari fra quelli richiesti — tipicamente 'test_report_all_tests.json',
             che e' anche l'unica intrinsecamente coerente;
          3. la piu' recente.

        Un grafico non mescola MAI due run: e' cio' che impedisce di confrontare
        misure prese su carichi o dataset diversi.
        """
        best, best_key = None, None
        for run in self.runs:
            if mode is not None and run["mode"] != mode:
                continue
            scenarios = run["scenarios"]
            coverage = sum(1 for n in required_names if self._is_usable(scenarios.get(n)))
            if coverage == 0:
                continue
            env_rank = -self.ENV_PRIORITY.index(run["env"])
            # prefer="env": conta prima l'ambiente (un grafico a sorgente unica
            #   deve venire da AWS se AWS ce l'ha).
            # prefer="coverage": conta prima quanti degli scenari richiesti sono
            #   presenti nella STESSA sessione — indispensabile per i grafici
            #   comparativi, dove una run AWS con il solo job pulito non puo'
            #   battere una run locale che contiene job pulito E guasti.
            # A parita', vince la sessione piu' completa (tipicamente
            # 'test_report_all_tests.json') e poi la piu' recente.
            key = ((env_rank, coverage) if prefer == "env" else (coverage, env_rank))
            key = key + (len(run["scenarios"]), run["mtime"])
            if best_key is None or key > best_key:
                best, best_key = run, key
        return best

    @staticmethod
    def _is_usable(scenario: dict) -> bool:
        """
        Uno scenario e' utilizzabile se esiste e non e' stato saltato.
        Gli status 'SKIPPED_*' sono legittimi (es. niente CAP_NET_ADMIN su
        Fargate) ma non contengono misure confrontabili.
        """
        if not isinstance(scenario, dict):
            return False
        status = str(scenario.get("status", "SUCCESS")).upper()
        return not status.startswith("SKIPPED") and status != "FAILED"

    @staticmethod
    def _get_float(container, key, default=None):
        """Estrae un numero da un dict, restituendo default se assente o non numerico."""
        if not isinstance(container, dict):
            return default
        value = container.get(key, default)
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
        return default

    @staticmethod
    def _normalize_metrics(metrics: dict) -> dict:
        """
        Uniforma i nomi delle metriche fra baseline e sistema distribuito.
        La baseline salva 'f1'/'roc_auc', il cluster 'f1_score'/'auc':
        senza questa normalizzazione il grafico di confronto mostrerebbe
        barre vuote su meta' delle metriche.
        """
        if not isinstance(metrics, dict):
            return {}
        aliases = {
            "f1": "f1_score",
            "roc_auc": "auc",
            "mean_squared_error": "mse",
        }
        out = {}
        for key, value in metrics.items():
            out[aliases.get(key, key)] = value
        return out

    def _find_accuracy_metrics(self, run=None):
        """
        Cerca il primo blocco di metriche di qualita' disponibile, in ordine di
        affidabilita': lo scenario di performance e' quello progettato per
        misurarle, gli altri le riportano come sottoprodotto.
        Restituisce (metriche_normalizzate, nome_scenario_di_provenienza).
        """
        scen = self._scenarios(run)

        perf = scen.get("performance_and_metrics")
        if isinstance(perf, dict):
            m = perf.get("model_accuracy_metrics")
            if isinstance(m, dict) and m:
                return self._normalize_metrics(m), "performance_and_metrics"

        scal = scen.get("scalability")
        if isinstance(scal, dict):
            per_scale = scal.get("metrics_per_scale")
            if isinstance(per_scale, dict):
                for key in sorted(per_scale, key=self._worker_count_of, reverse=True):
                    block = per_scale.get(key)
                    m = block.get("accuracy_metrics") if isinstance(block, dict) else None
                    if isinstance(m, dict) and m:
                        return self._normalize_metrics(m), f"scalability[{key}]"

        for name in ("network_simulation", "inference_worker_fault"):
            block = scen.get(name)
            if isinstance(block, dict):
                m = block.get("accuracy_metrics")
                if isinstance(m, dict) and m:
                    return self._normalize_metrics(m), name

        return {}, None

    @staticmethod
    def _worker_count_of(scale_key: str) -> int:
        """'workers_7' -> 7. Restituisce -1 se la chiave non e' nel formato atteso."""
        try:
            return int(str(scale_key).split("_")[-1])
        except (ValueError, IndexError):
            return -1

    def _scaling_series(self, run=None):
        """
        Estrae la serie ordinata dello strong scaling.
        Restituisce una lista di dict, uno per configurazione di worker, oppure
        [] se lo scenario manca o e' stato saltato.
        """
        scal = self._scenario("scalability", run)
        if not self._is_usable(scal):
            return []
        per_scale = scal.get("metrics_per_scale")
        if not isinstance(per_scale, dict) or not per_scale:
            return []

        series = []
        for key in sorted(per_scale, key=self._worker_count_of):
            n = self._worker_count_of(key)
            block = per_scale.get(key)
            if n <= 0 or not isinstance(block, dict):
                continue
            training = block.get("training") if isinstance(block.get("training"), dict) else {}
            inference = block.get("inference") if isinstance(block.get("inference"), dict) else {}
            total = self._get_float(training, "duration_seconds")
            if total is None:
                continue
            series.append({
                "workers": n,
                "total": total,
                # 'training_only_seconds' e' assente nei report piu' vecchi:
                # in quel caso resta None e le curve relative vengono omesse
                # invece di essere disegnate con un valore inventato.
                "train_only": self._get_float(training, "training_only_seconds"),
                "instrumented": bool(training.get("training_only_instrumented", True)),
                "etl": self._get_float(training, "etl_seconds", 0.0),
                "aggregation": self._get_float(training, "aggregation_seconds", 0.0),
                "oob": self._get_float(training, "oob_estimation_seconds", 0.0),
                "speedup": self._get_float(training, "speedup"),
                "speedup_train_only": self._get_float(training, "speedup_training_only"),
                "throughput": self._get_float(training, "throughput_trees_per_s"),
                "throughput_train_only": self._get_float(training, "throughput_trees_per_s_training_only"),
                "infer_duration": self._get_float(inference, "duration_seconds"),
                "infer_throughput": self._get_float(inference, "throughput_samples_per_s"),
                "infer_speedup": self._get_float(inference, "speedup"),
            })
        return series

    # =======================================================================
    # UTILITY DI DISEGNO
    # =======================================================================

    def _save(self, fig, filename: str):
        path = os.path.join(self.plots_dir, filename)
        try:
            fig.savefig(path)
            self.generated.append(path)
            print(f"[PLOT] OK  -> {path}")
        except OSError as e:
            print(f"[WARN] Salvataggio di '{path}' fallito ({e}).")
        finally:
            plt.close(fig)

    def _skip(self, plot_name: str, reason: str):
        self.skipped.append((plot_name, reason))
        print(f"[WARN] Dati mancanti per {plot_name}: {reason}. Salto il grafico.")

    @staticmethod
    def _titles(fig, ax, title: str, subtitle: str):
        """
        Titolo principale sopra la figura e sottotitolo (ambiente/modalita')
        sopra gli assi. Tenerli su due livelli distinti evita la sovrapposizione
        che si ottiene mettendo suptitle e ax.set_title alla stessa altezza.
        """
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        ax.set_title(subtitle, fontsize=10.5, color="#595959", fontweight="normal", pad=12)

    @staticmethod
    def _footnote(fig, text: str, y: float = -0.02):
        """Nota metodologica a pie' di figura: in relazione evita fraintendimenti."""
        fig.text(0.01, y, text, ha="left", va="top", fontsize=8.5,
                 color="#595959", style="italic", wrap=True)

    def _env_subtitle(self, run=None) -> str:
        """Ambiente (e modalita', se nota) della run mostrata nella figura."""
        env = (run or {}).get("env", self.primary_env)
        mode = (run or {}).get("mode", self.primary_mode)
        label = self.ENV_LABELS.get(env, str(env))
        if mode and mode != "unknown":
            label += f" - modalita' {mode}"
        return label

    @staticmethod
    def _provenance(run) -> str:
        """
        Riga di provenienza da mettere in nota: quale file ha prodotto i numeri
        della figura. In relazione e' cio' che rende il grafico verificabile.
        """
        if not run:
            return ""
        fp = run.get("fingerprint", {})
        extra = []
        if fp.get("trees"):
            extra.append("carico: " + "/".join(str(t) for t in fp["trees"]) + " alberi")
        if fp.get("test_sizes"):
            extra.append("test set: " + "/".join(f"{s:,}".replace(",", ".")
                                                 for s in fp["test_sizes"]) + " campioni")
        suffix = f" ({', '.join(extra)})" if extra else ""
        return f"Fonte: test_reports/{run['env']}/{run['file']}{suffix}."

    # =======================================================================
    # ORCHESTRAZIONE
    # =======================================================================

    def generate_all_plots(self):
        print("\n" + "=" * 66)
        print("        GENERAZIONE DEI GRAFICI PER LA RELAZIONE FINALE")
        print("=" * 66)

        jobs = [
            ("Distribuzione delle classi",        self.plot_class_distribution),
            ("Feature Importance",                self.plot_feature_importance),
            ("Matrice di confusione",             self.plot_confusion_matrix),
            ("Confronto metriche ML",             self.plot_ml_metrics_comparison),
            ("Curva di strong scaling",           self.plot_strong_scaling),
            ("Speedup ed efficienza",             self.plot_speedup_and_efficiency),
            ("Scomposizione dei tempi (Amdahl)",  self.plot_time_breakdown),
            ("Throughput",                        self.plot_throughput),
            ("Overhead di fault tolerance",       self.plot_fault_tolerance_overhead),
            ("Confronto fra ambienti",            self.plot_environment_comparison),
            ("Impatto della latenza di rete",     self.plot_network_impact),
        ]

        for name, fn in jobs:
            try:
                fn()
            except Exception as e:
                # Rete di sicurezza finale: un bug in un singolo generatore non
                # deve mai impedire la produzione di tutti gli altri grafici.
                print(f"[WARN] Errore inatteso nel grafico '{name}' "
                      f"({type(e).__name__}: {e}). Lo salto e proseguo.")
                traceback.print_exc()
                self.skipped.append((name, f"errore inatteso: {type(e).__name__}"))

        self._print_summary()
        return {"generated": list(self.generated), "skipped": list(self.skipped)}

    def _print_summary(self):
        print("\n" + "-" * 66)
        print(f"[PLOT] Grafici generati: {len(self.generated)} "
              f"(cartella '{os.path.abspath(self.plots_dir)}')")
        for path in self.generated:
            print(f"        + {os.path.basename(path)}")
        if self.skipped:
            print(f"[PLOT] Grafici saltati: {len(self.skipped)}")
            for name, reason in self.skipped:
                print(f"        - {name}: {reason}")
        print("-" * 66 + "\n")

    # =======================================================================
    # SEZIONE MACHINE LEARNING
    # =======================================================================

    def plot_class_distribution(self):
        """
        Distribuzione delle classi del test set, ricostruita dal 'support' del
        classification_report (o, in mancanza, dalle somme di riga della matrice
        di confusione).

        NOTA: si legge dai report invece che dal CSV di partenza perche' il
        dataset non e' necessariamente presente sulla macchina che genera i
        grafici (ad esempio su un task ECS one-off), mentre il support e' la
        stessa informazione gia' misurata sui dati reali.
        """
        name = "Distribuzione delle classi"
        run = self._pick_run(self.METRIC_SOURCES)
        metrics, source = self._find_accuracy_metrics(run)
        if not metrics:
            self._skip(name, "nessun blocco di metriche di classificazione nei report")
            return

        counts, labels = {}, {}
        report = metrics.get("classification_report")
        if isinstance(report, dict):
            for key, block in report.items():
                if key in ("accuracy", "macro avg", "weighted avg") or not isinstance(block, dict):
                    continue
                support = self._get_float(block, "support")
                if support is not None and support > 0:
                    counts[key] = support

        if not counts:
            cm = metrics.get("confusion_matrix")
            matrix = self._as_matrix(cm)
            if matrix is not None:
                for idx, row in enumerate(matrix):
                    counts[str(idx)] = float(np.sum(row))

        if not counts:
            self._skip(name, "ne' 'classification_report[*].support' ne' 'confusion_matrix' disponibili")
            return

        # Etichette leggibili per il caso binario tipico del dataset CICIDS.
        if set(counts.keys()) == {"0", "1"}:
            labels = {"0": "Classe 0 - Traffico benigno", "1": "Classe 1 - Attacco"}

        keys = sorted(counts, key=lambda k: counts[k], reverse=True)
        values = [counts[k] for k in keys]
        total = sum(values)
        names = [labels.get(k, f"Classe {k}") for k in keys]
        colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
                  PALETTE["success"]] * (len(keys) // 4 + 1)

        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        bars = ax.bar(names, values, color=colors[:len(keys)], width=0.55,
                      edgecolor="white", linewidth=1.2, zorder=3)

        for bar, value in zip(bars, values):
            pct = (value / total * 100) if total else 0.0
            ax.annotate(f"{int(value):,}\n({pct:.1f} %)".replace(",", "."),
                        xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                        color="#333333")

        ratio = (max(values) / min(values)) if min(values) > 0 else float("inf")
        ax.set_ylabel("Numero di campioni")
        ax.set_ylim(0, max(values) * 1.18)
        ax.set_axisbelow(True)
        ax.margins(x=0.2)
        self._titles(fig, ax, "Distribuzione delle classi nel test set",
                     f"{self._env_subtitle(run)} - sbilanciamento {ratio:.1f} : 1")
        self._footnote(fig, f"Conteggi ricavati dal campo 'support' di {source}. "
                            f"Lo sbilanciamento motiva l'uso di F1 e AUC accanto all'accuracy. "
                            f"{self._provenance(run)}")
        self._save(fig, "ml_01_distribuzione_classi.png")

    def plot_feature_importance(self, top_n: int = 15):
        """Top-N feature per importanza (Gini/impurity decrease) dalla baseline."""
        name = "Feature Importance"
        if not isinstance(self.baseline, dict):
            self._skip(name, f"'{self.baseline_pkl_path}' non caricato")
            return

        model = self.baseline.get("modello_addestrato")
        features = self.baseline.get("features_mappate")
        importances = getattr(model, "feature_importances_", None)

        if importances is None:
            self._skip(name, "'modello_addestrato' assente o privo di 'feature_importances_'")
            return
        importances = np.asarray(importances, dtype=float).ravel()

        if not isinstance(features, (list, tuple)) or len(features) != len(importances):
            print(f"[WARN] 'features_mappate' assente o di lunghezza incoerente "
                  f"({len(features) if isinstance(features, (list, tuple)) else 'n/d'} "
                  f"vs {len(importances)} importanze): uso nomi generici.")
            features = [f"Feature_{i}" for i in range(len(importances))]

        k = int(min(top_n, len(importances)))
        if k <= 0:
            self._skip(name, "vettore delle importanze vuoto")
            return

        order = np.argsort(importances)[::-1][:k][::-1]  # crescente per barh
        values = importances[order]
        names = [str(features[i]) for i in order]

        # Gradiente monocromatico: piu' scuro = piu' importante. Evita la
        # scala arcobaleno, sconsigliata in stampa e non accessibile.
        shades = sns.light_palette(PALETTE["primary"], n_colors=k + 3)[3:]

        fig, ax = plt.subplots(figsize=(9.4, max(5.0, 0.42 * k + 1.6)))
        bars = ax.barh(names, values, color=list(shades), height=0.72,
                       edgecolor="white", linewidth=0.8, zorder=3)

        span = float(values.max()) if values.size else 1.0
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.4f}",
                        xy=(value, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=9.5, color="#404040")

        cumulative = float(np.sum(values)) if values.size else 0.0
        ax.set_xlabel("Importanza (mean decrease in impurity)")
        ax.set_xlim(0, span * 1.16)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        hp = self.baseline.get("iperparametri_usati") or {}
        self._titles(fig, ax, f"Top {k} feature per importanza - modello baseline",
                     f"RandomForest {hp.get('tree_type', 'n/d')} - "
                     f"{hp.get('n_estimators', 'n/d')} alberi - "
                     f"le prime {k} feature pesano il {cumulative * 100:.1f} % del totale")
        self._footnote(fig, "L'importanza MDI e' distorta a favore delle feature ad alta cardinalita': "
                            "va letta come ordinamento indicativo, non come misura causale.")
        self._save(fig, "ml_02_feature_importance.png")

    @staticmethod
    def _as_matrix(cm):
        """Converte una matrice di confusione JSON in ndarray 2D, o None."""
        if not isinstance(cm, (list, tuple)) or not cm:
            return None
        try:
            matrix = np.asarray(cm, dtype=float)
        except (ValueError, TypeError):
            return None
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
            return None
        return matrix

    def plot_confusion_matrix(self):
        """
        Matrice di confusione del sistema distribuito, in due pannelli:
        conteggi assoluti e normalizzazione per riga (= recall per classe).

        La normalizzazione per riga e' indispensabile con classi sbilanciate:
        sui conteggi assoluti l'errore sulla classe minoritaria e' invisibile.
        """
        name = "Matrice di confusione"
        run = self._pick_run(self.METRIC_SOURCES)
        metrics, source = self._find_accuracy_metrics(run)
        if not metrics:
            self._skip(name, "nessun blocco di metriche nei report")
            return

        matrix = self._as_matrix(metrics.get("confusion_matrix"))
        if matrix is None:
            self._skip(name, "'confusion_matrix' assente o malformata "
                             "(probabile task di regressione)")
            return

        n = matrix.shape[0]
        labels = (["Benigno (0)", "Attacco (1)"] if n == 2
                  else [f"Classe {i}" for i in range(n)])

        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix),
                                   where=row_sums > 0)

        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))

        # Separatore delle migliaia con il punto (convenzione italiana):
        # le annotazioni vengono passate gia' formattate come stringhe.
        annot_abs = np.array([[f"{int(v):,}".replace(",", ".") for v in row] for row in matrix])
        sns.heatmap(matrix, annot=annot_abs, fmt="", cmap="Blues", cbar=False,
                    linewidths=1.4, linecolor="white",
                    xticklabels=labels, yticklabels=labels, ax=axes[0],
                    annot_kws={"fontsize": 12, "fontweight": "bold"})
        axes[0].set_title("Conteggi assoluti", fontsize=12.5)

        # square=False: con la colorbar solo sul pannello destro, imporre celle
        # quadrate darebbe ai due pannelli altezze diverse.
        sns.heatmap(normalized, annot=True, fmt=".3f", cmap="Blues", cbar=True,
                    vmin=0, vmax=1, linewidths=1.4, linecolor="white",
                    xticklabels=labels, yticklabels=labels, ax=axes[1],
                    annot_kws={"fontsize": 12, "fontweight": "bold"},
                    cbar_kws={"shrink": 0.75, "label": "Frazione della classe reale"})
        axes[1].set_title("Normalizzata per riga (recall per classe)", fontsize=12.5)

        for idx, ax in enumerate(axes):
            ax.set_xlabel("Classe predetta")
            # Solo il pannello di sinistra porta l'etichetta dell'asse Y: sul
            # destro finirebbe schiacciata fra le due matrici.
            ax.set_ylabel("Classe reale" if idx == 0 else "")
            ax.grid(False)
            ax.tick_params(labelrotation=0)

        acc = self._get_float(metrics, "accuracy")
        f1 = self._get_float(metrics, "f1_score")
        headline = "Matrice di confusione - inferenza distribuita"
        if acc is not None and f1 is not None:
            headline += f"   (accuracy {acc * 100:.2f} %, F1 {f1 * 100:.2f} %)"
        fig.suptitle(headline, fontsize=14, fontweight="bold", y=1.08)
        fig.text(0.5, 1.01, self._env_subtitle(run), ha="center",
                 fontsize=10.5, color="#595959")
        self._footnote(fig, f"Fonte: scenario '{source}'. Test set di "
                            f"{int(matrix.sum()):,} campioni.".replace(",", "."))
        self._save(fig, "ml_03_matrice_confusione.png")

    def plot_ml_metrics_comparison(self):
        """
        Baseline monolitica locale vs sistema distribuito, metrica per metrica.

        E' il grafico che sostiene la tesi centrale del progetto: distribuire
        l'addestramento di una Random Forest non degrada la qualita' del
        modello, perche' gli alberi sono indipendenti per costruzione.
        """
        name = "Confronto metriche ML"
        run = self._pick_run(self.METRIC_SOURCES)
        distributed, source = self._find_accuracy_metrics(run)
        if not distributed:
            self._skip(name, "metriche del sistema distribuito non disponibili")
            return
        if not isinstance(self.baseline, dict):
            self._skip(name, f"baseline '{self.baseline_pkl_path}' non caricata")
            return

        base = self._normalize_metrics(self.baseline.get("metriche_test") or {})
        if not base:
            self._skip(name, "'metriche_test' assente nel .pkl della baseline")
            return

        # Il set di metriche dipende dal task: classificazione o regressione.
        classification = ("accuracy", "precision", "recall", "f1_score", "auc")
        regression = ("r2", "mse", "rmse", "mae")
        candidates = classification if any(k in base for k in classification) else regression
        pretty = {
            "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
            "f1_score": "F1-score", "auc": "ROC-AUC",
            "r2": "R²", "mse": "MSE", "rmse": "RMSE", "mae": "MAE",
        }

        keys = [k for k in candidates
                if self._get_float(base, k) is not None
                and self._get_float(distributed, k) is not None]
        if not keys:
            self._skip(name, "nessuna metrica presente in entrambe le fonti "
                             "(baseline e distribuito potrebbero riferirsi a task diversi)")
            return

        is_classification = candidates is classification
        base_values = [self._get_float(base, k) for k in keys]
        dist_values = [self._get_float(distributed, k) for k in keys]

        legend_labels = ("Baseline monolitica (locale, n_jobs=1)",
                         "Sistema distribuito (Master-Worker RPC)")
        note = (f"Distribuito: scenario '{source}'. Stesso random_state e stessi iperparametri "
                f"della baseline: un Δ nullo e' il risultato atteso e dimostra la neutralita' "
                f"algoritmica della distribuzione. {self._provenance(run)}")

        if not is_classification:
            # Le metriche di regressione hanno ordini di grandezza diversi
            # (MSE ~ 10^2, R² ~ 1): su un asse unico R² sarebbe una barra
            # invisibile. Un pannello per metrica, ciascuno con la propria
            # scala, e' l'unica lettura corretta.
            fig, axes = plt.subplots(1, len(keys), figsize=(3.5 * len(keys) + 1.0, 5.2),
                                     squeeze=False)
            for ax, key, bv, dv in zip(axes[0], keys, base_values, dist_values):
                bars = ax.bar([0, 1], [bv, dv], width=0.58,
                              color=[PALETTE["neutral"], PALETTE["primary"]],
                              edgecolor="white", linewidth=1.1, zorder=3)
                for bar, v in zip(bars, (bv, dv)):
                    ax.annotate(f"{v:.4g}", xy=(bar.get_x() + bar.get_width() / 2, v),
                                xytext=(0, 4), textcoords="offset points", ha="center",
                                fontsize=10, color="#404040")
                delta = dv - bv
                ax.set_title(f"{pretty.get(key, key)}\nΔ {delta:+.3g}", fontsize=12,
                             color="#333333")
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Baseline", "Distribuito"], fontsize=9.5)
                ax.set_ylim(0, max(bv, dv) * 1.24)
                ax.set_axisbelow(True)
            handles = [Patch(facecolor=PALETTE["neutral"], edgecolor="white", label=legend_labels[0]),
                       Patch(facecolor=PALETTE["primary"], edgecolor="white", label=legend_labels[1])]
            fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.10))
            # Titoli sopra l'area dei pannelli (che hanno gia' un proprio
            # titolo su due righe): y > 1 con bbox_inches='tight' li tiene fuori.
            fig.suptitle("Qualita' del modello: baseline centralizzata vs sistema distribuito",
                         fontsize=14, fontweight="bold", y=1.16)
            fig.text(0.5, 1.09, self._env_subtitle(run), ha="center",
                     fontsize=10.5, color="#595959")
            self._footnote(fig, note, y=-0.15)
            self._save(fig, "ml_04_confronto_baseline_distribuito.png")
            return

        base_values = [v * 100 for v in base_values]
        dist_values = [v * 100 for v in dist_values]

        x = np.arange(len(keys), dtype=float)
        width = 0.36

        fig, ax = plt.subplots(figsize=(max(8.6, 1.9 * len(keys) + 3.0), 5.8))
        b1 = ax.bar(x - width / 2, base_values, width, label=legend_labels[0],
                    color=PALETTE["neutral"], edgecolor="white", linewidth=1.1, zorder=3)
        b2 = ax.bar(x + width / 2, dist_values, width, label=legend_labels[1],
                    color=PALETTE["primary"], edgecolor="white", linewidth=1.1, zorder=3)

        top = max(max(base_values), max(dist_values)) if keys else 1.0
        fmt = lambda v: f"{v:.2f}"
        for bars in (b1, b2):
            for bar in bars:
                ax.annotate(fmt(bar.get_height()),
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9.5, color="#404040")

        # Delta esplicito, in punti percentuali: e' il numero che il lettore cerca.
        for xi, bv, dv in zip(x, base_values, dist_values):
            delta = dv - bv
            ax.annotate(f"Δ {delta:+.3g} p.p.", xy=(xi, top * 1.09), ha="center",
                        fontsize=9.5, fontweight="bold",
                        color=PALETTE["success"] if abs(delta) < 1e-6 else PALETTE["secondary"])

        ax.set_xticks(x)
        ax.set_xticklabels([pretty.get(k, k) for k in keys])
        ax.set_ylabel("Valore della metrica (%)")
        ax.set_ylim(0, top * 1.18)
        # Legenda fuori dagli assi: con barre quasi a fondo scala qualunque
        # posizione interna finirebbe sopra i dati.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        ax.set_axisbelow(True)
        self._titles(fig, ax, "Qualita' del modello: baseline centralizzata vs sistema distribuito",
                     self._env_subtitle(run))
        self._footnote(fig, note, y=-0.16)
        self._save(fig, "ml_04_confronto_baseline_distribuito.png")

    # =======================================================================
    # SEZIONE SISTEMI DISTRIBUITI
    # =======================================================================

    def plot_strong_scaling(self):
        """Tempo di addestramento in funzione del numero di worker (strong scaling)."""
        name = "Curva di strong scaling"
        run = self._pick_run(["scalability"])
        series = self._scaling_series(run)
        if len(series) < 2:
            self._skip(name, "servono almeno due configurazioni di worker in "
                             "'scalability.metrics_per_scale'")
            return

        workers = np.array([p["workers"] for p in series], dtype=float)
        totals = np.array([p["total"] for p in series], dtype=float)

        fig, ax = plt.subplots(figsize=(9.6, 6.0))

        ax.plot(workers, totals, marker="o", markersize=8, linewidth=2.4,
                color=PALETTE["primary"], label="Tempo totale (ETL + alberi + aggregazione + OOB)",
                zorder=4)

        # Curva "soli alberi": solo se realmente strumentata. Se
        # 'training_only_instrumented' e' False il valore e' un placeholder pari
        # al totale (vedi ScalabilityScenario) e disegnarlo sarebbe fuorviante.
        only = [p["train_only"] for p in series]
        instrumented = all(p["instrumented"] for p in series)
        if instrumented and all(v is not None for v in only):
            ax.plot(workers, np.array(only, dtype=float), marker="s", markersize=7,
                    linewidth=2.4, color=PALETTE["secondary"],
                    label="Sola costruzione degli alberi (parte parallela)", zorder=4)
        else:
            print("[WARN] 'training_only_seconds' non strumentato (o assente) per almeno una "
                  "configurazione: ometto la curva della sola costruzione degli alberi.")

        # Curva ideale: T(N) = T(N_base) * N_base / N.
        ideal = totals[0] * workers[0] / workers
        ax.plot(workers, ideal, linestyle="--", linewidth=1.8, color=PALETTE["neutral"],
                label=f"Scaling ideale a partire da {int(workers[0])} worker", zorder=2)

        # Riferimenti della baseline locale: T_seq (monocore) e T_1node (multicore).
        tempi = (self.baseline or {}).get("baseline_tempi_locali") or {}
        reference_values = []
        for key, label, color in (
            ("t_seq", "Baseline monocore (T_seq)", PALETTE["accent"]),
            ("t_1node_parallel", "Baseline multicore singolo nodo (T_1node)", PALETTE["success"]),
        ):
            value = self._get_float(tempi, key)
            if value is not None and value > 0:
                reference_values.append(value)
                ax.axhline(value, linestyle=":", linewidth=1.7, color=color, zorder=1)
                # Etichetta ancorata a sinistra: a destra finirebbe sotto la legenda.
                ax.annotate(f"{label}: {value:.1f} s",
                            xy=(workers[0], value), xytext=(4, -5),
                            textcoords="offset points", ha="left", va="top",
                            fontsize=9, color=color, fontweight="bold")

        for xi, yi in zip(workers, totals):
            ax.annotate(f"{yi:.1f} s", xy=(xi, yi), xytext=(0, 10),
                        textcoords="offset points", ha="center",
                        fontsize=9.5, color=PALETTE["primary"])

        ax.set_xlabel("Numero di worker")
        ax.set_ylabel("Tempo di addestramento (secondi)")
        # set_xticks DOPO aver disegnato, e senza locator automatico: le
        # configurazioni testate sono 1,3,5,7 e non vanno interpolate con 2,4,6.
        ax.set_xticks(workers)
        ax.set_xticklabels([str(int(w)) for w in workers])
        ceiling = max([float(totals.max()), float(ideal.max())] + reference_values)
        ax.set_ylim(0, ceiling * 1.22)
        ax.legend(loc="upper right")
        ax.set_axisbelow(True)

        trees = self._trees_per_scale(run)
        subtitle = self._env_subtitle(run)
        if trees:
            subtitle += f" - carico fisso di {trees} alberi"
        self._titles(fig, ax, "Strong scaling: tempo di addestramento a carico costante", subtitle)
        self._footnote(fig, "Il divario fra curva misurata e curva ideale e' l'overhead "
                            "distribuito: parte seriale (ETL, aggregazione, OOB) piu' costo "
                            "di comunicazione RPC. " + self._provenance(run))
        self._save(fig, "sdcc_01_strong_scaling.png")

    def _trees_per_scale(self, run=None):
        """Numero di alberi dichiarato dallo scenario di scalabilita', se noto."""
        scal = self._scenario("scalability", run)
        if not isinstance(scal, dict):
            return None
        desc = str(scal.get("scenario_description", ""))
        for token in desc.replace(".", " ").split():
            if token.isdigit():
                return int(token)
        return None

    def plot_speedup_and_efficiency(self):
        """Speedup misurato vs ideale, con l'efficienza parallela sul pannello destro."""
        name = "Speedup ed efficienza"
        run = self._pick_run(["scalability"])
        series = self._scaling_series(run)
        if len(series) < 2:
            self._skip(name, "servono almeno due configurazioni di worker per calcolare lo speedup")
            return

        workers = np.array([p["workers"] for p in series], dtype=float)
        base_n = workers[0]

        # Speedup totale: sempre disponibile (ricalcolato se assente dal report).
        totals = np.array([p["total"] for p in series], dtype=float)
        speedup_total = np.array(
            [p["speedup"] if p["speedup"] is not None else (totals[0] / p["total"] if p["total"] > 0 else 1.0)
             for p in series], dtype=float)

        instrumented = all(p["instrumented"] for p in series)
        only = [p["train_only"] for p in series]
        speedup_only = None
        if instrumented and all(v is not None and v > 0 for v in only):
            declared = [p["speedup_train_only"] for p in series]
            speedup_only = np.array(
                [d if d is not None else (only[0] / v) for d, v in zip(declared, only)],
                dtype=float)

        ideal = workers / base_n

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))

        ax = axes[0]
        ax.plot(workers, ideal, linestyle="--", linewidth=1.9, color=PALETTE["neutral"],
                marker="", label="Speedup ideale (lineare)", zorder=2)
        ax.plot(workers, speedup_total, marker="o", markersize=8, linewidth=2.4,
                color=PALETTE["primary"], label="Speedup misurato (tempo totale)", zorder=4)
        if speedup_only is not None:
            ax.plot(workers, speedup_only, marker="s", markersize=7, linewidth=2.4,
                    color=PALETTE["secondary"], label="Speedup della sola parte parallela", zorder=4)
        ax.fill_between(workers, speedup_total, ideal, color=PALETTE["light"],
                        alpha=0.35, zorder=1, label="Perdita rispetto all'ideale")

        for xi, yi in zip(workers, speedup_total):
            ax.annotate(f"{yi:.2f}x", xy=(xi, yi), xytext=(0, -16),
                        textcoords="offset points", ha="center", fontsize=9.5,
                        color=PALETTE["primary"])

        ax.set_title("Speedup relativo", fontsize=13)
        ax.set_xlabel("Numero di worker")
        ax.set_ylabel(f"Speedup rispetto a {int(base_n)} worker")
        ax.set_xticks(workers)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left")
        ax.set_axisbelow(True)

        ax = axes[1]
        eff_source = speedup_only if speedup_only is not None else speedup_total
        efficiency = eff_source / ideal
        bars = ax.bar(workers, efficiency, width=0.55 if len(workers) > 2 else 0.35,
                      color=[PALETTE["success"] if e >= 0.75
                             else PALETTE["accent"] if e >= 0.5
                             else PALETTE["secondary"] for e in efficiency],
                      edgecolor="white", linewidth=1.1, zorder=3)
        ax.axhline(1.0, linestyle="--", linewidth=1.6, color=PALETTE["neutral"], zorder=2)
        ax.annotate("Efficienza ideale = 1.00", xy=(workers[-1], 1.0), xytext=(-4, 5),
                    textcoords="offset points", ha="right", fontsize=9,
                    color=PALETTE["neutral"], fontweight="bold")
        for bar, e in zip(bars, efficiency):
            ax.annotate(f"{e:.2f}", xy=(bar.get_x() + bar.get_width() / 2, e),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=10, fontweight="bold", color="#404040")

        which = "sola parte parallela" if speedup_only is not None else "tempo totale"
        ax.set_title(f"Efficienza parallela ({which})", fontsize=13)
        ax.set_xlabel("Numero di worker")
        ax.set_ylabel("Efficienza = Speedup / N")
        ax.set_xticks(workers)
        ax.set_ylim(0, max(1.15, float(efficiency.max()) * 1.15))
        ax.set_axisbelow(True)

        fig.suptitle(f"Scalabilita' del sistema distribuito - {self._env_subtitle(run)}",
                     fontsize=14, fontweight="bold", y=1.0)
        self._footnote(fig, "Lo speedup sul tempo totale e' limitato dalla frazione seriale "
                            "(legge di Amdahl): l'ETL e l'aggregazione non si accorciano "
                            "aggiungendo worker. L'efficienza misura quanto di ogni worker "
                            "aggiunto viene effettivamente convertito in lavoro utile. " + self._provenance(run))
        self._save(fig, "sdcc_02_speedup_efficienza.png")

    def plot_time_breakdown(self):
        """
        Scomposizione del tempo per configurazione di worker (barre impilate):
        e' la visualizzazione diretta della legge di Amdahl.
        """
        name = "Scomposizione dei tempi (Amdahl)"
        run = self._pick_run(["scalability", "performance_and_metrics"])
        series = self._scaling_series(run)
        phases = ("etl_seconds", "training_only_seconds", "aggregation_seconds",
                  "oob_estimation_seconds", "unaccounted_seconds")

        rows, labels = [], []
        if len(series) >= 2 and all(p["instrumented"] and p["train_only"] is not None for p in series):
            for p in series:
                accounted = (p["etl"] or 0) + (p["train_only"] or 0) + (p["aggregation"] or 0) + (p["oob"] or 0)
                rows.append({
                    "etl_seconds": p["etl"] or 0.0,
                    "training_only_seconds": p["train_only"] or 0.0,
                    "aggregation_seconds": p["aggregation"] or 0.0,
                    "oob_estimation_seconds": p["oob"] or 0.0,
                    "unaccounted_seconds": max(0.0, p["total"] - accounted),
                })
                labels.append(f"{p['workers']} worker")
            xlabel = "Configurazione di worker"
        else:
            # Fallback: lo scenario di performance espone comunque un
            # 'timing_breakdown' per una singola configurazione.
            perf = self._scenario("performance_and_metrics", run)
            timing = perf.get("timing_breakdown") if isinstance(perf, dict) else None
            if not isinstance(timing, dict):
                self._skip(name, "ne' 'scalability' con tempi strumentati ne' "
                                 "'performance_and_metrics.timing_breakdown' disponibili")
                return
            row = {ph: self._get_float(timing, ph, 0.0) or 0.0 for ph in phases}
            if sum(row.values()) <= 0:
                self._skip(name, "'timing_breakdown' presente ma tutto a zero "
                                 "(orchestratore non strumentato in questa modalita')")
                return
            rows.append(row)
            labels.append("Configurazione corrente")
            xlabel = ""

        totals = [sum(r.values()) for r in rows]
        if not any(t > 0 for t in totals):
            self._skip(name, "tempi di fase tutti nulli")
            return

        x = np.arange(len(rows), dtype=float)
        width = 0.5 if len(rows) > 2 else 0.34

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))

        for ax, normalize in zip(axes, (False, True)):
            bottom = np.zeros(len(rows))
            for phase in phases:
                values = np.array([r.get(phase, 0.0) for r in rows], dtype=float)
                if normalize:
                    values = np.divide(values, np.array(totals), out=np.zeros_like(values),
                                       where=np.array(totals) > 0) * 100
                if not np.any(values > 0):
                    continue
                ax.bar(x, values, width, bottom=bottom, color=PHASE_COLORS[phase],
                       edgecolor="white", linewidth=0.9, zorder=3,
                       label=PHASE_LABELS[phase] if not normalize else None)
                bottom += values

            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel(xlabel)
            ax.set_axisbelow(True)
            if normalize:
                ax.set_ylabel("Quota sul tempo totale (%)")
                ax.set_title("Composizione percentuale", fontsize=13)
                ax.set_ylim(0, 100)
            else:
                ax.set_ylabel("Tempo (secondi)")
                ax.set_title("Tempo assoluto per fase", fontsize=13)
                for xi, total in zip(x, totals):
                    ax.annotate(f"{total:.1f} s", xy=(xi, total), xytext=(0, 5),
                                textcoords="offset points", ha="center",
                                fontsize=9.5, fontweight="bold", color="#404040")

        handles = [Patch(facecolor=PHASE_COLORS[p], edgecolor="white", label=PHASE_LABELS[p])
                   for p in phases]
        fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.14),
                   frameon=True)
        fig.suptitle(f"Scomposizione del tempo di addestramento - {self._env_subtitle(run)}",
                     fontsize=14, fontweight="bold", y=1.0)
        self._footnote(fig, "Solo la fascia blu si riduce all'aumentare dei worker: le fasi "
                            "seriali restano pressoche' costanti e, crescendo in quota "
                            "relativa (pannello destro), fissano il tetto di Amdahl allo "
                            "speedup ottenibile. " + self._provenance(run))
        self._save(fig, "sdcc_03_scomposizione_tempi.png")

    def plot_throughput(self):
        """Throughput di addestramento (alberi/s) e di inferenza (campioni/s)."""
        name = "Throughput"
        run = self._pick_run(["scalability"])
        series = self._scaling_series(run)
        if not series:
            self._skip(name, "scenario 'scalability' assente o saltato")
            return

        workers = np.array([p["workers"] for p in series], dtype=float)
        train = [p["throughput"] for p in series]
        infer = [p["infer_throughput"] for p in series]

        has_train = any(v is not None and v > 0 for v in train)
        has_infer = any(v is not None and v > 0 for v in infer)
        if not has_train and not has_infer:
            self._skip(name, "nessun valore di throughput valido nei report")
            return

        n_panels = int(has_train) + int(has_infer)
        fig, axes = plt.subplots(1, n_panels, figsize=(7.0 * n_panels, 5.4), squeeze=False)
        axes = list(axes[0])

        if has_train:
            ax = axes.pop(0)
            values = np.array([v if v is not None else 0.0 for v in train], dtype=float)
            bars = ax.bar(workers, values, width=0.55 if len(workers) > 2 else 0.35,
                          color=PALETTE["primary"], edgecolor="white", linewidth=1.1, zorder=3)
            only = [p["throughput_train_only"] for p in series]
            if all(p["instrumented"] for p in series) and any(v for v in only if v):
                ax.plot(workers, [v or 0.0 for v in only], marker="D", markersize=7,
                        linewidth=2.2, color=PALETTE["secondary"], zorder=4,
                        label="Soli alberi (al netto dell'overhead)")
                ax.legend(loc="upper left")
            for bar, v in zip(bars, values):
                ax.annotate(f"{v:.2f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                            xytext=(0, 4), textcoords="offset points", ha="center",
                            fontsize=10, color="#404040")
            ax.set_title("Addestramento", fontsize=13)
            ax.set_xlabel("Numero di worker")
            ax.set_ylabel("Throughput (alberi / secondo)")
            ax.set_xticks(workers)
            ax.set_axisbelow(True)
            ceiling = max([float(values.max())] + [v for v in only if v])
            ax.set_ylim(0, ceiling * 1.28)   # spazio per legenda ed etichette

        if has_infer:
            ax = axes.pop(0)
            values = np.array([v if v is not None else 0.0 for v in infer], dtype=float)
            bars = ax.bar(workers, values, width=0.55 if len(workers) > 2 else 0.35,
                          color=PALETTE["success"], edgecolor="white", linewidth=1.1, zorder=3)
            for bar, v in zip(bars, values):
                ax.annotate(f"{v:,.0f}".replace(",", "."),
                            xy=(bar.get_x() + bar.get_width() / 2, v),
                            xytext=(0, 4), textcoords="offset points", ha="center",
                            fontsize=10, color="#404040")
            ax.set_title("Inferenza", fontsize=13)
            ax.set_xlabel("Numero di worker")
            ax.set_ylabel("Throughput (campioni / secondo)")
            ax.set_xticks(workers)
            ax.set_axisbelow(True)
            ax.set_ylim(0, float(values.max()) * 1.15)

        fig.suptitle(f"Throughput del sistema distribuito - {self._env_subtitle(run)}",
                     fontsize=14, fontweight="bold", y=1.0)

        # Il caveat sul federato e' scritto dallo scenario stesso: se c'e', va
        # riportato, perche' cambia il modo in cui il grafico va letto.
        scal = self._scenario("scalability", run) or {}
        caveat = scal.get("inference_speedup_caveat")
        note = ("Il throughput e' la misura corretta anche in modalita' federata, dove il "
                "test set totale cresce con il numero di worker e lo speedup dell'inferenza "
                "non sarebbe confrontabile."
                if caveat else
                "Throughput misurato a carico fisso: cresce con i worker fintanto che la "
                "parte parallela domina il tempo totale.")
        self._footnote(fig, note + " " + self._provenance(run))
        self._save(fig, "sdcc_04_throughput.png")

    def plot_fault_tolerance_overhead(self):
        """
        Costo della tolleranza ai guasti: job pulito vs crash del worker vs
        failover dell'orchestratore.

        I tre scenari possono aver addestrato un numero DIVERSO di alberi
        (dipende dalla configurazione con cui sono stati lanciati): confrontare
        le durate assolute sarebbe scorretto, quindi si normalizza a
        secondi/albero e si annota comunque il dato grezzo.
        """
        name = "Overhead di fault tolerance"
        run = self._pick_run([
            "performance_and_metrics", "fault_tolerance", "orchestrator_failover",
            "inference_worker_fault", "inference_orchestrator_failover", "scalability",
        ], prefer="coverage")
        scen = self._scenarios(run)

        train_specs = [
            ("performance_and_metrics", "Job pulito\n(nessun guasto)", PALETTE["success"]),
            ("fault_tolerance",         "Crash Worker\n(training)", PALETTE["accent"]),
            ("orchestrator_failover",   "Failover Orchestratore\n(training)", PALETTE["secondary"]),
        ]
        train_rows = []
        for key, label, color in train_specs:
            block = scen.get(key)
            if not self._is_usable(block):
                print(f"[WARN] Scenario '{key}' assente o saltato: escluso dal confronto di fault tolerance.")
                continue
            duration = self._get_float(block, "duration_seconds")
            trees = self._get_float(block, "trees_built")
            if duration is None or duration <= 0 or trees is None or trees <= 0:
                print(f"[WARN] Scenario '{key}' privo di 'duration_seconds'/'trees_built' validi: escluso.")
                continue
            train_rows.append({"label": label, "color": color, "value": duration / trees,
                               "raw": duration, "trees": int(trees)})

        infer_specs = [
            ("inference_worker_fault",           "Crash Worker\n(inferenza)", PALETTE["accent"]),
            ("inference_orchestrator_failover",  "Failover Orchestratore\n(inferenza)", PALETTE["secondary"]),
        ]
        infer_rows = []
        # Riferimento pulito per l'inferenza: la configurazione di scalabilita'
        # con piu' worker, che e' l'unica a cronometrare l'inferenza da sola.
        series = self._scaling_series(run)
        clean_infer = next((p["infer_duration"] for p in reversed(series)
                            if p["infer_duration"] is not None and p["infer_duration"] > 0), None)
        if clean_infer is not None:
            infer_rows.append({"label": "Inferenza pulita\n(nessun guasto)",
                               "color": PALETTE["success"], "value": clean_infer, "raw": clean_infer})
        for key, label, color in infer_specs:
            block = scen.get(key)
            if not self._is_usable(block):
                print(f"[WARN] Scenario '{key}' assente o saltato: escluso dal confronto di fault tolerance.")
                continue
            duration = self._get_float(block, "duration_seconds")
            if duration is None or duration <= 0:
                continue
            infer_rows.append({"label": label, "color": color, "value": duration, "raw": duration})

        if len(train_rows) < 2 and len(infer_rows) < 2:
            self._skip(name, "servono almeno due scenari confrontabili "
                             "(job pulito + almeno un guasto) per training o inferenza")
            return

        panels = [p for p in (("train", train_rows), ("infer", infer_rows)) if len(p[1]) >= 2]
        fig, axes = plt.subplots(1, len(panels), figsize=(7.4 * len(panels), 5.8), squeeze=False)
        axes = list(axes[0])

        for (kind, rows), ax in zip(panels, axes):
            labels = [r["label"] for r in rows]
            values = [r["value"] for r in rows]
            bars = ax.bar(labels, values, width=0.5,
                          color=[r["color"] for r in rows],
                          edgecolor="white", linewidth=1.2, zorder=3)

            reference = values[0] if values else 0.0
            for bar, row in zip(bars, rows):
                caption = (f"{row['value']:.2f} s/albero\n"
                           f"({row['raw']:.1f} s su {row['trees']} alberi)"
                           if kind == "train" else f"{row['raw']:.2f} s")
                if reference > 0 and row["value"] != reference:
                    caption += f"\n{(row['value'] / reference - 1) * 100:+.1f} % vs pulito"
                ax.annotate(caption, xy=(bar.get_x() + bar.get_width() / 2, row["value"]),
                            xytext=(0, 6), textcoords="offset points", ha="center",
                            va="bottom", fontsize=9.5, color="#404040")

            if reference > 0:
                ax.axhline(reference, linestyle="--", linewidth=1.5,
                           color=PALETTE["neutral"], zorder=2)
            ax.set_ylabel("Tempo normalizzato (s / albero)" if kind == "train"
                          else "Tempo di inferenza (s)")
            ax.set_title("Addestramento" if kind == "train" else "Inferenza", fontsize=13)
            ax.set_ylim(0, max(values) * 1.42)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelsize=9.5)
            # Le etichette multi-riga sopra le barre sono larghe: senza margine
            # orizzontale quella della prima barra viene tagliata dall'asse.
            ax.margins(x=0.16)

        fig.suptitle(f"Costo della tolleranza ai guasti - {self._env_subtitle(run)}",
                     fontsize=14, fontweight="bold", y=1.0)
        self._footnote(fig, "I tempi di addestramento sono normalizzati per albero perche' i tre "
                            "scenari possono essere stati eseguiti con carichi diversi. "
                            "La differenza rispetto al job pulito e' il costo di rilevazione del "
                            "guasto piu' la ridistribuzione del lavoro perso. " + self._provenance(run))
        self._save(fig, "sdcc_05_overhead_fault_tolerance.png")

    # =======================================================================
    # CONFRONTI TRASVERSALI
    # =======================================================================

    def plot_environment_comparison(self):
        """Stesso scenario, ambienti diversi: bare-metal vs Docker vs AWS ECS."""
        name = "Confronto fra ambienti"
        rows = []
        for env in self.ENV_PRIORITY:
            # Una sola run per ambiente: quella piu' completa/recente che
            # contenga lo scenario di performance.
            candidates = [r for r in self.runs if r["env"] == env
                          and self._is_usable(r["scenarios"].get("performance_and_metrics"))]
            if not candidates:
                continue
            run = max(candidates, key=lambda r: (len(r["scenarios"]), r["mtime"]))
            perf = self._scenario("performance_and_metrics", run)
            throughput = self._get_float(perf, "throughput_trees_per_sec")
            duration = self._get_float(perf, "duration_seconds")
            metrics, _ = self._find_accuracy_metrics(run)
            accuracy = self._get_float(metrics, "accuracy")
            if accuracy is None:
                accuracy = self._get_float(metrics, "r2")
            if throughput is None and duration is None:
                continue
            rows.append({"env": env, "label": self.ENV_LABELS.get(env, env),
                         "throughput": throughput, "duration": duration,
                         "accuracy": accuracy,
                         "trees": self._get_float(perf, "trees_built"),
                         "run": run})

        if len(rows) < 2:
            self._skip(name, "servono report di 'performance_and_metrics' in almeno "
                             "due ambienti fra aws/docker/local")
            return

        # Confrontare tempi ASSOLUTI misurati su carichi diversi non significa
        # nulla: se i tre ambienti hanno addestrato un numero di alberi diverso
        # si mostra solo il throughput, che e' per definizione normalizzato.
        loads = {int(r["trees"]) for r in rows if r["trees"]}
        comparable_durations = len(loads) <= 1
        if not comparable_durations:
            print(f"[WARN] Gli ambienti confrontati hanno carichi diversi ({sorted(loads)} alberi): "
                  f"ometto il pannello dei tempi assoluti e mostro solo il throughput, "
                  f"che e' indipendente dal carico.")

        labels = [r["label"] for r in rows]
        colors = [PALETTE["primary"], PALETTE["accent"], PALETTE["success"]] * 2

        show_accuracy = any(r["accuracy"] is not None for r in rows)
        n_panels = int(comparable_durations) + 1 + int(show_accuracy)
        fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 5.4), squeeze=False)
        axes = list(axes[0])

        panels = []
        if comparable_durations:
            panels.append(("duration", "Tempo totale di addestramento", "Secondi",
                           lambda v: f"{v:.1f} s"))
        panels.append(("throughput", "Throughput di addestramento", "Alberi / secondo",
                       lambda v: f"{v:.3f}"))
        if show_accuracy:
            panels.append(("accuracy", "Qualita' del modello", "Accuracy / R²", lambda v: f"{v:.4f}"))

        for (key, title, ylabel, fmt), ax in zip(panels, axes):
            values = [r.get(key) for r in rows]
            usable = [(l, v, c) for l, v, c in zip(labels, values, colors) if v is not None]
            if not usable:
                ax.axis("off")
                continue
            u_labels, u_values, u_colors = zip(*usable)
            bars = ax.bar(u_labels, u_values, width=0.5, color=list(u_colors),
                          edgecolor="white", linewidth=1.2, zorder=3)
            for bar, v in zip(bars, u_values):
                ax.annotate(fmt(v), xy=(bar.get_x() + bar.get_width() / 2, v),
                            xytext=(0, 5), textcoords="offset points", ha="center",
                            fontsize=10, fontweight="bold", color="#404040")
            ax.set_title(title, fontsize=12.5)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(u_values) * 1.22)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelrotation=14, labelsize=9.5)
            if key == "accuracy":
                # Scala 0-1 fissa (piu' spazio per le etichette): un asse
                # troncato esagererebbe visivamente differenze irrilevanti.
                ax.set_ylim(0, 1.12)

        fig.suptitle("Ambiente di esecuzione: locale, containerizzato, cloud",
                     fontsize=14, fontweight="bold", y=1.0)
        self._footnote(fig, "I tempi confrontano hardware diverso (CPU locale vs vCPU Fargate) e "
                            "vanno letti come caratterizzazione dell'ambiente, non come merito "
                            "dell'architettura. Le metriche di qualita', invece, devono coincidere: "
                            "un loro scostamento segnalerebbe un bug, non un effetto dell'ambiente.")
        self._save(fig, "cmp_01_ambienti.png")

    def plot_network_impact(self):
        """
        Impatto della latenza di rete iniettata con tc netem sulle chiamate RPC.

        Su AWS Fargate lo scenario riporta 'SKIPPED_NO_TC_PERMISSIONS'
        (CAP_NET_ADMIN non disponibile): in quel caso non si inventa una barra,
        si dichiara la limitazione nel grafico stesso.
        """
        name = "Impatto della latenza di rete"
        # Il confronto "con e senza latenza" ha senso SOLO se i due termini
        # vengono dalla stessa sessione di test: stesso dataset, stesso numero
        # di alberi, stessa macchina. Prendere il job degradato da un file e il
        # riferimento pulito da un altro e' il modo piu' rapido per ottenere il
        # risultato assurdo "con 1500 ms di latenza il sistema va piu' veloce".
        candidates = [
            r for r in self.runs
            if (r["scenarios"].get("network_simulation") or {}).get("tc_rules_successfully_injected")
            and self._is_usable(r["scenarios"].get("performance_and_metrics"))
        ]
        if not candidates:
            any_net = any(isinstance(r["scenarios"].get("network_simulation"), dict) for r in self.runs)
            if not any_net:
                self._skip(name, "scenario 'network_simulation' assente")
            else:
                self._skip(name, "nessuna singola sessione di test contiene sia lo scenario di rete "
                                 "con regole tc effettivamente iniettate sia il job di riferimento "
                                 "senza latenza (lanciali insieme con SCENARIO=all)")
            return

        run = max(candidates, key=lambda r: (-self.ENV_PRIORITY.index(r["env"]),
                                             len(r["scenarios"]), r["mtime"]))
        net = self._scenario("network_simulation", run)
        perf = self._scenario("performance_and_metrics", run)

        latency = self._get_float(net, "applied_latency_ms", 0.0) or 0.0
        loss = self._get_float(net, "applied_loss_percent", 0.0) or 0.0
        net_throughput = self._get_float(net, "throughput_trees_per_second")
        net_duration = self._get_float(net, "duration_seconds")
        base_throughput = self._get_float(perf, "throughput_trees_per_sec")
        base_duration = self._get_float(perf, "duration_seconds")

        if (net_throughput is None and net_duration is None) or \
           (base_throughput is None and base_duration is None):
            self._skip(name, "durate/throughput non disponibili in entrambi i termini del confronto")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
        labels = ["Rete non degradata", f"+{latency:.0f} ms di latenza\n{loss:.1f} % di perdita"]

        for ax, (base, degraded, title, ylabel, fmt) in zip(axes, [
            (base_duration, net_duration, "Tempo totale di addestramento", "Secondi", "{:.1f} s"),
            (base_throughput, net_throughput, "Throughput di addestramento", "Alberi / secondo", "{:.3f}"),
        ]):
            if base is None or degraded is None:
                ax.axis("off")
                continue
            values = [base, degraded]
            bars = ax.bar(labels, values, width=0.48,
                          color=[PALETTE["success"], PALETTE["secondary"]],
                          edgecolor="white", linewidth=1.2, zorder=3)
            for bar, v in zip(bars, values):
                ax.annotate(fmt.format(v), xy=(bar.get_x() + bar.get_width() / 2, v),
                            xytext=(0, 5), textcoords="offset points", ha="center",
                            fontsize=10.5, fontweight="bold", color="#404040")
            if base > 0:
                ax.annotate(f"{(degraded / base - 1) * 100:+.1f} %",
                            xy=(1, degraded), xytext=(0, 26), textcoords="offset points",
                            ha="center", fontsize=11, fontweight="bold",
                            color=PALETTE["secondary"])
            ax.set_title(title, fontsize=13)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(values) * 1.30)
            ax.set_axisbelow(True)

        fig.suptitle(f"Sensibilita' alla latenza di rete - {self._env_subtitle(run)}",
                     fontsize=14, fontweight="bold", y=1.0)
        self._footnote(fig, "Latenza iniettata con tc netem sull'interfaccia dei container. "
                            "La degradazione misura quanto il protocollo RPC sincrono e' "
                            "sensibile al RTT: piu' chiamate per job, piu' il ritardo si "
                            "accumula sul percorso critico. " + self._provenance(run))
        self._save(fig, "cmp_02_latenza_rete.png")


if __name__ == "__main__":
    PlotGenerator().generate_all_plots()