"""utils.py — пути, атомарная запись JSON, статус обучения, хелперы Ollama."""
import json
import os
import re
import tempfile
import threading
import time
import urllib.request

os.environ.setdefault("LLAMA_CPP_DIR", r"C:\tools\llama.cpp")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
AGENTS_FILE = os.path.join(DATA_DIR, "agents.json")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
STATUS_FILE = os.path.join(DATA_DIR, "training_status.json")
STOP_FLAG = os.path.join(DATA_DIR, "stop.flag")
WORKSPACE_DIR = os.path.join(DATA_DIR, "workspace")
ADAPTERS_DIR = os.path.join(DATA_DIR, "lora_adapters")
MERGED_DIR = os.path.join(DATA_DIR, "merged")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
JOBS_DIR = os.path.join(DATA_DIR, "jobs")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

_lock = threading.RLock()


def ensure_data_dir():
    for d in (DATA_DIR, WORKSPACE_DIR, ADAPTERS_DIR, MERGED_DIR, DATASETS_DIR, JOBS_DIR, LOGS_DIR):
        os.makedirs(d, exist_ok=True)


def atomic_write_json(path, data):
    """tempfile + os.replace; на Windows повторяем, если файл на миг занят читателем."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def reset_status(**fields):
    with _lock:
        atomic_write_json(STATUS_FILE, fields)


def update_status(**fields):
    with _lock:
        data = read_json(STATUS_FILE, {}) or {}
        data.update(fields)
        atomic_write_json(STATUS_FILE, data)


def slugify_ollama(name):
    """Ollama принимает только a-z0-9.- (без '_' и кириллицы)."""
    s = re.sub(r"[^a-z0-9.-]+", "-", (name or "").lower()).strip("-.")
    return (s or "agent")[:40]


def to_posix(path):
    """Абсолютный путь с прямыми слэшами (нужно для Modelfile / llama.cpp)."""
    return os.path.abspath(path).replace("\\", "/")


def ollama_request(path, payload=None, timeout=30):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        OLLAMA_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8")
    return json.loads(body) if body.strip() else {}


def ollama_models():
    try:
        return [m["name"] for m in ollama_request("/api/tags", timeout=5).get("models", [])]
    except Exception:
        return []


def ollama_unload_all():
    """Выгружает модели из VRAM — обучению нужны все 8 ГБ."""
    try:
        for m in ollama_request("/api/ps", timeout=5).get("models", []):
            try:
                ollama_request("/api/generate", {"model": m["name"], "keep_alive": 0}, timeout=30)
            except Exception:
                pass
        time.sleep(1.5)
    except Exception:
        pass


# ----------------------------------------------------------------- прочее
def stop_requested():
    return os.path.exists(STOP_FLAG)


def clear_stop_flag():
    try:
        os.remove(STOP_FLAG)
    except OSError:
        pass


def estimate_params_b(model_id):
    """Размер модели в млрд параметров по имени ('Qwen2.5-7B' -> 7.0); None если не видно."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model_id or "")
    return float(m.group(1)) if m else None


def required_disk_gb(model_id):
    """Грубая оценка места: merged bf16 (2 Б/параметр) + GGUF (до 2 Б/параметр) + запас."""
    p = estimate_params_b(model_id) or 3.0
    return round(p * 4 + 2, 1)


def tail_lines(path, n=15):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [ln.rstrip() for ln in f.readlines()[-n:]]
    except Exception:
        return []


def now_ts():
    return int(time.time())
