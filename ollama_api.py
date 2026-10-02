"""ollama_api.py — тонкая обёртка над HTTP API Ollama (только stdlib)."""
import json
import urllib.error
import urllib.request

from utils import OLLAMA_URL


def _request(path, payload=None, timeout=30, method=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        OLLAMA_URL + path,
        data=data,
        method=method or ("POST" if data is not None else "GET"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama недоступна ({OLLAMA_URL}): {e.reason}")
    return json.loads(body) if body.strip() else {}


def version():
    try:
        return _request("/api/version", timeout=3).get("version")
    except Exception:
        return None


def list_models():
    try:
        tags = _request("/api/tags", timeout=5).get("models", [])
    except Exception:
        return []
    return [{"name": m.get("name") or m.get("model"), "size": m.get("size", 0)} for m in tags]


def _norm(name):
    return name if ":" in name else name + ":latest"


def has_model(name):
    want = _norm(name or "")
    return any(_norm(m["name"]) == want for m in list_models())


def chat(model, messages, temperature=0.7, json_mode=False, num_ctx=4096,
         num_predict=None, seed=None, timeout=600):
    options = {"temperature": temperature, "num_ctx": num_ctx}
    if num_predict:
        options["num_predict"] = num_predict
    if seed is not None:
        options["seed"] = seed
    payload = {"model": model, "messages": messages, "stream": False, "options": options}
    if json_mode:
        payload["format"] = "json"
    data = _request("/api/chat", payload, timeout=timeout)
    return (data.get("message") or {}).get("content", "")


def unload(model):
    """Выгрузить модель из VRAM (keep_alive=0)."""
    try:
        _request("/api/generate", {"model": model, "keep_alive": 0}, timeout=30)
    except Exception:
        pass


def remove_model(name):
    try:
        _request("/api/delete", {"model": name}, timeout=30, method="DELETE")
        return True
    except Exception:
        return False
