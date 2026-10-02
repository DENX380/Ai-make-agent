"""
main.py — FastAPI-сервер хаба агентов (http://127.0.0.1:8000).
Запуск: python main.py
Обучение выполняется отдельным процессом qlora_trainer.py; здесь — управление, статус, шаг 6 (проверка).
"""
import importlib.metadata as md
import os
import shutil
import subprocess
import sys
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agent_core
import dataset_gen
import merge_lora
import ollama_api
import train_control
from utils import (ADAPTERS_DIR, BASE_DIR, DATA_DIR, DATASETS_DIR, JOBS_DIR, LOGS_DIR, STATUS_FILE, WORKSPACE_DIR,
                   clear_stop_flag, ensure_data_dir, now_ts, read_json, required_disk_gb, reset_status, tail_lines,
                   update_status)

PRESETS = [
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "label": "Qwen2.5 0.5B — быстрый тест (~2 ГБ VRAM)"},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "label": "Qwen2.5 1.5B — рекомендуется (~4 ГБ)"},
    {"id": "Qwen/Qwen2.5-3B-Instruct", "label": "Qwen2.5 3B — качество лучше (~6 ГБ)"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "label": "Qwen2.5 7B — впритык по VRAM, merge требует ~16 ГБ RAM"},
    {"id": "meta-llama/Llama-3.2-1B-Instruct", "label": "Llama 3.2 1B — нужен доступ на Hugging Face"},
]
STEP_NAMES = ["Датасет", "Обучение QLoRA", "Merge LoRA", "Конвертация GGUF", "Регистрация в Ollama", "Проверка"]
TRAIN = {"proc": None, "agent_id": None}
TRAIN_LOCK = threading.Lock()


def training_alive():
    p = TRAIN["proc"]
    return p is not None and p.poll() is None


# ---------------------------------------------------------------- запуск обучения
def start_agent_training(p, dataset_bytes=None):
    """Создаёт агента и запускает QLoRA. p — словарь параметров. Бросает HTTPException."""
    name = (p.get("name") or "").strip()[:60]
    prompt = (p.get("prompt") or "").strip()
    base_model = (p.get("base_model") or "").strip()
    access = p.get("access", "read-only")
    if not name or len(prompt) < 10:
        raise HTTPException(400, "Укажите имя и системный промпт (не короче 10 символов)")
    if access not in agent_core.ACCESS_LEVELS:
        raise HTTPException(400, "Неверный уровень доступа")
    if not merge_lora.detect_family(base_model):
        raise HTTPException(400, "Поддерживаются модели Qwen, Llama-3 и Gemma (Hugging Face id)")
    if not merge_lora.find_convert_script():
        raise HTTPException(400, f"Не найден convert_hf_to_gguf.py в {merge_lora.llama_dir()}")
    if not shutil.which("ollama"):
        raise HTTPException(400, "Команда ollama не найдена в PATH")
    need = required_disk_gb(base_model)
    free = shutil.disk_usage(DATA_DIR).free / 1024 ** 3
    if free < need:
        raise HTTPException(400, f"Мало места на диске: нужно ~{need} ГБ, свободно {free:.1f} ГБ")

    installed = [m["name"] for m in ollama_api.list_models()]
    gen_model = (p.get("gen_model") or (installed[0] if installed else "")).strip()
    if not dataset_bytes:
        if not gen_model or gen_model not in installed:
            raise HTTPException(400, "Выберите модель Ollama для генерации датасета (или загрузите JSONL)")

    with TRAIN_LOCK:
        if training_alive():
            raise HTTPException(409, "Уже идёт обучение другого агента")
        aid = agent_core.new_agent_id()
        dataset_path = ""
        if dataset_bytes:
            dataset_path = os.path.join(DATASETS_DIR, f"{aid}_upload.jsonl")
            with open(dataset_path, "wb") as f:
                f.write(dataset_bytes)
            if len(dataset_gen.load_jsonl(dataset_path)) < 5:
                os.remove(dataset_path)
                raise HTTPException(400, "В файле меньше 5 корректных примеров (поля instruction/response)")
        clamp = lambda v, lo, hi: max(lo, min(hi, int(v)))
        agent = {
            "id": aid, "name": name, "prompt": prompt, "access": access, "base_model": base_model,
            "model": gen_model, "fallback_model": gen_model, "trained": False, "train_state": "training",
            "can_create_agents": False, "allowed_apps": [], "created": now_ts(),
            "training": {
                "gen_model": gen_model, "compare_model": (p.get("compare_model") or gen_model),
                "topic": (p.get("topic") or "").strip(), "dataset_path": dataset_path,
                "num_examples": clamp(p.get("num_examples", 100), 20, 1000),
                "epochs": clamp(p.get("epochs", 3), 1, 10),
                "max_seq_len": clamp(p.get("max_seq_len", 512), 128, 2048),
                "lora_r": int(p.get("lora_r", 16)) if int(p.get("lora_r", 16)) in (8, 16, 32, 64) else 16,
                "gguf_quant": p.get("gguf_quant", "auto"), "lang": p.get("lang", "ru"),
            },
        }
        agent_core.add_agent(agent)
        clear_stop_flag()
        reset_status(state="running", step=1, progress=0, status="Запуск воркера…", is_error=False,
                     agent_id=aid, agent_name=name, loss_history=[], log_tail=[], result={},
                     started_at=now_ts())
        log_path = os.path.join(LOGS_DIR, f"train_{aid}.log")
        log = open(log_path, "w", encoding="utf-8", errors="replace")
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", TOKENIZERS_PARALLELISM="false",
                   HF_HUB_DISABLE_SYMLINKS_WARNING="1")
        proc = subprocess.Popen([sys.executable, "-u", os.path.join(BASE_DIR, "qlora_trainer.py"),
                                 "--agent-id", aid], cwd=BASE_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
        TRAIN.update(proc=proc, agent_id=aid)
        threading.Thread(target=_finalize, args=(aid, proc, log, log_path), daemon=True).start()
    return agent


def _finalize(aid, proc, log, log_path):
    proc.wait()
    log.close()
    st = read_json(STATUS_FILE, {}) or {}
    state = st.get("state")
    try:
        if state == "worker_done":
            _post_train(aid)
        elif state == "stopped" or (state not in ("error", "done") and os.path.exists(os.path.join(DATA_DIR, "stop.flag"))):
            update_status(state="stopped", is_error=False, status="Остановлено пользователем")
            agent_core.update_agent(aid, train_state="stopped")
        elif state == "error":
            agent_core.update_agent(aid, train_state="failed")
        else:  # процесс упал без отчёта (OOM-killer, падение драйвера)
            tail = [ln for ln in tail_lines(log_path, 8) if ln.strip()]
            update_status(state="error", is_error=True, log_tail=tail,
                          status="Воркер завершился неожиданно: " + (tail[-1][:200] if tail else "см. лог"))
            agent_core.update_agent(aid, train_state="failed")
    except Exception as e:
        update_status(state="error", is_error=True, status=f"Ошибка финализации: {e}")
        agent_core.update_agent(aid, train_state="failed")
    finally:
        clear_stop_flag()
        TRAIN.update(proc=None, agent_id=None)


def _post_train(aid):
    """Шаг 6: проверка модели в Ollama, переключение агента, авто-оценка."""
    update_status(step=6, progress=96, status="Проверка модели в Ollama")
    st = read_json(STATUS_FILE, {}) or {}
    name = (st.get("result") or {}).get("ollama_model")
    if not name or not ollama_api.has_model(name):
        raise RuntimeError("Модель не найдена в Ollama после регистрации")
    agent = agent_core.update_agent(aid, model=name, ollama_model=name, trained=True, train_state="done")
    update_status(progress=97, status="Авто-оценка: 5 тестовых вопросов…")
    try:
        job = read_json(os.path.join(JOBS_DIR, f"{aid}.json"), {}) or {}
        results = train_control.evaluate_agent(name, agent["training"].get("compare_model"),
                                               job.get("eval_items"), agent["prompt"])
        agent_core.update_agent(aid, eval=results)
    except Exception as e:
        update_status(log_tail=[f"Авто-оценка не удалась: {e}"])
    update_status(state="done", progress=100, is_error=False, finished_at=now_ts(),
                  status=f"Готово: агент переключён на {name}")


# ---------------------------------------------------------------- хук для агента-создателя
def _creator_hook(args):
    try:
        agent = start_agent_training({
            "name": args.get("name"), "prompt": args.get("prompt"), "topic": args.get("topic", ""),
            "num_examples": args.get("num_examples", 100),
            "base_model": args.get("base_model") or "Qwen/Qwen2.5-1.5B-Instruct",
            "access": args.get("access", "read-only"),
        })
        return f"Агент «{agent['name']}» создан (id {agent['id']}), QLoRA-обучение запущено. Прогресс — в окне обучения."
    except HTTPException as e:
        return f"Ошибка: {e.detail}"


# ---------------------------------------------------------------- приложение
@asynccontextmanager
async def lifespan(app):
    ensure_data_dir()
    agent_core.seed_default_agents()
    agent_core.create_agent_hook = _creator_hook
    st = read_json(STATUS_FILE, {}) or {}
    if st.get("state") in ("running", "worker_done", "stopping"):
        update_status(state="error", is_error=True, status="Обучение прервано (сервер был перезапущен)")
        if st.get("agent_id"):
            agent_core.update_agent(st["agent_id"], train_state="failed")
    yield


app = FastAPI(title="QLoRA Agent Hub", lifespan=lifespan)
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)  # сервер стартует, даже если папку забыли скопировать
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    page = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(page):
        return HTMLResponse(f"<h3>Не найден интерфейс</h3><p>Положите файл <code>index.html</code> в папку "
                            f"<code>{STATIC_DIR}</code> и обновите страницу.</p>", status_code=404)
    return FileResponse(page)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


def _ver(pkg):
    try:
        return md.version(pkg)
    except Exception:
        return None


@app.get("/api/system/info")
def system_info():
    gpu = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        n, tot, used, drv = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        gpu = {"name": n, "vram_total_mb": int(tot), "vram_used_mb": int(used), "driver": drv}
    except Exception:
        pass
    disk = shutil.disk_usage(DATA_DIR)
    return {
        "gpu": gpu,
        "libs": {k: _ver(k) for k in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")},
        "python": sys.version.split()[0],
        "ollama": ollama_api.version(),
        "llama_cpp": {"dir": merge_lora.llama_dir(), "convert": bool(merge_lora.find_convert_script()),
                      "quantize": bool(merge_lora.find_quantize_exe())},
        "disk_free_gb": round(disk.free / 1024 ** 3, 1),
        "training": training_alive(),
    }


@app.get("/api/local_models")
def local_models():
    hf = []
    root = os.path.join(os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"), "hub")
    if os.path.isdir(root):
        for d in sorted(os.listdir(root)):
            if d.startswith("models--"):
                hf.append(d[len("models--"):].replace("--", "/"))
    return {"ollama": ollama_api.list_models(), "hf_cache": hf, "presets": PRESETS}


@app.get("/api/training/status")
def training_status():
    st = read_json(STATUS_FILE, {}) or {"state": "idle"}
    st["running"] = training_alive()
    st["step_names"] = STEP_NAMES
    return st


@app.post("/api/training/stop")
def training_stop():
    proc = TRAIN["proc"]
    if not training_alive():
        raise HTTPException(409, "Обучение не запущено")
    update_status(state="stopping", status="Остановка…")
    threading.Thread(target=train_control.stop_training, args=(proc,), daemon=True).start()
    return {"ok": True}


@app.post("/api/agents/create_qlora")
def create_qlora(name: str = Form(...), prompt: str = Form(...), base_model: str = Form(...),
                 gen_model: str = Form(""), compare_model: str = Form(""), access: str = Form("read-only"),
                 topic: str = Form(""), num_examples: int = Form(100), epochs: int = Form(3),
                 max_seq_len: int = Form(512), lora_r: int = Form(16), gguf_quant: str = Form("auto"),
                 lang: str = Form("ru"), dataset_file: UploadFile = File(None)):
    data = dataset_file.file.read() if dataset_file is not None and dataset_file.filename else None
    agent = start_agent_training(dict(
        name=name, prompt=prompt, base_model=base_model, gen_model=gen_model, compare_model=compare_model,
        access=access, topic=topic, num_examples=num_examples, epochs=epochs, max_seq_len=max_seq_len,
        lora_r=lora_r, gguf_quant=gguf_quant, lang=lang), data)
    return {"ok": True, "agent": agent}


@app.get("/api/agents")
def list_agents():
    agents = sorted(agent_core.load_agents().values(), key=lambda a: a.get("created", 0))
    return [{**a, "training": None} for a in agents]  # конфиг обучения клиенту не нужен


@app.get("/api/agents/{agent_id}/history")
def history(agent_id: str):
    if not agent_core.get_agent(agent_id):
        raise HTTPException(404, "Агент не найден")
    return agent_core.get_history(agent_id)


class ChatBody(BaseModel):
    message: str


@app.post("/api/agents/{agent_id}/chat")
def chat(agent_id: str, body: ChatBody):
    if not agent_core.get_agent(agent_id):
        raise HTTPException(404, "Агент не найден")
    if not body.message.strip():
        raise HTTPException(400, "Пустое сообщение")
    if training_alive():
        raise HTTPException(409, "Идёт обучение: чат временно отключён, чтобы не занять VRAM")
    try:
        return agent_core.chat(agent_id, body.message.strip())
    except RuntimeError as e:
        raise HTTPException(502, str(e))


@app.post("/api/agents/{agent_id}/clear_history")
def clear_history(agent_id: str):
    agent_core.clear_history(agent_id)
    return {"ok": True}


@app.post("/api/agents/{agent_id}/eval")
def run_eval(agent_id: str):
    agent = agent_core.get_agent(agent_id)
    if not agent or not agent.get("trained"):
        raise HTTPException(400, "Оценка доступна для обученных агентов")
    if training_alive():
        raise HTTPException(409, "Идёт обучение")
    job = read_json(os.path.join(JOBS_DIR, f"{agent_id}.json"), {}) or {}
    res = train_control.evaluate_agent(agent["model"], (agent.get("training") or {}).get("compare_model"),
                                       job.get("eval_items"), agent["prompt"])
    agent_core.update_agent(agent_id, eval=res)
    return res


@app.delete("/api/agents/{agent_id}")
def delete_agent(agent_id: str):
    if TRAIN["agent_id"] == agent_id and training_alive():
        raise HTTPException(409, "Сначала остановите обучение этого агента")
    agent = agent_core.remove_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Агент не найден")
    if agent.get("ollama_model"):
        ollama_api.remove_model(agent["ollama_model"])
    shutil.rmtree(os.path.join(ADAPTERS_DIR, agent_id), ignore_errors=True)
    for f in (f"{agent_id}.jsonl", f"{agent_id}_upload.jsonl"):
        try:
            os.remove(os.path.join(DATASETS_DIR, f))
        except OSError:
            pass
    return {"ok": True}


if __name__ == "__main__":
    ensure_data_dir()
    # только localhost: агенты с доступом execute не должны быть видны по сети
    uvicorn.run(app, host="127.0.0.1", port=8000)
