"""
agent_core.py — агенты, чат, инструменты, история.

Доступ (накопительный): read-only < file-system < specific-apps < execute.
Файлы — только внутри песочницы data/workspace (проверка realpath, а не чёрный список).
Протокол инструментов — текст: <tool>{"tool": "...", "args": {...}}</tool> (надёжнее для мелких моделей).
"""
import json
import os
import re
import subprocess
import threading
import time
import uuid

import ollama_api
from utils import (AGENTS_FILE, HISTORY_FILE, WORKSPACE_DIR, atomic_write_json, now_ts, read_json)

ACCESS_LEVELS = ["read-only", "file-system", "specific-apps", "execute"]
_FS = ["list_files", "read_file", "write_file"]
ACCESS_TOOLS = {
    "read-only": [],
    "file-system": _FS,
    "specific-apps": _FS + ["run_app"],
    "execute": _FS + ["run_app", "run_command"],
}
DEFAULT_APPS = {"notepad": "notepad.exe", "calc": "calc.exe", "mspaint": "mspaint.exe"}
TOOL_DOCS = {
    "list_files": 'list_files {"path": "."} — список файлов в песочнице',
    "read_file": 'read_file {"path": "notes.txt"} — прочитать файл',
    "write_file": 'write_file {"path": "a.txt", "content": "..."} — записать файл (перезапись)',
    "run_app": 'run_app {"app": "notepad", "args": ["a.txt"]} — запустить разрешённое приложение',
    "run_command": 'run_command {"command": "dir"} — выполнить команду shell в песочнице',
    "create_agent": 'create_agent {"name": "...", "prompt": "роль агента", "topic": "тема", "num_examples": 100} '
                    "— создать нового агента и запустить его QLoRA-обучение",
}
MAX_STEPS = 4
HISTORY_LIMIT = 40
SUMMARIZE_CHUNK = 20
CTX_MESSAGES = 24

BLACKLIST = [
    r"\brm\s+-[a-z]*r", r"\bformat\s+[a-z]:", r"\bshutdown\b", r"\bdel\s+/[a-z]", r"\brmdir\s+/s", r"\brd\s+/s",
    r"\bmkfs\b", r"\breg\s+(delete|add)", r"\bdiskpart\b", r"\bcipher\s+/w", r"\bbcdedit\b", r"\btaskkill\b",
    r"remove-item.*-recurse", r"c:[\\/]+windows", r"(^|\s)/etc\b", r"\bnet\s+user\b", r"\bdd\s+if=", r":\(\)\s*\{",
]

_lock = threading.RLock()
create_agent_hook = None  # main.py подставляет функцию создания QLoRA-агента


class ToolError(Exception):
    pass


# ---------------------------------------------------------------- хранилище
def load_agents():
    with _lock:
        return read_json(AGENTS_FILE, {}) or {}


def save_agents(agents):
    with _lock:
        atomic_write_json(AGENTS_FILE, agents)


def get_agent(agent_id):
    return load_agents().get(agent_id)


def update_agent(agent_id, **fields):
    with _lock:
        agents = load_agents()
        if agent_id in agents:
            agents[agent_id].update(fields)
            save_agents(agents)
        return agents.get(agent_id)


def add_agent(agent):
    with _lock:
        agents = load_agents()
        agents[agent["id"]] = agent
        save_agents(agents)
    return agent


def remove_agent(agent_id):
    with _lock:
        agents = load_agents()
        agent = agents.pop(agent_id, None)
        save_agents(agents)
        hist = read_json(HISTORY_FILE, {}) or {}
        hist.pop(agent_id, None)
        atomic_write_json(HISTORY_FILE, hist)
    return agent


def new_agent_id():
    return uuid.uuid4().hex[:8]


def seed_default_agents():
    """Два базовых агента-помощника (не обучаются; модель подбирается из установленных в Ollama)."""
    agents = load_agents()
    if agents:
        return
    defaults = [
        ("assistant", "Помощник", "Ты дружелюбный и точный ассистент. Отвечай кратко и по делу на языке пользователя.",
         "read-only", False),
        ("creator", "Создатель агентов",
         "Ты помогаешь пользователю проектировать ИИ-агентов. Обсуди роль и тему агента, затем создай его "
         "инструментом create_agent: агент будет обучен методом QLoRA. Файлы можно хранить в песочнице.",
         "file-system", True),
    ]
    for aid, name, prompt, access, creator in defaults:
        agents[aid] = {"id": aid, "name": name, "prompt": prompt, "model": "", "base_model": "", "access": access,
                       "trained": False, "train_state": None, "can_create_agents": creator,
                       "allowed_apps": [], "created": now_ts()}
    save_agents(agents)


# ---------------------------------------------------------------- песочница и инструменты
def _safe_path(rel):
    rel = str(rel or ".").replace("\\", "/").lstrip("/")
    root = os.path.realpath(WORKSPACE_DIR)
    full = os.path.realpath(os.path.join(root, rel))
    try:
        inside = os.path.commonpath([full, root]) == root
    except ValueError:  # другой диск на Windows
        inside = False
    if not inside:
        raise ToolError("Путь вне песочницы запрещён")
    return full


def _oem_encoding():
    if os.name == "nt":
        try:
            import ctypes
            return f"cp{ctypes.windll.kernel32.GetOEMCP()}"
        except Exception:
            pass
    return "utf-8"


def tool_list_files(args, agent):
    path = _safe_path(args.get("path", "."))
    if not os.path.isdir(path):
        raise ToolError("Это не папка")
    items = [(n + "/" if os.path.isdir(os.path.join(path, n)) else n) for n in sorted(os.listdir(path))]
    return "\n".join(items[:200]) or "(пусто)"


def tool_read_file(args, agent):
    path = _safe_path(args.get("path"))
    if not os.path.isfile(path):
        raise ToolError("Файл не найден")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = f.read(20001)
    return data[:20000] + ("\n…(обрезано)" if len(data) > 20000 else "")


def tool_write_file(args, agent):
    path = _safe_path(args.get("path"))
    content = str(args.get("content", ""))
    if len(content) > 200_000:
        raise ToolError("Файл слишком большой (>200 КБ)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return f"Записано {len(content)} символов в {os.path.relpath(path, os.path.realpath(WORKSPACE_DIR))}"


def tool_run_app(args, agent):
    name = str(args.get("app", "")).lower()
    allowed = agent.get("allowed_apps") or list(DEFAULT_APPS)
    if name not in allowed or name not in DEFAULT_APPS:
        raise ToolError(f"Приложение не разрешено. Доступны: {', '.join(a for a in allowed if a in DEFAULT_APPS)}")
    extra = [_safe_path(a) for a in (args.get("args") or [])][:5]
    subprocess.Popen([DEFAULT_APPS[name]] + extra, cwd=WORKSPACE_DIR)
    return f"Приложение {name} запущено"


def tool_run_command(args, agent):
    cmd = str(args.get("command", "")).strip()
    if not cmd:
        raise ToolError("Пустая команда")
    low = cmd.lower()
    for pat in BLACKLIST:
        if re.search(pat, low):
            raise ToolError("Команда заблокирована политикой безопасности")
    try:
        p = subprocess.run(cmd, shell=True, cwd=WORKSPACE_DIR, capture_output=True, text=True,
                           encoding=_oem_encoding(), errors="replace", timeout=30)
    except subprocess.TimeoutExpired:
        raise ToolError("Таймаут 30 с")
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    return f"[код {p.returncode}]\n{out[:6000]}" if out else f"[код {p.returncode}] (нет вывода)"


def tool_create_agent(args, agent):
    if not create_agent_hook:
        raise ToolError("Создание агентов недоступно")
    return create_agent_hook(args)


TOOL_FUNCS = {"list_files": tool_list_files, "read_file": tool_read_file, "write_file": tool_write_file,
              "run_app": tool_run_app, "run_command": tool_run_command, "create_agent": tool_create_agent}


def tools_for(agent):
    tools = list(ACCESS_TOOLS.get(agent.get("access", "read-only"), []))
    if agent.get("can_create_agents"):
        tools.append("create_agent")
    return tools


TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)


def parse_tool_call(text):
    m = TOOL_RE.search(text or "")
    if not m:
        return None
    raw = re.sub(r"^```(?:json)?|```$", "", m.group(1).strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict) or "tool" not in obj:
            raise ValueError
    except Exception:
        return {"tool": "__invalid__", "args": {}}
    args = obj.get("args")
    return {"tool": str(obj["tool"]), "args": args if isinstance(args, dict) else {}}


def run_tool(agent, call):
    name = call["tool"]
    if name == "__invalid__":
        return 'Ошибка: неверный формат. Нужно <tool>{"tool": "имя", "args": {...}}</tool>'
    if name not in tools_for(agent):
        return f"Ошибка: инструмент {name} недоступен"
    try:
        return str(TOOL_FUNCS[name](call["args"], agent))
    except ToolError as e:
        return f"Ошибка: {e}"
    except Exception as e:
        return f"Ошибка выполнения: {e}"


# ---------------------------------------------------------------- история
def _entry(hist, agent_id):
    e = hist.get(agent_id)
    if not isinstance(e, dict):
        e = {"summary": "", "messages": []}
        hist[agent_id] = e
    return e


def get_history(agent_id):
    with _lock:
        return list(_entry(read_json(HISTORY_FILE, {}) or {}, agent_id)["messages"])


def clear_history(agent_id):
    with _lock:
        hist = read_json(HISTORY_FILE, {}) or {}
        hist[agent_id] = {"summary": "", "messages": []}
        atomic_write_json(HISTORY_FILE, hist)


def _append_history(agent_id, msgs):
    with _lock:
        hist = read_json(HISTORY_FILE, {}) or {}
        e = _entry(hist, agent_id)
        e["messages"].extend(msgs)
        if len(e["messages"]) > HISTORY_LIMIT * 2:  # страховка, если суммаризация не успела
            e["messages"] = e["messages"][-HISTORY_LIMIT:]
        atomic_write_json(HISTORY_FILE, hist)
        return len(e["messages"])


_summarizing = set()


def _summarize(agent_id, model):
    try:
        with _lock:
            e = _entry(read_json(HISTORY_FILE, {}) or {}, agent_id)
            old = e["messages"][:SUMMARIZE_CHUNK]
            prev = e.get("summary", "")
        text = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in old)
        summary = ollama_api.chat(model, [
            {"role": "system", "content": "Сделай краткое резюме диалога (5-8 предложений). Сохрани факты, "
                                          "имена, договорённости и открытые задачи."},
            {"role": "user", "content": (f"Прежнее резюме:\n{prev}\n\n" if prev else "") + f"Новые сообщения:\n{text}"},
        ], temperature=0.2, num_predict=400)
        with _lock:
            hist = read_json(HISTORY_FILE, {}) or {}
            e = _entry(hist, agent_id)
            e["summary"] = summary.strip()
            e["messages"] = e["messages"][len(old):]
            atomic_write_json(HISTORY_FILE, hist)
    except Exception:
        pass
    finally:
        _summarizing.discard(agent_id)


# ---------------------------------------------------------------- чат
def resolve_model(agent):
    installed = [m["name"] for m in ollama_api.list_models()]
    if not installed:
        raise RuntimeError("В Ollama нет моделей или Ollama не запущена")
    norm = lambda n: n if ":" in n else n + ":latest"
    for cand in (agent.get("model"), agent.get("fallback_model")):
        if cand and norm(cand) in [norm(i) for i in installed]:
            return cand
    for pref in ("qwen2.5", "llama3", "qwen", "llama"):
        for i in installed:
            if pref in i.lower():
                return i
    return installed[0]


def build_system_prompt(agent, summary=""):
    parts = [agent["prompt"].strip()]
    tools = tools_for(agent)
    if tools:
        parts.append(
            "У тебя есть инструменты. Чтобы вызвать инструмент, ответь ТОЛЬКО блоком:\n"
            '<tool>{"tool": "имя", "args": {...}}</tool>\n'
            "Один вызов за ответ; дождись результата. Когда задача выполнена — ответь пользователю "
            "обычным текстом без <tool>. Файлы — только в песочнице, относительные пути.\nИнструменты:\n"
            + "\n".join("- " + TOOL_DOCS[t] for t in tools))
    if summary:
        parts.append("Краткое содержание предыдущей беседы:\n" + summary)
    return "\n\n".join(parts)


def chat(agent_id, text):
    agent = get_agent(agent_id)
    if not agent:
        raise KeyError(agent_id)
    model = resolve_model(agent)
    with _lock:
        e = _entry(read_json(HISTORY_FILE, {}) or {}, agent_id)
        past = [{"role": m["role"], "content": m["content"]} for m in e["messages"][-CTX_MESSAGES:]]
        summary = e.get("summary", "")
    messages = [{"role": "system", "content": build_system_prompt(agent, summary)}] + past
    messages.append({"role": "user", "content": text})

    steps, final = [], ""
    use_tools = bool(tools_for(agent))
    for _ in range(MAX_STEPS):
        reply = ollama_api.chat(model, messages, temperature=0.7)
        call = parse_tool_call(reply) if use_tools else None
        if not call:
            final = reply.strip()
            break
        result = run_tool(agent, call)
        steps.append({"tool": call["tool"], "args": call["args"], "result": result[:2000]})
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"[Результат инструмента {call['tool']}]\n{result}\n"
                         "Продолжай. Если задача выполнена — дай финальный ответ без <tool>."})
    else:
        final = "(достигнут лимит шагов)"
    final = TOOL_RE.sub("", final).strip() or "(пустой ответ)"

    ts = now_ts()
    n = _append_history(agent_id, [{"role": "user", "content": text, "ts": ts},
                                   {"role": "assistant", "content": final, "steps": steps, "ts": ts}])
    if n > HISTORY_LIMIT and agent_id not in _summarizing:
        _summarizing.add(agent_id)
        threading.Thread(target=_summarize, args=(agent_id, model), daemon=True).start()
    return {"reply": final, "steps": steps, "model": model}
