"""
train_control.py — прогресс, loss-график, Stop и авто-оценка.

  StatusCallback  : шаг/loss/ETA в training_status.json + остановка по флаг-файлу
  Pulse           : «пульс» прогресса для долгих блокирующих операций (merge)
  run_stage       : запуск внешней команды с парсингом процентов и логом
  stop_training   : мягкая остановка, затем taskkill /T (на Windows terminate() не убивает детей)
  evaluate_agent  : сравнение ответов базовой и обученной моделей
"""
import os
import re
import subprocess
import threading
import time

import ollama_api
from utils import STOP_FLAG, read_json, STATUS_FILE, update_status, stop_requested

try:
    from transformers import TrainerCallback
except Exception:  # сервер main.py может работать без transformers
    TrainerCallback = object

MAX_LOSS_POINTS = 400
MAX_LOG_LINES = 30


class Stopped(Exception):
    """Пользователь нажал Stop."""


def check_stop():
    if stop_requested():
        raise Stopped()


def fmt_eta(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 60} мин {seconds % 60:02d} с" if seconds >= 60 else f"{seconds} с"


# ---------------------------------------------------------------- callback обучения
class StatusCallback(TrainerCallback):
    def __init__(self, p0=15.0, p1=65.0, step_no=2):
        self.p0, self.p1, self.step_no = p0, p1, step_no
        self.t0 = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        st = read_json(STATUS_FILE, {}) or {}
        hist = st.get("loss_history", [])
        hist.append([int(state.global_step), round(float(logs["loss"]), 4)])
        if len(hist) > MAX_LOSS_POINTS:
            hist = hist[::2]
        done, total = state.global_step, max(1, state.max_steps)
        elapsed = time.time() - (self.t0 or time.time())
        eta = elapsed / max(1, done) * (total - done)
        update_status(
            step=self.step_no, loss_history=hist,
            progress=round(self.p0 + (self.p1 - self.p0) * done / total, 1),
            status=f"Обучение QLoRA: шаг {done}/{total}, loss {logs['loss']:.4f}, осталось ~{fmt_eta(eta)}",
        )

    def on_step_end(self, args, state, control, **kwargs):
        if stop_requested():
            control.should_training_stop = True
        return control


# ---------------------------------------------------------------- пульс
class Pulse:
    """Плавно двигает прогресс, пока выполняется блокирующая операция."""

    def __init__(self, label, p0, p1, step_no, tau=90.0):
        self.label, self.p0, self.p1, self.step_no, self.tau = label, p0, p1, step_no, tau
        self._ev = threading.Event()
        self._t = None

    def _run(self):
        t0 = time.time()
        while not self._ev.wait(1.0):
            el = time.time() - t0
            frac = 1 - 1 / (1 + el / self.tau)
            update_status(step=self.step_no, progress=round(self.p0 + (self.p1 - self.p0) * frac, 1),
                          status=f"{self.label} ({int(el)} с)")

    def __enter__(self):
        update_status(step=self.step_no, progress=self.p0, status=self.label)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._ev.set()
        self._t.join(timeout=2)
        if exc[0] is None:
            update_status(progress=self.p1)
        return False


# ---------------------------------------------------------------- внешние команды
def kill_tree(proc):
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        proc.kill()


def run_stage(cmd, label, p0, p1, step_no, cwd=None, env=None):
    """
    Запускает команду, парсит проценты (tqdm llama.cpp, «copying file ... 45%» у ollama),
    пишет хвост лога в статус. Возвращает (returncode, tail).
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    pct_re = re.compile(r"(\d{1,3})%")
    tail, t0, last = [], time.time(), 0.0
    update_status(step=step_no, progress=p0, status=label)
    for line in proc.stdout:
        line = line.strip()
        if line:
            tail = (tail + [line])[-MAX_LOG_LINES:]
        if stop_requested():
            kill_tree(proc)
            raise Stopped()
        now = time.time()
        if now - last < 0.5:
            continue
        last = now
        m = pct_re.search(line)
        frac = min(100, int(m.group(1))) / 100.0 if m else 1 - 1 / (1 + (now - t0) / 60.0)
        update_status(step=step_no, progress=round(p0 + (p1 - p0) * frac, 1),
                      status=f"{label}: {line[:80]}", log_tail=tail)
    proc.wait()
    update_status(progress=p1, log_tail=tail)
    return proc.returncode, tail


def stop_training(proc, grace_seconds=25):
    """Мягко: флаг-файл (callback остановит Trainer). Жёстко: через grace_seconds убиваем дерево."""
    with open(STOP_FLAG, "w") as f:
        f.write("stop")
    deadline = time.time() + grace_seconds
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.5)
    kill_tree(proc)


# ---------------------------------------------------------------- авто-оценка
DEFAULT_QUESTIONS = [
    "Представься и расскажи, чем ты можешь помочь.",
    "Объясни простыми словами, что ты умеешь лучше всего.",
    "Приведи короткий пример типичной задачи, с которой ты справляешься.",
    "Что ты ответишь на вопрос не по твоей теме?",
    "Составь краткий план действий для типового запроса пользователя.",
]


def evaluate_agent(tuned_model, base_model=None, items=None, system_prompt=""):
    """
    items: [{"question","reference"}] — отложенные примеры (не участвовали в обучении).
    Возвращает [{"question","reference","base","tuned"}].
    """
    items = items or [{"question": q, "reference": ""} for q in DEFAULT_QUESTIONS]
    results = []
    for it in items[:5]:
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        msgs.append({"role": "user", "content": it["question"]})
        row = {"question": it["question"], "reference": it.get("reference", ""), "base": "", "tuned": ""}
        for key, model in (("base", base_model), ("tuned", tuned_model)):
            if not model:
                continue
            try:
                row[key] = ollama_api.chat(model, msgs, temperature=0.3, num_predict=300, timeout=300).strip()
            except Exception as e:
                row[key] = f"[ошибка: {e}]"
        results.append(row)
    return results
