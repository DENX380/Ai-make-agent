"""
dataset_gen.py — генерация/загрузка датасета (instruction -> response).

  * Ollama format="json": даже мелкие модели отдают валидный JSON
  * сначала подтемы, потом примеры по каждой -> разнообразие
  * валидация (длина, язык, мусор) + дедупликация по Jaccard
  * progress_cb и should_stop для UI и кнопки Stop
Только стандартная библиотека.
"""
import json
import os
import random
import re

import ollama_api

_KEY_ALIASES = {
    "instruction": ("instruction", "question", "prompt", "input", "user", "q"),
    "response": ("response", "answer", "completion", "output", "assistant", "a"),
}


# ---------------------------------------------------------------- парсинг
def _extract_json_array(text):
    """Достаёт список объектов из ответа LLM (dict-обёртка, массив, куски {...})."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

    def as_list(obj):
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return v
            return [obj]
        return []

    try:
        return as_list(json.loads(text))
    except Exception:
        pass
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        try:
            return as_list(json.loads(m.group(0)))
        except Exception:
            pass
    items = []
    for chunk in re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            items.append(json.loads(chunk))
        except Exception:
            continue
    return items


def normalize(item):
    """Любые варианты ключей -> {instruction, response}; иначе None."""
    if not isinstance(item, dict):
        return None
    low = {str(k).lower().strip(): v for k, v in item.items()}
    out = {}
    for target, aliases in _KEY_ALIASES.items():
        for a in aliases:
            if a in low and isinstance(low[a], str) and low[a].strip():
                out[target] = low[a].strip()
                break
    return out if len(out) == 2 else None


def load_jsonl(path):
    """Читает .jsonl или .json (массив) и нормализует поля. Возвращает список примеров."""
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()
    stripped = raw.lstrip()
    items = []
    if stripped.startswith("["):
        try:
            items = json.loads(stripped)
        except Exception:
            items = []
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    for it in items:
        ex = normalize(it)
        if ex:
            rows.append(ex)
    return rows


# ---------------------------------------------------------------- валидация
def _words(s):
    return set(re.findall(r"\w+", s.lower()))


def _cyr_ratio(s):
    letters = re.findall(r"[^\W\d_]", s)
    if not letters:
        return 0.0
    cyr = sum(1 for c in letters if "а" <= c.lower() <= "я" or c.lower() == "ё")
    return cyr / len(letters)


_JUNK = re.compile(r"(as an ai|language model|как языковая модель|я не могу помочь|\[.*(вставьте|insert).*\])", re.I)


def is_valid(ex, lang="ru"):
    ins, res = ex["instruction"], ex["response"]
    if not (8 <= len(ins) <= 600) or not (25 <= len(res) <= 3000):
        return False
    if ins.lower() == res.lower() or _JUNK.search(res):
        return False
    if lang == "ru" and _cyr_ratio(ins + res) < 0.5:
        return False
    return True


def _is_duplicate(ex, seen_words, threshold=0.8):
    w = _words(ex["instruction"])
    if not w:
        return True
    for other in seen_words:
        if len(w & other) / max(1, len(w | other)) >= threshold:
            return True
    return False


# ---------------------------------------------------------------- подтемы
def generate_subtopics(topic, model, n=12, lang="ru"):
    if lang == "ru":
        prompt = (f"Тема ассистента: «{topic}».\nПридумай {n} разных подтем или сценариев, "
                  'по которым пользователи задают вопросы.\nОтветь JSON: {"subtopics": ["...", "..."]}')
    else:
        prompt = (f"Assistant topic: '{topic}'.\nList {n} distinct sub-topics or scenarios users ask about.\n"
                  'Reply as JSON: {"subtopics": ["...", "..."]}')
    try:
        raw = ollama_api.chat(model, [{"role": "user", "content": prompt}], temperature=0.9, json_mode=True)
        data = json.loads(raw)
        subs = [s.strip() for s in data.get("subtopics", []) if isinstance(s, str) and s.strip()]
        if subs:
            return subs[:n]
    except Exception:
        pass
    return [topic]


# ---------------------------------------------------------------- основной цикл
def generate_dataset(topic, model, target=100, agent_prompt="", lang="ru", batch=5,
                     out_path=None, progress_cb=None, should_stop=None, max_failures=25):
    """
    Возвращает список {"instruction","response"}; при out_path пишет JSONL.
    progress_cb(done, target, message); should_stop() -> True прерывает генерацию.
    Совет: генератором лучше брать модель посильнее (qwen2.5:7b), чем 1b.
    """
    subtopics = generate_subtopics(topic, model, n=max(8, target // 6), lang=lang)
    dataset, seen_words = [], []
    failures, i = 0, 0
    styles_ru = ["короткий вопрос", "подробный вопрос с контекстом", "просьба выполнить задачу", "уточняющий вопрос"]

    while len(dataset) < target and failures < max_failures:
        if should_stop and should_stop():
            break
        sub = subtopics[i % len(subtopics)]
        i += 1
        style = random.choice(styles_ru)
        if lang == "ru":
            sys_msg = "Ты генерируешь обучающие данные для ассистента. Отвечай только JSON."
            user_msg = (
                f"Роль ассистента: {agent_prompt or topic}\nПодтема: {sub}\nТип запроса: {style}\n"
                f"Сгенерируй {batch} РАЗНЫХ пар «запрос пользователя — идеальный ответ ассистента» на русском.\n"
                "Ответы содержательные (2-6 предложений), строго в стиле роли, без воды.\n"
                'Формат: {"examples": [{"instruction": "...", "response": "..."}]}'
            )
        else:
            sys_msg = "You generate training data for an assistant. Reply with JSON only."
            user_msg = (
                f"Assistant role: {agent_prompt or topic}\nSub-topic: {sub}\nRequest type: {style}\n"
                f"Generate {batch} DIFFERENT pairs of user request and ideal assistant answer (2-6 sentences each).\n"
                'Format: {"examples": [{"instruction": "...", "response": "..."}]}'
            )
        try:
            raw = ollama_api.chat(
                model,
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                temperature=0.85, json_mode=True, seed=random.randint(1, 10**9),
            )
        except Exception as e:
            failures += 1
            if progress_cb:
                progress_cb(len(dataset), target, f"Ошибка Ollama: {e}")
            if "недоступна" in str(e):
                break
            continue

        added = 0
        for item in _extract_json_array(raw):
            ex = normalize(item)
            if not ex or not is_valid(ex, lang) or _is_duplicate(ex, seen_words):
                continue
            dataset.append(ex)
            seen_words.append(_words(ex["instruction"]))
            added += 1
            if len(dataset) >= target:
                break
        failures = 0 if added else failures + 1
        if progress_cb:
            progress_cb(len(dataset), target, f"Датасет: {len(dataset)}/{target} (подтема: {sub[:40]})")

    if out_path:
        save_jsonl(dataset, out_path)
    return dataset


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
