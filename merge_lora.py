"""
merge_lora.py — LoRA -> merged HF -> GGUF -> ollama create.

Шаги статуса: 3 merge, 4 GGUF, 5 ollama create.
Modelfile: абсолютный путь с прямыми слэшами + TEMPLATE под семейство модели
(без правильного TEMPLATE дообученная модель отвечает «мимо» формата обучения).
"""
import os
import re
import shutil
import subprocess
import sys

import ollama_api
from train_control import Pulse, check_stop, run_stage
from utils import MERGED_DIR, slugify_ollama, to_posix, update_status

# ---------------------------------------------------------------- шаблоны Ollama
_CHATML = (
    '{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}'
    '{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}'
    '<|im_start|>assistant\n{{ .Response }}<|im_end|>\n'
)
_LLAMA3 = (
    '{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}'
    '{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|>{{ end }}'
    '<|start_header_id|>assistant<|end_header_id|>\n\n{{ .Response }}<|eot_id|>'
)
_GEMMA = (
    '<start_of_turn>user\n{{ if .System }}{{ .System }}\n\n{{ end }}{{ .Prompt }}<end_of_turn>\n'
    '<start_of_turn>model\n{{ .Response }}<end_of_turn>\n'
)
FAMILIES = {
    "qwen": (_CHATML, ["<|im_start|>", "<|im_end|>"]),
    "llama3": (_LLAMA3, ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]),
    "gemma": (_GEMMA, ["<start_of_turn>", "<end_of_turn>"]),
}


def detect_family(model_id):
    s = (model_id or "").lower()
    if "qwen" in s:
        return "qwen"
    if re.search(r"llama[-_ ]?3", s):
        return "llama3"
    if "gemma" in s:
        return "gemma"
    return None


def ollama_name(agent_name, agent_id):
    return f"agent-{slugify_ollama(agent_name)[:24]}-{agent_id[:6]}".lower()


def build_modelfile(gguf_path, family):
    template, stops = FAMILIES[family]
    lines = [f"FROM {to_posix(gguf_path)}", f'TEMPLATE """{template}"""']
    lines += [f'PARAMETER stop "{s}"' for s in stops]
    lines += ["PARAMETER temperature 0.7", "PARAMETER num_ctx 4096"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- llama.cpp
def llama_dir():
    return os.environ.get("LLAMA_CPP_DIR", r"C:\tools\llama.cpp")


def find_convert_script():
    p = os.path.join(llama_dir(), "convert_hf_to_gguf.py")
    return p if os.path.exists(p) else None


def find_quantize_exe():
    base = llama_dir()
    for rel in ("llama-quantize.exe", os.path.join("build", "bin", "Release", "llama-quantize.exe"),
                os.path.join("build", "bin", "llama-quantize.exe"), "llama-quantize"):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return shutil.which("llama-quantize")


def _convert_env():
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    gguf_py = os.path.join(llama_dir(), "gguf-py")
    if os.path.isdir(gguf_py):  # версия gguf, соответствующая скрипту конвертации
        env["PYTHONPATH"] = gguf_py + os.pathsep + env.get("PYTHONPATH", "")
    return env


# ---------------------------------------------------------------- merge
def merge_adapter(base_model, adapter_dir, out_dir, dtype_name):
    """Merge на CPU (VRAM не нужна; для 7B нужно ~16 ГБ RAM)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype,
                                                 low_cpu_mem_usage=True, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    del model


def _fail(msg, tail):
    raise RuntimeError(msg + ("\n" + "\n".join(tail[-6:]) if tail else ""))


def run_pipeline(agent_id, agent_name, base_model, adapter_dir, dtype_name, gguf_pref="auto", cleanup=True):
    """Возвращает имя модели в Ollama."""
    family = detect_family(base_model)
    if not family:
        raise RuntimeError("Неподдерживаемое семейство модели (нужны Qwen, Llama-3 или Gemma)")
    convert = find_convert_script()
    if not convert:
        raise RuntimeError(f"Не найден convert_hf_to_gguf.py в {llama_dir()} (переменная LLAMA_CPP_DIR)")
    ollama_exe = shutil.which("ollama")
    if not ollama_exe:
        raise RuntimeError("Команда ollama не найдена в PATH")

    name = ollama_name(agent_name, agent_id)
    work = os.path.join(MERGED_DIR, agent_id)
    hf_dir = os.path.join(work, "hf")
    os.makedirs(work, exist_ok=True)

    # --- 3. merge
    with Pulse("Merge LoRA в базовую модель (CPU)", 65, 75, 3, tau=120):
        merge_adapter(base_model, adapter_dir, hf_dir, dtype_name)
    check_stop()

    # --- 4. GGUF
    quant_exe = find_quantize_exe()
    pref = (gguf_pref or "auto").lower()
    if pref == "auto":
        pref = "q4_k_m" if quant_exe else "q8_0"
    if pref == "q4_k_m" and not quant_exe:
        pref = "q8_0"
    final_gguf = os.path.join(work, f"{name}.{pref}.gguf")
    outtype = pref if pref in ("q8_0", "f16") else "f16"
    first_gguf = final_gguf if outtype == pref else os.path.join(work, f"{name}.f16.gguf")

    rc, tail = run_stage([sys.executable, convert, hf_dir, "--outfile", first_gguf, "--outtype", outtype],
                         f"Конвертация в GGUF ({outtype})", 75, 84 if outtype != pref else 88, 4,
                         cwd=llama_dir(), env=_convert_env())
    if rc != 0 or not os.path.exists(first_gguf):
        _fail("Конвертация в GGUF не удалась", tail)
    if first_gguf != final_gguf:
        rc, tail = run_stage([quant_exe, first_gguf, final_gguf, pref.upper()],
                             f"Квантизация {pref.upper()}", 84, 88, 4)
        if rc != 0 or not os.path.exists(final_gguf):
            _fail("Квантизация не удалась", tail)
    check_stop()

    # --- 5. ollama create
    modelfile = os.path.join(work, "Modelfile")
    with open(modelfile, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_modelfile(final_gguf, family))
    rc, tail = run_stage([ollama_exe, "create", name, "-f", modelfile],
                         "Регистрация в Ollama", 88, 96, 5, cwd=work)
    if rc != 0 or not ollama_api.has_model(name):
        _fail("ollama create не удался", tail)

    if cleanup:  # Ollama скопировал веса в свой blob-store; адаптер остаётся
        shutil.rmtree(work, ignore_errors=True)
    return name
