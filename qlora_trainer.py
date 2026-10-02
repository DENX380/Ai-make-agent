"""
qlora_trainer.py — воркер полного цикла (запускается main.py отдельным процессом):
  1) датасет (JSONL или генерация через Ollama)  2) QLoRA  3) merge  4) GGUF  5) ollama create
Шаг 6 (проверка и переключение агента) выполняет main.py.

Обучение: 4-bit NF4 + double quant + LoRA (все линейные слои), loss ТОЛЬКО на ответах
(prompt маскируется), формат — родной chat template модели. Используется transformers.Trainer
(без TRL: меньше зависимостей от версий, полный контроль над маскированием).
"""
import argparse
import gc
import os
import random
import sys
import traceback

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import dataset_gen
import merge_lora
import ollama_api
from train_control import StatusCallback, Stopped, check_stop
from utils import (ADAPTERS_DIR, AGENTS_FILE, DATASETS_DIR, JOBS_DIR, atomic_write_json,
                   ensure_data_dir, read_json, stop_requested, update_status)

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------- шаг 1: датасет
def step_dataset(agent_id, agent, tr):
    update_status(step=1, progress=1, status="Подготовка датасета")
    upload = tr.get("dataset_path")
    if upload and os.path.exists(upload):
        rows = dataset_gen.load_jsonl(upload)
        update_status(progress=14, status=f"Загружен датасет: {len(rows)} примеров")
    else:
        gen_model = tr["gen_model"]
        target = int(tr["num_examples"])

        def cb(done, total, msg):
            update_status(step=1, progress=round(1 + 13 * done / max(1, total), 1), status=msg)

        rows = dataset_gen.generate_dataset(
            topic=tr.get("topic") or agent["prompt"][:200], model=gen_model, target=target,
            agent_prompt=agent["prompt"], lang=tr.get("lang", "ru"),
            out_path=os.path.join(DATASETS_DIR, f"{agent_id}.jsonl"),
            progress_cb=cb, should_stop=stop_requested)
        ollama_api.unload(gen_model)  # освободить VRAM перед обучением
    check_stop()
    if len(rows) < 5:
        raise RuntimeError(f"Получено только {len(rows)} примеров (нужно ≥ 5). "
                           "Возьмите для генерации модель посильнее или загрузите свой JSONL.")
    random.seed(42)
    random.shuffle(rows)
    eval_rows = []
    if len(rows) >= 30:  # отложенная выборка для проверки обобщения
        eval_rows, rows = rows[:5], rows[5:]
    atomic_write_json(os.path.join(JOBS_DIR, f"{agent_id}.json"),
                      {"eval_items": [{"question": r["instruction"], "reference": r["response"]} for r in eval_rows],
                       "train_size": len(rows)})
    update_status(progress=15, status=f"Датасет готов: {len(rows)} для обучения, {len(eval_rows)} для проверки")
    return rows


# ---------------------------------------------------------------- токенизация
def encode_rows(rows, tok, system_prompt, max_len):
    out, skipped = [], 0
    for r in rows:
        user_msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": r["instruction"]}]
        try:
            prompt = tok.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # шаблон без роли system (Gemma): вклеиваем в user
            user_msgs = [{"role": "user", "content": system_prompt + "\n\n" + r["instruction"]}]
            prompt = tok.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + r["response"] + (tok.eos_token or "")
        try:
            tmpl_full = tok.apply_chat_template(
                user_msgs + [{"role": "assistant", "content": r["response"]}], tokenize=False).rstrip("\n")
            if tmpl_full.startswith(prompt):
                full = tmpl_full  # завершающий токен хода берём из родного шаблона
        except Exception:
            pass
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
        if len(p_ids) >= len(f_ids):
            skipped += 1
            continue
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        out.append({"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels})
    return out, skipped


# ---------------------------------------------------------------- шаг 2: QLoRA
def train(agent_id, agent, tr, rows):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна: проверьте драйвер и torch+cu124")
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    base = agent["base_model"]
    max_len = int(tr.get("max_seq_len", 512))

    update_status(step=2, progress=15, status=f"Загрузка {base} (4-bit)…")
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    data, skipped = encode_rows(rows, tok, agent["prompt"], max_len)
    if len(data) < 5:
        raise RuntimeError(f"После токенизации осталось {len(data)} примеров (слишком длинные?). Увеличьте max_seq_len.")

    class ListDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(data)

        def __getitem__(self, i):
            return data[i]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb,
                                                 device_map={"": 0}, torch_dtype=dtype)
    model.config.use_cache = False
    # без prepare_model_for_kbit_training: он переводит эмбеддинги/lm_head в fp32 (+4 ГБ у 7B)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    r = int(tr.get("lora_r", 16))
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules=TARGETS))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    update_status(status=f"LoRA r={r}: обучаемых параметров {trainable / 1e6:.1f}M, примеров {len(data)}")

    ga = min(8, max(1, len(data) // 4))
    args = TrainingArguments(
        output_dir=os.path.join(ADAPTERS_DIR, agent_id, "_ckpt"),
        per_device_train_batch_size=1, gradient_accumulation_steps=ga,
        num_train_epochs=int(tr.get("epochs", 3)), learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.0,
        logging_steps=1, save_strategy="no", report_to="none", disable_tqdm=True,
        bf16=use_bf16, fp16=not use_bf16, optim="adamw_torch",
        gradient_checkpointing=False, remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ListDataset(),
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100,
                                                           pad_to_multiple_of=8),
                      callbacks=[StatusCallback(15, 65, 2)])
    trainer.train()
    check_stop()

    adapter_dir = os.path.join(ADAPTERS_DIR, agent_id)
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return adapter_dir, ("bf16" if use_bf16 else "fp16")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-id", required=True)
    agent_id = ap.parse_args().agent_id
    ensure_data_dir()
    agent = (read_json(AGENTS_FILE, {}) or {}).get(agent_id)
    if not agent:
        update_status(state="error", is_error=True, status="Агент не найден")
        sys.exit(1)
    tr = agent["training"]
    try:
        rows = step_dataset(agent_id, agent, tr)
        adapter_dir, dtype_name = train(agent_id, agent, tr, rows)
        name = merge_lora.run_pipeline(agent_id, agent["name"], agent["base_model"], adapter_dir,
                                       dtype_name, tr.get("gguf_quant", "auto"))
        update_status(state="worker_done", result={"ollama_model": name}, progress=96,
                      status="Модель зарегистрирована в Ollama")
    except Stopped:
        update_status(state="stopped", is_error=False, status="Остановлено пользователем")
        sys.exit(2)
    except Exception as e:
        traceback.print_exc()
        msg = str(e)
        if "out of memory" in msg.lower():
            msg = "Не хватило VRAM. Уменьшите max_seq_len, возьмите модель поменьше и закройте другие GPU-программы."
        update_status(state="error", is_error=True, status=f"Ошибка: {msg[:400]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
