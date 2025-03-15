#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для обучения LoRA адаптеров для Llama 3.1 8B модели
на датасете Nerus для задачи именованных сущностей
"""

import os
import sys
import json
import time
import logging
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import wandb
from torch.utils.data import DataLoader

# Устанавливаем переменные окружения для улучшения стабильности CUDA операций
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Поможет с отладкой CUDA ошибок

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

# Устанавливаем сид для воспроизводимости
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Настройка логирования
log_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train_lora.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Создаем директории для сохранения моделей
models_dir = os.path.join(PROJECT_ROOT, "models", "lora")
os.makedirs(models_dir, exist_ok=True)

# Константы
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Выбираем GPU с наибольшим количеством свободной памяти
def select_best_gpu():
    """Выбор GPU с наибольшим объемом свободной памяти"""
    if not torch.cuda.is_available():
        logging.info("CUDA не доступна, используем CPU")
        return "cpu"
    
    # Получаем количество доступных GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        logging.info(f"Доступна только одна GPU: {torch.cuda.get_device_name(0)}")
        return "cuda:0"
    
    # Выбираем GPU с наибольшим объемом свободной памяти
    free_memory = []
    for i in range(num_gpus):
        # Получаем информацию о памяти для каждой GPU
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        free_memory.append(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i))
    
    best_gpu = free_memory.index(max(free_memory))
    logging.info(f"Выбрана GPU {best_gpu}: {torch.cuda.get_device_name(best_gpu)}")
    logging.info(f"Свободная память: {max(free_memory) / 1024**3:.2f} ГБ")
    
    return f"cuda:{best_gpu}"

# Функция для загрузки данных Nerus
def load_nerus_data():
    """Загрузка данных Nerus для обучения"""
    logging.info("Загрузка данных Nerus...")
    
    train_file = os.path.join(PROJECT_ROOT, "data", "nerus", "train.json")
    val_file = os.path.join(PROJECT_ROOT, "data", "nerus", "val.json")
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        logging.error(f"Файлы данных не найдены. Убедитесь, что данные подготовлены в {train_file} и {val_file}")
        return None, None
    
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    with open(val_file, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    logging.info(f"Загружено {len(train_data)} тренировочных и {len(val_data)} валидационных примеров")
    
    # Преобразуем данные в формат datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset

# Функция для подготовки данных для обучения
def prepare_dataset_for_training(dataset, tokenizer, max_length=256):
    """Подготовка датасета для обучения с безопасной обработкой индексов"""
    def preprocess_function(examples):
        # Кодируем только промпты с padding
        inputs = tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None  # Возвращаем словарь, а не тензоры
        )
        
        # Кодируем полные тексты промпт+ответ
        full_texts = [examples["prompt"][i] + examples["output"][i] for i in range(len(examples["prompt"]))]
        full_encodings = tokenizer(
            full_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
        
        # Копируем input_ids для создания меток
        labels = []
        for i in range(len(examples["prompt"])):
            # Кодируем только промпт для точного определения длины
            prompt_encoding = tokenizer(examples["prompt"][i], add_special_tokens=True)
            prompt_length = len(prompt_encoding.input_ids)
            
            # Создаем копию полных input_ids
            label = full_encodings["input_ids"][i].copy()
            
            # Маскируем токены промпта и pad-токены
            for j in range(len(label)):
                if j < prompt_length or label[j] == tokenizer.pad_token_id:
                    label[j] = -100
            
            labels.append(label)
        
        result = {
            "input_ids": full_encodings["input_ids"],
            "attention_mask": full_encodings["attention_mask"],
            "labels": labels
        }
        
        return result
    
    # Применяем предобработку к датасету
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return processed_dataset

# Функция для настройки LoRA конфигурации
def get_lora_config(
    r=4,
    alpha=8,
    dropout=0.05,
    # Ограничиваем целевые модули для лучшей стабильности
    target_modules=["q_proj", "v_proj"]  # только query и value проекции
):
    """Создание конфигурации LoRA"""
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    return config

# Функция для загрузки модели с квантизацией
def load_model_for_training():
    """Загрузка модели с квантизацией для обучения"""
    logging.info(f"Загружаем модель {BASE_MODEL}...")
    
    # Настраиваем конфигурацию для квантизации
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Загружаем модель с квантизацией
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        max_memory={0: "32GiB"},  # Ограничиваем использование памяти
        use_cache=False  # Отключаем кэширование ключей/значений для обучения
    )
    
    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Настройка PAD-токена для токенизатора
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Подготавливаем модель для 4-битного обучения
    model = prepare_model_for_kbit_training(model)
    
    logging.info("Модель и токенизатор успешно загружены")
    
    return model, tokenizer

# Функция для обучения модели с LoRA
def train_lora_adapter(
    train_dataset,
    val_dataset,
    model,
    tokenizer,
    lora_config,
    output_dir=os.path.join(PROJECT_ROOT, "models", "lora"),
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    max_grad_norm=0.3,
    logging_steps=10,
    eval_steps=100,
    save_steps=100,
    warmup_ratio=0.03,
    use_wandb=False
):
    """Обучение модели с LoRA адаптерами"""
    
    # Инициализируем wandb для отслеживания процесса обучения
    if use_wandb:
        wandb.init(
            project="llama-lora-adapters",
            name=f"lora-r{lora_config.r}-alpha{lora_config.lora_alpha}",
            config={
                "model": BASE_MODEL,
                "lora_rank": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "target_modules": lora_config.target_modules,
                "dropout": lora_config.lora_dropout,
                "epochs": num_train_epochs,
                "batch_size": per_device_train_batch_size,
                "learning_rate": learning_rate
            }
        )
    
    # Применяем LoRA адаптеры к модели
    logging.info("Применяем LoRA адаптеры к модели...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Создаем аргументы для обучения
    logging.info("Настраиваем параметры обучения...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if use_wandb else "none",
        warmup_ratio=warmup_ratio,
        remove_unused_columns=False,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
        run_name=f"lora-r{lora_config.r}-alpha{lora_config.lora_alpha}" if use_wandb else None,
        # Параметры для стабильности
        local_rank=-1,
        dataloader_num_workers=4,
        no_cuda=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,  # Включаем для экономии памяти
        torch_compile=False,  # Отключаем для снижения риска ошибок
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Более стабильный режим
        deepspeed=None,  # Явно отключаем DeepSpeed
        auto_find_batch_size=False,  # Отключаем автоматический поиск размера батча
    )
    
    # Создаем обучающий объект
    from transformers import Trainer
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Начинаем обучение
    logging.info("Начинаем обучение...")
    train_result = trainer.train()
    
    # Сохраняем модель и токенизатор
    logging.info("Сохраняем обученную модель...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Выводим метрики обучения
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Оцениваем модель на валидационном наборе
    logging.info("Оцениваем модель на валидационном наборе...")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    # Закрываем wandb
    if use_wandb:
        wandb.finish()
    
    return model, tokenizer, metrics

# Функция для оценки модели с LoRA на датасете Nerus
def evaluate_model_on_nerus(model, tokenizer, device, adapter_name=None):
    """Оценка модели с LoRA на датасете Nerus"""
    from src.models.evaluate_base import evaluate_nerus
    
    # Если указан адаптер, включаем его
    if adapter_name:
        logging.info(f"Включаем адаптер {adapter_name}...")
        model.set_adapter(adapter_name)
    
    # Оцениваем модель
    logging.info(f"Оцениваем модель на датасете Nerus...")
    metrics = evaluate_nerus(model, tokenizer, device)
    
    return metrics

# Функция для оценки модели с LoRA на датасете TERRa
def evaluate_model_on_terra(model, tokenizer, device, adapter_name=None):
    """Оценка модели с LoRA на датасете TERRa"""
    from src.models.evaluate_base import evaluate_terra
    
    # Если указан адаптер, включаем его
    if adapter_name:
        logging.info(f"Включаем адаптер {adapter_name}...")
        model.set_adapter(adapter_name)
    
    # Оцениваем модель
    logging.info(f"Оцениваем модель на датасете TERRa...")
    metrics = evaluate_terra(model, tokenizer, device)
    
    return metrics

# Функция для инференса с использованием обученной модели
def generate_with_lora(
    model,
    tokenizer,
    prompt,
    adapter_name=None,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
):
    """Генерация ответа с использованием обученной модели с LoRA"""
    # Если указан адаптер, включаем его
    if adapter_name:
        logging.info(f"Включаем адаптер {adapter_name}...")
        model.set_adapter(adapter_name)
    
    # Формируем инструкцию в формате промпта
    instruction_prompt = f"""Инструкция: Найди и классифицируй все именованные сущности в тексте, указав их тип (PER - персона, LOC - локация, ORG - организация).

Вопрос: {prompt}

Ответ:"""
    
    # Токенизируем промпт
    inputs = tokenizer(instruction_prompt, return_tensors="pt").to(model.device)
    
    # Замеряем время инференса
    start_time = time.time()
    
    # Генерируем ответ с улучшенными параметрами
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            num_return_sequences=1,
            no_repeat_ngram_size=3
        )
    
    inference_time = time.time() - start_time
    
    # Декодируем только сгенерированную часть (ответ)
    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    # Количество сгенерированных токенов
    generated_tokens = outputs.shape[1] - input_length
    
    logging.info(f"Ответ сгенерирован за {inference_time:.2f} сек, {generated_tokens} токенов")
    return response, inference_time, generated_tokens

# Функция для загрузки обученной модели с LoRA
def load_model_with_lora(lora_path):
    """Загрузка обученной модели с LoRA адаптерами"""
    logging.info(f"Загружаем базовую модель {BASE_MODEL}...")
    
    # Настраиваем конфигурацию для квантизации
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Загружаем модель с квантизацией
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Настройка PAD-токена для токенизатора
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Загружаем LoRA адаптеры
    logging.info(f"Загружаем LoRA адаптеры из {lora_path}...")
    from peft import PeftModel
    
    model = PeftModel.from_pretrained(model, lora_path)
    
    logging.info("Модель с LoRA адаптерами успешно загружена")
    
    return model, tokenizer

# Функция для сравнения результатов с базовой моделью
def compare_with_base_model(base_metrics, lora_metrics):
    """Сравнение результатов базовой модели и модели с LoRA"""
    logging.info("Сравнение результатов базовой модели и модели с LoRA:")
    
    for dataset in ["terra", "nerus"]:
        if dataset in base_metrics and dataset in lora_metrics:
            logging.info(f"\nДатасет {dataset.upper()}:")
            for metric in base_metrics[dataset]:
                if metric in lora_metrics[dataset]:
                    base_value = base_metrics[dataset][metric]
                    lora_value = lora_metrics[dataset][metric]
                    change = lora_value - base_value
                    change_percent = (change / base_value * 100) if base_value != 0 else float('inf')
                    
                    logging.info(f"{metric}: {base_value:.4f} -> {lora_value:.4f} [Изменение: {change:.4f} ({change_percent:+.2f}%)]")

# Основная функция
def main(
    lora_rank=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    batch_size=2,
    grad_accum=4,
    epochs=2,
    learning_rate=5e-5,
    train_model=True,
    evaluate_base=True,
    evaluate_lora=True,
    test_examples=True,
    use_wandb=False
):
    """Основная функция для обучения и оценки модели с LoRA"""
    # Выбираем GPU
    device = select_best_gpu()
    
    # Загружаем данные
    train_dataset, val_dataset = load_nerus_data()
    if train_dataset is None or val_dataset is None:
        logging.error("Ошибка при загрузке данных. Выход.")
        return
    
    # Создаем директорию для результатов
    output_dir = os.path.join(PROJECT_ROOT, "models", "lora", f"r{lora_rank}_alpha{lora_alpha}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Определяем конфигурацию LoRA
    lora_config = get_lora_config(
        r=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=target_modules
    )
    
    # Сохраняем общие метрики
    metrics = {}
    
    # Оцениваем базовую модель
    if evaluate_base:
        # Загружаем базовую модель
        logging.info("Загружаем базовую модель для оценки...")
        model, tokenizer = load_model_for_training()
        
        # Оцениваем на TERRa
        logging.info("Оцениваем базовую модель на датасете TERRa...")
        terra_metrics = evaluate_model_on_terra(model, tokenizer, device)
        metrics["base"] = {"terra": terra_metrics}
        
        # Оцениваем на Nerus
        logging.info("Оцениваем базовую модель на датасете Nerus...")
        nerus_metrics = evaluate_model_on_nerus(model, tokenizer, device)
        metrics["base"]["nerus"] = nerus_metrics
        
        # Сохраняем метрики базовой модели
        results_dir = os.path.join(PROJECT_ROOT, "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "base_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics["base"], f, ensure_ascii=False, indent=2)
    
    # Обучаем модель с LoRA
    if train_model:
        # Загружаем модель для обучения
        model, tokenizer = load_model_for_training()
        
        # Подготавливаем данные для обучения
        logging.info("Подготавливаем данные для обучения...")
        train_data_processed = prepare_dataset_for_training(train_dataset, tokenizer)
        val_data_processed = prepare_dataset_for_training(val_dataset, tokenizer)
        
        # Обучаем модель
        model, tokenizer, train_metrics = train_lora_adapter(
            train_data_processed,
            val_data_processed,
            model,
            tokenizer,
            lora_config,
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            use_wandb=use_wandb
        )
        
        metrics["lora_train"] = train_metrics
    
    # Оцениваем модель с LoRA
    if evaluate_lora:
        if not train_model:
            # Если мы не обучали модель, загружаем предварительно обученную
            logging.info(f"Загружаем предварительно обученную модель с LoRA из {output_dir}...")
            model, tokenizer = load_model_with_lora(output_dir)
        
        # Оцениваем на TERRa
        logging.info("Оцениваем модель с LoRA на датасете TERRa...")
        terra_metrics = evaluate_model_on_terra(model, tokenizer, device, adapter_name="default")
        metrics["lora"] = {"terra": terra_metrics}
        
        # Оцениваем на Nerus
        logging.info("Оцениваем модель с LoRA на датасете Nerus...")
        nerus_metrics = evaluate_model_on_nerus(model, tokenizer, device, adapter_name="default")
        metrics["lora"]["nerus"] = nerus_metrics
        
        # Сохраняем метрики модели с LoRA
        with open(os.path.join(output_dir, "lora_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics["lora"], f, ensure_ascii=False, indent=2)
        
        # Сравниваем с базовой моделью, если есть данные
        if "base" in metrics:
            compare_with_base_model(metrics["base"], metrics["lora"])
    
    # Тестируем модель на нескольких примерах
    if test_examples and (train_model or evaluate_lora):
        logging.info("\n--- Тестирование модели на примерах ---")
        
        # Примеры для тестирования
        test_examples = [
            "Президент России Владимир Путин провел совещание в Москве с представителями Газпрома.",
            "В Санкт-Петербурге на фестивале выступили музыканты из группы Аквариум.",
            "Глава МИД Сергей Лавров встретился со своим французским коллегой Жаном-Ивом Ле Дрианом."
        ]
        
        # Тестируем с базовой моделью
        if evaluate_base:
            logging.info("\nБазовая модель:")
            # Загружаем базовую модель, если нужно
            if "model" not in locals() or model.peft_config is not None:
                model, tokenizer = load_model_for_training()
            
            for idx, example in enumerate(test_examples):
                logging.info(f"\nПример {idx+1}: {example}")
                response, time_taken, tokens = generate_with_lora(model, tokenizer, example)
                logging.info(f"Ответ базовой модели: {response}")
                logging.info(f"Время: {time_taken:.2f} сек, Токенов: {tokens}")
        
        # Тестируем с LoRA
        logging.info("\nМодель с LoRA:")
        # Загружаем модель с LoRA, если нужно
        if "model" not in locals() or model.peft_config is None:
            model, tokenizer = load_model_with_lora(output_dir)
        
        for idx, example in enumerate(test_examples):
            logging.info(f"\nПример {idx+1}: {example}")
            response, time_taken, tokens = generate_with_lora(model, tokenizer, example, adapter_name="default")
            logging.info(f"Ответ модели с LoRA: {response}")
            logging.info(f"Время: {time_taken:.2f} сек, Токенов: {tokens}")
    
    logging.info("\nРабота скрипта завершена!")
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Обучение и оценка LoRA адаптеров для Llama 3.1 8B")
    parser.add_argument("--rank", type=int, default=4, help="ранг LoRA адаптера")
    parser.add_argument("--alpha", type=int, default=8, help="alpha параметр LoRA")
    parser.add_argument("--dropout", type=float, default=0.05, help="dropout для LoRA")
    parser.add_argument("--batch-size", type=int, default=2, help="размер батча")
    parser.add_argument("--grad-accum", type=int, default=4, help="количество шагов накопления градиента")
    parser.add_argument("--epochs", type=int, default=2, help="количество эпох обучения")
    parser.add_argument("--lr", type=float, default=5e-5, help="скорость обучения")
    parser.add_argument("--target-modules", type=str, default="q_proj,v_proj", 
                        help="целевые модули для адаптации (разделенные запятыми)")
    parser.add_argument("--train", action="store_true", help="обучить модель")
    parser.add_argument("--evaluate-base", action="store_true", help="оценить базовую модель")
    parser.add_argument("--evaluate-lora", action="store_true", help="оценить модель с LoRA")
    parser.add_argument("--test", action="store_true", help="протестировать на примерах")
    parser.add_argument("--use-wandb", action="store_true", help="использовать wandb для отслеживания")
    
    args = parser.parse_args()
    
    # Разбираем целевые модули
    target_modules = args.target_modules.split(',')
    
    # Запускаем основную функцию с переданными аргументами
    main(
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        learning_rate=args.lr,
        target_modules=target_modules,
        train_model=args.train,
        evaluate_base=args.evaluate_base,
        evaluate_lora=args.evaluate_lora,
        test_examples=args.test,
        use_wandb=args.use_wandb
    ) 