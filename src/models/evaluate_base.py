#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для оценки базовой модели Llama 3.1 8B на датасетах TERRa и Nerus.
Вычисляет метрики и сохраняет результаты для последующего сравнения.
"""

import os
import json
import time
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import random
import re

# Устанавливаем сид для воспроизводимости
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Пути к данным и директориям результатов
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TERRA_DIR = os.path.join(BASE_DIR, "data", "terra")
NERUS_DIR = os.path.join(BASE_DIR, "data", "nerus")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "baseline")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Создаем директории, если их нет
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Настройки модели
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_LENGTH = 512  # Максимальная длина входного текста
MAX_NEW_TOKENS = 128  # Максимальное количество генерируемых токенов
BATCH_SIZE = 1  # Размер батча (1 для экономии памяти)
NUM_EXAMPLES = 100  # Количество примеров для оценки (ограничиваем для экономии времени)

# Инициализация логирования
log_file = os.path.join(LOG_DIR, "evaluate_base.log")

def log_message(message):
    """Логирование сообщений"""
    with open(log_file, "a", encoding="utf-8") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def load_model():
    """Загрузка модели и токенизатора"""
    log_message(f"Загружаем модель {MODEL_ID}...")
    
    # Проверяем доступность GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log_message(f"Используем GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        log_message("GPU не найден, используем CPU (работа будет медленной)")
    
    # Настройка квантизации для экономии памяти
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Загружаем модель с квантизацией
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log_message("Модель и токенизатор успешно загружены")
    return model, tokenizer, device

def generate_response(model, tokenizer, prompt, device, max_new_tokens=MAX_NEW_TOKENS):
    """Генерация ответа модели на заданный промпт"""
    # Токенизируем промпт
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Ограничиваем длину входных данных, чтобы избежать ошибок
    if inputs.input_ids.shape[1] > MAX_LENGTH:
        inputs.input_ids = inputs.input_ids[:, :MAX_LENGTH]
        inputs.attention_mask = inputs.attention_mask[:, :MAX_LENGTH]
    
    # Генерируем ответ с ограниченным числом токенов, убираем несовместимые параметры
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,          # Уменьшаем максимальное количество токенов
            do_sample=True,             # Включаем семплирование для разнообразия ответов
            temperature=0.1,            # Низкая температура для более детерминированных ответов
            num_beams=1,                # Отключаем beam search для ускорения
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Декодируем только сгенерированную часть (ответ)
    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response.strip()

def evaluate_terra(model, tokenizer, device):
    """Оценка модели на датасете TERRa"""
    log_message("\n--- Оценка на датасете TERRa ---")
    
    # Загружаем валидационные данные
    val_path = os.path.join(TERRA_DIR, "val.json")
    if not os.path.exists(val_path):
        log_message(f"Ошибка: Файл {val_path} не найден.")
        
        # Проверим наличие альтернативных файлов
        log_message(f"Проверяем содержимое директории {TERRA_DIR}:")
        if os.path.exists(TERRA_DIR):
            files = os.listdir(TERRA_DIR)
            log_message(f"Найденные файлы: {files}")
            
            # Проверяем наличие файла val.jsonl из исходного датасета
            terra_dir = os.path.join(TERRA_DIR, "TERRa")
            if os.path.exists(terra_dir):
                log_message(f"Проверяем содержимое директории {terra_dir}:")
                terra_files = os.listdir(terra_dir)
                log_message(f"Найденные файлы: {terra_files}")
                
                # Если найден исходный файл, но не преобразованный, предлагаем решение
                if "val.jsonl" in terra_files:
                    log_message("Найден исходный файл val.jsonl, но отсутствует преобразованный val.json.")
                    log_message("Необходимо запустить скрипт prepare_terra.py для преобразования данных.")
                    return None
        else:
            log_message(f"Директория {TERRA_DIR} не существует.")
            log_message("Необходимо запустить скрипт prepare_terra.py для подготовки данных.")
            return None
    
    log_message(f"Загружаем данные из файла {val_path}")
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    # Ограничиваем количество примеров для экономии времени
    if len(val_data) > NUM_EXAMPLES:
        val_data = random.sample(val_data, NUM_EXAMPLES)
    
    log_message(f"Загружено {len(val_data)} валидационных примеров")
    
    results = []
    y_true = []
    y_pred = []
    
    # Оцениваем модель на каждом примере
    for i, example in enumerate(tqdm(val_data, desc="Оценка TERRa")):
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        true_label = example["label"]  # entailment или not_entailment
        
        # Полностью новый формат промпта с выбором варианта по номеру
        prompt = f"""Инструкция: Внимательно проанализируй два предложения A и B. Определи, следует ли B логически из A.
Выбери ТОЛЬКО ОДИН вариант ответа - 1 или 2:

1. ENTAILMENT (логически следует) - выбирается, когда информация в B полностью содержится в A
2. NOT_ENTAILMENT (не следует) - выбирается, когда B содержит информацию, которой нет в A, или противоречит A

Примеры:
Пример 1:
A: В зоопарке живут слоны, жирафы и зебры.
B: В зоопарке есть жирафы.
Ответ: 1 (ENTAILMENT), потому что B полностью содержится в A.

Пример 2:
A: Иван посетил Москву и Санкт-Петербург в прошлом году.
B: Иван был только в Москве в прошлом году.
Ответ: 2 (NOT_ENTAILMENT), потому что в A сказано, что Иван был в двух городах, а не только в Москве.

Пример 3:
A: Студенты могут выбрать математику или физику в качестве факультатива.
B: Все студенты обязаны изучать математику.
Ответ: 2 (NOT_ENTAILMENT), потому что в A речь идет о возможности выбора, а в B - об обязательном изучении.

Твоя задача:
A: {premise}
B: {hypothesis}

Ответ (выбери 1 или 2):"""
        
        # Генерируем ответ
        response = generate_response(model, tokenizer, prompt, device)
        
        # Новая логика извлечения ответа - сначала ищем цифры в ответе
        predicted_label = "unknown"
        
        # Сначала проверяем наличие чисел 1 или 2 в первых нескольких символах
        first_chars = response[:10].lower()
        if "1" in first_chars or ("entail" in first_chars and "not" not in first_chars):
            predicted_label = "entailment"
        elif "2" in first_chars or "not_entail" in first_chars or ("not" in first_chars and "entail" in first_chars):
            predicted_label = "not_entailment"
        # Если цифр нет, используем базовую логику
        elif "entailment" in response.lower() and "not_entailment" not in response.lower() and "not entail" not in response.lower():
            predicted_label = "entailment"
        elif "not_entailment" in response.lower() or "not entail" in response.lower():
            predicted_label = "not_entailment"
        
        # Сохраняем результаты
        results.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "full_response": response
        })
        
        y_true.append(true_label)
        y_pred.append(predicted_label)
        
        # Выводим примеры для отладки
        if i < 3:
            log_message(f"\nПример {i+1}:")
            log_message(f"Premise: {premise}")
            log_message(f"Hypothesis: {hypothesis}")
            log_message(f"True label: {true_label}")
            log_message(f"Predicted label: {predicted_label}")
            log_message(f"Response: {response}")
    
    # Вычисляем метрики
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    log_message("\nРезультаты оценки TERRa:")
    log_message(f"Accuracy: {accuracy:.4f}")
    log_message(f"Precision: {precision:.4f}")
    log_message(f"Recall: {recall:.4f}")
    log_message(f"F1: {f1:.4f}")
    
    # Сохраняем результаты
    results_file = os.path.join(RESULTS_DIR, "terra_results.json")
    metrics_file = os.path.join(RESULTS_DIR, "terra_metrics.json")
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    log_message(f"Результаты сохранены в {results_file} и {metrics_file}")
    
    return metrics

def extract_entities_from_response(response):
    """Извлечение именованных сущностей из ответа модели с поддержкой различных форматов"""
    entities = []
    
    # Пытаемся извлечь сущности в разных форматах
    
    # Формат 1: [текст](тип)
    pattern1 = r'\[(.*?)\]\((PER|LOC|ORG)\)'
    matches1 = re.findall(pattern1, response)
    
    # Формат 2: текст (тип)
    pattern2 = r'(\S+(?:\s+\S+)*?)\s*\((PER|LOC|ORG)\)'
    matches2 = re.findall(pattern2, response)
    
    # Формат 3: #текст (тип) или # текст (тип)
    pattern3 = r'#\s*(\S+(?:\s+\S+)*?)\s*\((PER|LOC|ORG)\)'
    matches3 = re.findall(pattern3, response)
    
    # Формат 4: * текст (тип) - с маркером списка
    pattern4 = r'\*\s*(\S+(?:\s+\S+)*?)\s*\((PER|LOC|ORG)\)'
    matches4 = re.findall(pattern4, response)
    
    # Формат 5: текст - тип или текст – тип (с разными типами тире)
    pattern5 = r'(\S+(?:\s+\S+)*?)\s*[-–—]\s*(PER|LOC|ORG)'
    matches5 = re.findall(pattern5, response)
    
    # Собираем все найденные сущности
    all_matches = matches1 + matches2 + matches3 + matches4 + matches5
    
    # Удаляем дубликаты и добавляем в результат
    seen = set()
    for text, entity_type in all_matches:
        # Удаляем лишние пробелы и проверяем, что сущность не пустая
        text = text.strip()
        if text and (text, entity_type) not in seen:
            entities.append({
                "text": text,
                "type": entity_type
            })
            seen.add((text, entity_type))
    
    return entities

def evaluate_entity_extraction(true_entities, predicted_entities):
    """Оценка извлечения именованных сущностей"""
    # Преобразуем сущности в строки для простого сравнения
    true_entity_strings = set([f"{e['text']}_{e['type']}" for e in true_entities])
    pred_entity_strings = set([f"{e['text']}_{e['type']}" for e in predicted_entities])
    
    # Вычисляем метрики
    true_positives = len(true_entity_strings.intersection(pred_entity_strings))
    false_positives = len(pred_entity_strings - true_entity_strings)
    false_negatives = len(true_entity_strings - pred_entity_strings)
    
    # Если нет предсказаний и истинных сущностей, считаем метрики равными 1
    if len(true_entity_strings) == 0 and len(pred_entity_strings) == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    # Если нет предсказаний, но есть истинные сущности
    elif len(pred_entity_strings) == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    # Если есть предсказания, но нет истинных сущностей
    elif len(true_entity_strings) == 0:
        precision = 0.0
        recall = 1.0
        f1 = 0.0
    # Обычный случай
    else:
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def evaluate_nerus(model, tokenizer, device):
    """Оценка модели на датасете Nerus"""
    log_message("\n--- Оценка на датасете Nerus ---")
    
    # Загружаем валидационные данные
    val_path = os.path.join(NERUS_DIR, "val.json")
    if not os.path.exists(val_path):
        log_message(f"Ошибка: Файл {val_path} не найден.")
        
        # Проверим наличие альтернативных файлов
        log_message(f"Проверяем содержимое директории {NERUS_DIR}:")
        if os.path.exists(NERUS_DIR):
            files = os.listdir(NERUS_DIR)
            log_message(f"Найденные файлы: {files}")
            
            # Если найден исходный файл Nerus, но не преобразованный
            if any(f.endswith(".conllu.gz") for f in files):
                log_message("Найден исходный файл .conllu.gz, но отсутствует преобразованный val.json.")
                log_message("Необходимо запустить скрипт prepare_nerus.py для преобразования данных.")
        else:
            log_message(f"Директория {NERUS_DIR} не существует.")
            log_message("Необходимо запустить скрипт prepare_nerus.py для подготовки данных.")
        
        return None
    
    log_message(f"Загружаем данные из файла {val_path}")
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    # Ограничиваем количество примеров для экономии времени
    if len(val_data) > NUM_EXAMPLES:
        val_data = random.sample(val_data, NUM_EXAMPLES)
    
    log_message(f"Загружено {len(val_data)} валидационных примеров")
    
    results = []
    metrics_list = []
    
    # Оцениваем модель на каждом примере
    for i, example in enumerate(tqdm(val_data, desc="Оценка Nerus")):
        text = example["text"]
        true_entities = example["entities"]
        
        # Еще более строгие инструкции с прямым запретом использовать другие форматы
        prompt = f"""Инструкция: Найди все именованные сущности в тексте и классифицируй их.

СТРОГО СОБЛЮДАЙ ФОРМАТ! Для каждой найденной сущности напиши ТОЛЬКО в формате:
[Сущность](Тип)

Где Тип - это ТОЛЬКО один из:
* PER - для имен людей
* LOC - для географических мест
* ORG - для организаций

ЗАПРЕЩЕНО использовать:
* маркеры списка (*, -, 1., •)
* дефисы или тире между сущностью и типом
* любые другие форматы

Примеры ПРАВИЛЬНОГО формата:
[Владимир Путин](PER)
[Москва](LOC)
[Газпром](ORG)

Примеры НЕПРАВИЛЬНОГО формата:
* Владимир Путин (PER)  ← НЕ ИСПОЛЬЗОВАТЬ маркеры списка
Москва - LOC            ← НЕ ИСПОЛЬЗОВАТЬ тире
"Газпром" (ORG)         ← НЕ ИСПОЛЬЗОВАТЬ кавычки

Текст для анализа: {text}

Найденные сущности (строго в формате [Сущность](Тип)):"""
        
        # Генерируем ответ
        response = generate_response(model, tokenizer, prompt, device)
        
        # Извлекаем предсказанные сущности
        predicted_entities = extract_entities_from_response(response)
        
        # Вычисляем метрики для этого примера
        example_metrics = evaluate_entity_extraction(true_entities, predicted_entities)
        metrics_list.append(example_metrics)
        
        # Сохраняем результаты
        results.append({
            "text": text,
            "true_entities": true_entities,
            "predicted_entities": predicted_entities,
            "full_response": response,
            "metrics": example_metrics
        })
        
        # Выводим примеры для отладки
        if i < 3:
            log_message(f"\nПример {i+1}:")
            log_message(f"Текст: {text}")
            log_message(f"Истинные сущности: {true_entities}")
            log_message(f"Предсказанные сущности: {predicted_entities}")
            log_message(f"Response: {response}")
            log_message(f"Метрики: P={example_metrics['precision']:.4f}, R={example_metrics['recall']:.4f}, F1={example_metrics['f1']:.4f}")
    
    # Вычисляем общие метрики
    avg_precision = np.mean([m["precision"] for m in metrics_list])
    avg_recall = np.mean([m["recall"] for m in metrics_list])
    avg_f1 = np.mean([m["f1"] for m in metrics_list])
    
    macro_metrics = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1
    }
    
    log_message("\nРезультаты оценки Nerus (macro):")
    log_message(f"Precision: {avg_precision:.4f}")
    log_message(f"Recall: {avg_recall:.4f}")
    log_message(f"F1: {avg_f1:.4f}")
    
    # Вычисляем micro-метрики (общие TP, FP, FN по всем примерам)
    total_tp = sum([m["true_positives"] for m in metrics_list])
    total_fp = sum([m["false_positives"] for m in metrics_list])
    total_fn = sum([m["false_negatives"] for m in metrics_list])
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    micro_metrics = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn
    }
    
    log_message("\nРезультаты оценки Nerus (micro):")
    log_message(f"Precision: {micro_precision:.4f}")
    log_message(f"Recall: {micro_recall:.4f}")
    log_message(f"F1: {micro_f1:.4f}")
    
    # Сохраняем результаты
    results_file = os.path.join(RESULTS_DIR, "nerus_results.json")
    metrics_file = os.path.join(RESULTS_DIR, "nerus_metrics.json")
    
    metrics = {
        "macro": macro_metrics,
        "micro": micro_metrics
    }
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    log_message(f"Результаты сохранены в {results_file} и {metrics_file}")
    
    return metrics

def plot_metrics(terra_metrics, nerus_metrics):
    """Визуализация метрик в виде графиков"""
    # Создаем директорию для визуализаций
    viz_dir = os.path.join(RESULTS_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Метрики TERRa
    plt.figure(figsize=(10, 6))
    metrics = list(terra_metrics.keys())
    values = list(terra_metrics.values())
    plt.bar(metrics, values, color='skyblue')
    plt.title('Метрики базовой модели на датасете TERRa')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(viz_dir, 'terra_metrics.png'), dpi=300, bbox_inches='tight')
    
    # Метрики Nerus
    plt.figure(figsize=(10, 6))
    metrics = ['precision', 'recall', 'f1']
    macro_values = [nerus_metrics['macro'][m] for m in metrics]
    micro_values = [nerus_metrics['micro'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, macro_values, width, label='Macro', color='skyblue')
    plt.bar(x + width/2, micro_values, width, label='Micro', color='lightcoral')
    
    plt.title('Метрики базовой модели на датасете Nerus')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(viz_dir, 'nerus_metrics.png'), dpi=300, bbox_inches='tight')
    
    log_message(f"Графики метрик сохранены в директории {viz_dir}")

def main():
    """Основная функция для оценки базовой модели"""
    log_message("Начинаем оценку базовой модели Llama 3.1 8B")
    
    # Загружаем модель
    model, tokenizer, device = load_model()
    
    # Оцениваем модель на TERRa
    terra_metrics = evaluate_terra(model, tokenizer, device)
    
    # Оцениваем модель на Nerus
    nerus_metrics = evaluate_nerus(model, tokenizer, device)
    
    # Визуализируем результаты только если есть метрики обоих датасетов
    if terra_metrics and nerus_metrics:
        plot_metrics(terra_metrics, nerus_metrics)
        log_message("Оценка базовой модели успешно завершена")
    else:
        log_message("Оценка не была полностью завершена из-за отсутствия данных.")
        log_message("Пожалуйста, запустите скрипты подготовки данных и повторите попытку.")
        
        # Проверяем наличие скриптов подготовки данных
        data_utils_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_utils")
        
        if os.path.exists(data_utils_dir):
            log_message(f"Скрипты подготовки данных можно найти в директории: {data_utils_dir}")
            scripts = [f for f in os.listdir(data_utils_dir) if f.startswith("prepare_")]
            if scripts:
                log_message(f"Доступные скрипты: {scripts}")
                log_message("Рекомендуется запустить данные скрипты для подготовки данных.")
        else:
            log_message(f"Директория со скриптами подготовки данных {data_utils_dir} не найдена.")
            
        log_message("После подготовки данных запустите этот скрипт снова.")

if __name__ == "__main__":
    main() 