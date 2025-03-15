# Скрипт для загрузки и подготовки данных TERRa
import os
import json
import pandas as pd
import requests
import zipfile
import random
import time

# Устанавливаем сид для воспроизводимости
SEED = 42
random.seed(SEED)

# Создаем директории для логов и данных
log_dir = "../../logs"
data_dir = "../../data/terra"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

log_file = os.path.join(log_dir, "prepare_terra.log")

def log_message(message):
    """Логирование сообщений"""
    with open(log_file, "a", encoding="utf-8") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def download_terra():
    """Скачивание датасета TERRa, если он еще не скачан"""
    terra_dir = os.path.join(data_dir, "TERRa")
    terra_zip = os.path.join(data_dir, "terra.zip")
    
    if os.path.exists(os.path.join(terra_dir, "train.jsonl")):
        log_message("Датасет TERRa уже скачан.")
        return True
    
    log_message("Скачиваем датасет TERRa...")
    try:
        # Скачиваем архив
        response = requests.get("https://russiansuperglue.com/tasks/download/TERRa")
        with open(terra_zip, "wb") as f:
            f.write(response.content)
        
        # Распаковываем архив
        with zipfile.ZipFile(terra_zip, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        
        log_message("Датасет TERRa успешно скачан и распакован.")
        return True
    except Exception as e:
        log_message(f"Ошибка при скачивании датасета TERRa: {str(e)}")
        return False

def load_jsonl(file_path):
    """Загрузка данных из JSONL файла"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def prepare_terra_data():
    """Подготовка данных TERRa в формате для обучения модели"""
    train_path = os.path.join(data_dir, "TERRa", "train.jsonl")
    val_path = os.path.join(data_dir, "TERRa", "val.jsonl")
    
    # Загружаем данные
    log_message("Загружаем тренировочные данные...")
    train_data = load_jsonl(train_path)
    log_message(f"Загружено {len(train_data)} тренировочных примеров")
    
    log_message("Загружаем валидационные данные...")
    val_data = load_jsonl(val_path)
    log_message(f"Загружено {len(val_data)} валидационных примеров")
    
    # Создаем промпты для обучения
    log_message("Создаем промпты для обучения...")
    
    train_examples = []
    for item in train_data:
        premise = item["premise"]
        hypothesis = item["hypothesis"]
        label = "entailment" if item["label"] == 1 else "not_entailment"
        
        # Формат промпта для инструкций
        prompt = f"Инструкция: Определи, подтверждает ли предложение A предложение B? Ответь только 'entailment' или 'not_entailment'.\n\nВопрос: A: {premise}\nB: {hypothesis}\n\nОтвет:"
        
        train_examples.append({
            "instruction": f"Определи, подтверждает ли предложение A предложение B? Ответь только 'entailment' или 'not_entailment'.",
            "input": f"A: {premise}\nB: {hypothesis}",
            "output": label,
            "prompt": prompt,
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label
        })
    
    val_examples = []
    for item in val_data:
        premise = item["premise"]
        hypothesis = item["hypothesis"]
        label = "entailment" if item["label"] == 1 else "not_entailment"
        
        # Формат промпта для инструкций
        prompt = f"Инструкция: Определи, подтверждает ли предложение A предложение B? Ответь только 'entailment' или 'not_entailment'.\n\nВопрос: A: {premise}\nB: {hypothesis}\n\nОтвет:"
        
        val_examples.append({
            "instruction": f"Определи, подтверждает ли предложение A предложение B? Ответь только 'entailment' или 'not_entailment'.",
            "input": f"A: {premise}\nB: {hypothesis}",
            "output": label,
            "prompt": prompt,
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label
        })
    
    # Сохраняем подготовленные данные
    train_df = pd.DataFrame(train_examples)
    val_df = pd.DataFrame(val_examples)
    
    # Сохраняем в разных форматах для совместимости с разными библиотеками
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    
    with open(os.path.join(data_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_examples, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(data_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_examples, f, ensure_ascii=False, indent=2)
    
    log_message(f"Данные сохранены. Всего: {len(train_examples)} тренировочных и {len(val_examples)} валидационных примеров.")
    
    # Создаем небольшую выборку для быстрого тестирования
    sample_size = min(100, len(train_examples))
    sampled_train = random.sample(train_examples, sample_size)
    with open(os.path.join(data_dir, "train_sample.json"), "w", encoding="utf-8") as f:
        json.dump(sampled_train, f, ensure_ascii=False, indent=2)
    
    log_message(f"Также создана выборка из {sample_size} примеров для быстрого тестирования.")
    
    return train_examples, val_examples

if __name__ == "__main__":
    log_message("Начинаем подготовку данных TERRa...")
    
    # Скачиваем датасет, если нужно
    if download_terra():
        # Подготавливаем данные
        prepare_terra_data()
    
    log_message("Подготовка данных TERRa завершена.") 