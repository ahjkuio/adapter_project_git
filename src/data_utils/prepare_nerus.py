# Скрипт для загрузки и подготовки данных Nerus
import os
import json
import pandas as pd
import requests
import gzip
import random
import time
import re
from tqdm import tqdm

# Устанавливаем сид для воспроизводимости
SEED = 42
random.seed(SEED)

# Создаем директории для логов и данных
log_dir = "../../logs"
data_dir = "../../data/nerus"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

log_file = os.path.join(log_dir, "prepare_nerus.log")

def log_message(message):
    """Логирование сообщений"""
    with open(log_file, "a", encoding="utf-8") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def download_nerus():
    """Скачивание датасета Nerus, если он еще не скачан"""
    nerus_file = os.path.join(data_dir, "nerus_lenta.conllu.gz")
    
    if os.path.exists(nerus_file):
        log_message(f"Файл {nerus_file} уже существует.")
        return True
    
    log_message("Скачиваем датасет Nerus (это может занять время)...")
    try:
        url = "https://storage.yandexcloud.net/natasha-nerus/data/nerus_lenta.conllu.gz"
        
        # Используем потоковую загрузку для больших файлов
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            
            with open(nerus_file, "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), 
                                total=total_size // 8192, 
                                desc="Скачивание Nerus"):
                    f.write(chunk)
        
        log_message("Датасет Nerus успешно скачан.")
        return True
    except Exception as e:
        log_message(f"Ошибка при скачивании датасета Nerus: {str(e)}")
        return False

def parse_natasha_nerus_format(lines):
    """Парсинг предложения в формате Natasha Nerus согласно документации"""
    sentence = {"text": "", "entities": []}
    
    # Извлекаем текст предложения из комментария
    for line in lines:
        if line.startswith("# text"):
            sentence["text"] = line[7:].strip()
            break
    
    if not sentence["text"]:
        return sentence  # Пропускаем предложения без текста
    
    # Собираем токены
    tokens = []
    spans = []  # Начальная и конечная позиции токенов
    
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        
        parts = line.split("\t")
        if len(parts) < 10:
            continue
        
        token = parts[1]  # FORM
        tokens.append(token)
        
        # Проверяем наличие NER меток в поле Tag
        misc = parts[9]  # MISC поле
        ner_tag = None
        
        # Обрабатываем тег в формате "Tag=B-PER" или "Tag=I-LOC"
        if "Tag=" in misc:
            tag_match = re.search(r"Tag=([BI])-([A-Z]+)", misc)
            if tag_match:
                ner_tag = (tag_match.group(1), tag_match.group(2))
        
        if ner_tag:
            tag_type, entity_type = ner_tag
            
            if tag_type == "B":  # Начало новой сущности
                # Начало и конец пока совпадают - один токен
                spans.append({
                    "start": len(tokens) - 1,  # Индекс начального токена (0-индексация)
                    "end": len(tokens),        # Индекс после последнего токена
                    "type": entity_type
                })
            elif tag_type == "I" and spans:  # Продолжение сущности
                # Проверяем, что это продолжение последней сущности
                last_span = spans[-1]
                if last_span["end"] == len(tokens) - 1 and last_span["type"] == entity_type:
                    last_span["end"] = len(tokens)  # Увеличиваем конец
    
    # Преобразуем индексы токенов в текст сущностей
    entities = []
    for span in spans:
        entity_tokens = tokens[span["start"]:span["end"]]
        entity_text = " ".join(entity_tokens)
        entities.append({
            "text": entity_text,
            "type": span["type"]
        })
    
    sentence["entities"] = entities
    return sentence

def analyze_nerus_sample(file_path, num_sentences=3):
    """Анализирует первые несколько предложений из файла Nerus для определения формата"""
    log_message(f"Анализируем формат данных Nerus из файла {file_path}...")
    
    sentences = []
    current_lines = []
    count = 0
    
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if not line and current_lines:
                sentences.append(current_lines)
                current_lines = []
                count += 1
                if count >= num_sentences:
                    break
            else:
                current_lines.append(line)
    
    for i, sentence_lines in enumerate(sentences):
        log_message(f"\n--- Предложение {i+1} ---")
        
        # Находим текст предложения
        text_found = False
        for line in sentence_lines:
            if line.startswith("# text"):
                log_message(f"Текст: {line[7:].strip()}")
                text_found = True
                break
        
        if not text_found:
            log_message("Текст предложения не найден в комментариях!")
        
        # Выводим примеры строк с токенами
        token_lines = [line for line in sentence_lines if not line.startswith("#") and line.strip()]
        log_message(f"Всего токенов: {len(token_lines)}")
        
        if token_lines:
            log_message("Примеры строк с токенами:")
            for j, line in enumerate(token_lines[:5]):
                log_message(f"  Токен {j+1}: {line}")
            
            # Проверяем структуру первого токена
            if token_lines:
                first_token = token_lines[0]
                parts = first_token.split("\t")
                log_message(f"Количество колонок: {len(parts)}")
                
                # Выводим названия колонок CoNLL-U
                columns = ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"]
                for i, (name, value) in enumerate(zip(columns, parts)):
                    log_message(f"  {name}: {value[:30]}")
                
                # Ищем NER в различных местах
                for i, part in enumerate(parts):
                    if "B-" in part or "I-" in part or "NER=" in part:
                        log_message(f"  Возможная NER информация в колонке {i+1} ({columns[i] if i < len(columns) else 'Unknown'}): {part}")
            
            # Особенно проверяем колонку MISC
            misc_values = [line.split("\t")[9] if len(line.split("\t")) > 9 else "" for line in token_lines]
            unique_misc = set(misc_values)
            log_message(f"Уникальные значения MISC (до 5): {list(unique_misc)[:5]}")
            
            # Проверяем каждую колонку на предмет информации о NER
            for col_idx in range(min(len(parts), 10)):
                col_values = [line.split("\t")[col_idx] if len(line.split("\t")) > col_idx else "" for line in token_lines]
                ner_values = [val for val in col_values if "NER" in val or ("B-" in val and len(val) <= 6) or ("I-" in val and len(val) <= 6)]
                
                if ner_values:
                    log_message(f"  Колонка {col_idx+1} ({columns[col_idx] if col_idx < len(columns) else 'Unknown'}) содержит возможные NER метки: {ner_values[:5]}")
    
    return sentences

def prepare_nerus_data(max_sentences=5000):
    """Подготовка данных Nerus в формате для обучения модели"""
    nerus_file = os.path.join(data_dir, "nerus_lenta.conllu.gz")
    
    # Анализируем пример данных для определения формата
    sample_sentences = analyze_nerus_sample(nerus_file)
    
    log_message(f"Обрабатываем до {max_sentences} предложений из Nerus...")
    
    # Читаем файл и извлекаем предложения
    sentences = []
    current_lines = []
    sentence_count = 0
    total_sentences = 0
    sentences_with_entities = 0
    
    with gzip.open(nerus_file, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Чтение Nerus"):
            line = line.strip()
            
            if not line and current_lines:
                # Пустая строка означает конец предложения
                total_sentences += 1
                
                # Используем специальный парсер для Natasha Nerus
                sentence = parse_natasha_nerus_format(current_lines)
                
                # Отладочная информация
                if total_sentences % 1000 == 0:
                    log_message(f"Обработано {total_sentences} предложений, найдено {sentences_with_entities} предложений с сущностями")
                    if sentence["text"]:
                        log_message(f"Пример текста: {sentence['text'][:100]}...")
                        log_message(f"Найдено сущностей: {len(sentence['entities'])}")
                        if sentence["entities"]:
                            log_message(f"Пример сущности: {sentence['entities'][0]}")
                
                if sentence["text"] and len(sentence["entities"]) > 0:
                    sentences.append(sentence)
                    sentences_with_entities += 1
                    sentence_count += 1
                    if sentence_count >= max_sentences:
                        break
                current_lines = []
            else:
                current_lines.append(line)
    
    log_message(f"Всего обработано {total_sentences} предложений, найдено {sentences_with_entities} предложений с именованными сущностями.")
    log_message(f"Загружено {len(sentences)} предложений с именованными сущностями.")
    
    # Разделяем на тренировочную и валидационную выборки
    random.shuffle(sentences)
    split_idx = int(0.9 * len(sentences))
    train_sentences = sentences[:split_idx]
    val_sentences = sentences[split_idx:]
    
    log_message(f"Разделено на {len(train_sentences)} тренировочных и {len(val_sentences)} валидационных примеров.")
    
    # Создаем промпты для обучения
    log_message("Создаем промпты для обучения...")
    
    train_examples = []
    for idx, sentence in enumerate(train_sentences):
        text = sentence["text"]
        entities = sentence["entities"]
        
        # Форматируем ожидаемый ответ
        entities_text = []
        for entity in entities:
            entities_text.append(f"[{entity['text']}]({entity['type']})")
        
        expected_output = ", ".join(entities_text) if entities_text else "В тексте нет именованных сущностей."
        
        # Формат промпта для инструкций
        prompt = f"Инструкция: Найди и классифицируй все именованные сущности в тексте, указав их тип (PER - персона, LOC - локация, ORG - организация).\n\nВопрос: {text}\n\nОтвет:"
        
        train_examples.append({
            "instruction": "Найди и классифицируй все именованные сущности в тексте, указав их тип (PER - персона, LOC - локация, ORG - организация).",
            "input": text,
            "output": expected_output,
            "prompt": prompt,
            "text": text,
            "entities": entities
        })
    
    val_examples = []
    for idx, sentence in enumerate(val_sentences):
        text = sentence["text"]
        entities = sentence["entities"]
        
        # Форматируем ожидаемый ответ
        entities_text = []
        for entity in entities:
            entities_text.append(f"[{entity['text']}]({entity['type']})")
        
        expected_output = ", ".join(entities_text) if entities_text else "В тексте нет именованных сущностей."
        
        # Формат промпта для инструкций
        prompt = f"Инструкция: Найди и классифицируй все именованные сущности в тексте, указав их тип (PER - персона, LOC - локация, ORG - организация).\n\nВопрос: {text}\n\nОтвет:"
        
        val_examples.append({
            "instruction": "Найди и классифицируй все именованные сущности в тексте, указав их тип (PER - персона, LOC - локация, ORG - организация).",
            "input": text,
            "output": expected_output,
            "prompt": prompt,
            "text": text,
            "entities": entities
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
    log_message("Начинаем подготовку данных Nerus...")
    
    # Скачиваем датасет, если нужно
    if download_nerus():
        # Подготавливаем данные (ограничиваем количество для экономии времени)
        prepare_nerus_data(max_sentences=5000)
    
    log_message("Подготовка данных Nerus завершена.") 