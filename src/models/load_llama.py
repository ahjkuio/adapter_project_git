# Скрипт для загрузки и запуска Llama 3.1 8B-Instruct на конкретной GPU
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

# Создаем директорию для логов
os.makedirs("../../logs", exist_ok=True)
log_file = "../../logs/model_load.log"

# Функция для логирования
def log_message(message):
    with open(log_file, "a", encoding="utf-8") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    print(message)

# Явно выбираем GPU 1
gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Проверяем доступность GPU
log_message("Проверка доступных GPU...")
if torch.cuda.is_available():
    log_message(f"Используем GPU {gpu_id}: {torch.cuda.get_device_name(0)}")
else:
    log_message("GPU не найден, используем CPU (работа будет медленной)")
    device = torch.device("cpu")

device = torch.device("cuda:0")  # После установки CUDA_VISIBLE_DEVICES, это будет GPU 1

# Настраиваем конфигурацию для квантизации модели
log_message("Настраиваем параметры квантизации...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                   # 4-битная квантизация
    bnb_4bit_compute_dtype=torch.bfloat16,  # Используем bfloat16 для вычислений
    bnb_4bit_use_double_quant=True,      # Двойная квантизация для экономии памяти
    bnb_4bit_quant_type="nf4"            # Нормализованная квантизация для лучшего качества
)

# Загружаем Instruct версию модели
model_id = "unsloth/Meta-Llama-3.1-8B-Instruct"
log_message(f"Загружаем модель {model_id}...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,      # Используем квантизацию
        device_map="auto",                   # Автоматически распределяем модель по доступным устройствам
        torch_dtype=torch.bfloat16,          # Используем bfloat16 там, где нет квантизации
        low_cpu_mem_usage=True               # Снижаем использование памяти CPU
    )
    log_message("Модель успешно загружена")
    
    # Загружаем токенизатор
    log_message("Загружаем токенизатор...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Настраиваем токенизатор
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log_message("Токенизатор успешно загружен")
    
    # Функция для генерации ответа
    def generate_response(prompt, max_tokens=100):
        log_message(f"Генерация ответа на промпт: {prompt[:50]}...")
        
        # Простой формат для запроса
        instruction_prompt = f"""Инструкция: Ответь на следующий вопрос кратко и точно.

Вопрос: {prompt}

Ответ:"""
        
        # Токенизируем промпт
        inputs = tokenizer(instruction_prompt, return_tensors="pt").to(device)
        
        # Замеряем время инференса
        start_time = time.time()
        
        # Генерируем ответ с измененными параметрами
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.2,          # Небольшая температура для некоторой вариативности
                do_sample=True,          # Включаем сэмплирование
                top_p=0.9,               # Параметр top_p для более стабильной генерации
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id  # Указываем токен конца последовательности
            )
        
        inference_time = time.time() - start_time
        
        # Декодируем только сгенерированную часть (ответ)
        input_length = inputs.input_ids.shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Количество сгенерированных токенов
        generated_tokens = outputs.shape[1] - input_length
        
        log_message(f"Ответ сгенерирован за {inference_time:.2f} сек, {generated_tokens} токенов")
        return response, inference_time, generated_tokens
    
    # Тестируем модель на нескольких примерах
    log_message("\n--- Тестирование модели ---")
    
    # Тест 1: Общий вопрос
    test_prompt = "Что такое языковые модели и каковы их основные возможности?"
    log_message(f"\nТестовый запрос 1: {test_prompt}")
    response, time_taken, tokens = generate_response(test_prompt)
    log_message(f"Ответ модели: {response}")
    log_message(f"Время: {time_taken:.2f} сек, Токенов: {tokens}")
    
    # Тест 2: Пример из TERRa
    terra_prompt = "Определи, подтверждает ли предложение A предложение B? Ответь только 'entailment' или 'not_entailment'.\nA: Мужчина в красной рубашке сидит на скамейке.\nB: Человек сидит на скамейке."
    log_message(f"\nТестовый запрос 2 (TERRa): {terra_prompt}")
    response, time_taken, tokens = generate_response(terra_prompt)
    log_message(f"Ответ модели: {response}")
    log_message(f"Время: {time_taken:.2f} сек, Токенов: {tokens}")
    
    # Тест 3: Пример для поиска именованных сущностей
    ner_prompt = "Найди и перечисли все именованные сущности в тексте, указав их тип (PER - персона, LOC - локация, ORG - организация): Президент России Владимир Путин провел совещание в Москве с представителями Газпрома."
    log_message(f"\nТестовый запрос 3 (NER): {ner_prompt}")
    response, time_taken, tokens = generate_response(ner_prompt)
    log_message(f"Ответ модели: {response}")
    log_message(f"Время: {time_taken:.2f} сек, Токенов: {tokens}")
    
    log_message("\nТестирование модели завершено успешно!")
    
except Exception as e:
    log_message(f"Ошибка при загрузке модели: {str(e)}")

log_message("Скрипт выполнен")