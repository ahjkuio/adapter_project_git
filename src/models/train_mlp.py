#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для обучения MLP адаптеров для Llama 3.1 8B модели
на датасете Nerus для задачи именованных сущностей.
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
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import wandb
from torch.utils.data import DataLoader
import traceback

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
log_file = os.path.join(log_dir, "train_mlp.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Создаем директории для сохранения моделей
models_dir = os.path.join(PROJECT_ROOT, "models", "mlp")
os.makedirs(models_dir, exist_ok=True)

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Класс MLP-адаптера
class MLPAdapter(nn.Module):
    """
    Класс для MLP-адаптера, который будет добавляться параллельно к выходам трансформерных блоков.
    Структура: input -> down_projection -> activation -> up_projection -> output
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 dropout=0.1, 
                 activation=nn.GELU, 
                 init_scale=0.01):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_scale = init_scale
        
        # Создаем слои MLP
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.activation = activation()
        self.up_proj = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Инициализация весов с малыми значениями для устойчивого обучения
        self._init_weights()
        
        # Явно устанавливаем requires_grad=True для всех параметров
        for param in self.parameters():
            param.requires_grad = True
    
    def _init_weights(self):
        """Инициализация весов адаптера с малыми значениями"""
        nn.init.normal_(self.down_proj.weight, std=self.init_scale)
        nn.init.normal_(self.up_proj.weight, std=self.init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        """Прямой проход через MLP-адаптер"""
        # Сохраняем оригинальный ввод для резидуального соединения
        residual = x
        
        # Проход через MLP
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        
        # Аддитивно комбинируем с резидуальным соединением
        output = residual + x
        
        return output

class MLPAdapterModel(nn.Module):
    """
    Класс-обертка для модели с MLP-адаптерами.
    Добавляет MLP-адаптеры к выходам трансформерных блоков.
    """
    def __init__(self, 
                 base_model, 
                 hidden_dim=128, 
                 dropout=0.1, 
                 num_layers=4, 
                 layers_to_adapt=None):
        super().__init__()
        
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Определяем слои, к которым будем добавлять адаптеры
        self.num_layers = len(self.base_model.model.layers)
        
        if layers_to_adapt is None:
            # По умолчанию адаптируем каждый num_layers/5 слой
            self.layers_to_adapt = list(range(0, self.num_layers, max(1, self.num_layers // 5)))[:num_layers]
        else:
            self.layers_to_adapt = layers_to_adapt
        
        # Создаем адаптеры для выбранных слоев
        self.adapters = nn.ModuleDict()
        
        # Получаем размер скрытого состояния из модели
        hidden_size = self.base_model.config.hidden_size
        
        # Создаем адаптер для каждого выбранного слоя
        for layer_idx in self.layers_to_adapt:
            adapter_name = f"adapter_layer_{layer_idx}"
            self.adapters[adapter_name] = MLPAdapter(
                input_dim=hidden_size,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        
        # Сохраняем исходные forward функции слоев
        self._save_original_forward_funcs()
        
        # Счетчик для подсчета обучаемых параметров
        self.trainable_params_count = sum(p.numel() for p in self.adapters.parameters() if p.requires_grad)
        logging.info(f"Всего обучаемых параметров в адаптерах: {self.trainable_params_count}")
        
    def _save_original_forward_funcs(self):
        """Сохраняет оригинальные forward функции для модифицируемых слоев"""
        self.original_forward_funcs = {}
        for layer_idx in self.layers_to_adapt:
            layer = self.base_model.model.layers[layer_idx]
            self.original_forward_funcs[layer_idx] = layer.forward
            
            # Перезаписываем метод forward для слоя
            def make_new_forward(layer_idx, old_forward):
                def new_forward(*args, **kwargs):
                    # Сначала вызываем оригинальную forward функцию
                    outputs = old_forward(*args, **kwargs)
                    
                    # Если outputs - это кортеж, то адаптируем первый элемент
                    if isinstance(outputs, tuple):
                        adapted_hidden_states = self.adapters[f"adapter_layer_{layer_idx}"](outputs[0])
                        return (adapted_hidden_states,) + outputs[1:]
                    # Иначе просто применяем адаптер к outputs
                    else:
                        return self.adapters[f"adapter_layer_{layer_idx}"](outputs)
                
                return new_forward
            
            # Присваиваем новую forward функцию слою
            layer.forward = make_new_forward(layer_idx, self.original_forward_funcs[layer_idx])
    
    def restore_original_forward_funcs(self):
        """Восстанавливает оригинальные forward функции для модифицированных слоев"""
        for layer_idx, forward_func in self.original_forward_funcs.items():
            layer = self.base_model.model.layers[layer_idx]
            layer.forward = forward_func
    
    def forward(self, *args, **kwargs):
        """Прямой проход модели с MLP-адаптерами"""
        # Просто пропускаем все аргументы через базовую модель,
        # адаптеры автоматически применяются в переопределенных forward функциях слоев
        return self.base_model(*args, **kwargs)
    
    def __del__(self):
        """Деструктор для восстановления оригинальных forward функций"""
        if hasattr(self, 'original_forward_funcs'):
            self.restore_original_forward_funcs()
    
    # Добавляем методы для поддержки gradient_checkpointing
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Включает gradient checkpointing для базовой модели"""
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            # Безопасная передача параметров - проверяем поддержку use_reentrant
            if gradient_checkpointing_kwargs is not None:
                # Создаем копию словаря, чтобы не модифицировать оригинал
                safe_kwargs = gradient_checkpointing_kwargs.copy() if gradient_checkpointing_kwargs else {}
                
                # Проверяем сигнатуру метода в базовой модели
                import inspect
                if 'use_reentrant' in safe_kwargs:
                    sig = inspect.signature(self.base_model.gradient_checkpointing_enable)
                    if 'use_reentrant' not in sig.parameters:
                        # Удаляем неподдерживаемый параметр
                        logging.info("Параметр 'use_reentrant' не поддерживается версией transformers, удаляем его")
                        safe_kwargs.pop('use_reentrant', None)
                
                # Вызываем с безопасными параметрами
                self.base_model.gradient_checkpointing_enable(**safe_kwargs)
            else:
                self.base_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Выключает gradient checkpointing для базовой модели"""
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()
    
    # Проксирование методов конфигурации для совместимости с Trainer
    @property
    def config(self):
        """Возвращает конфигурацию базовой модели"""
        return self.base_model.config
    
    def generate(self, *args, **kwargs):
        """Делегирует вызов методу generate базовой модели"""
        return self.base_model.generate(*args, **kwargs)
    
    def save_adapters(self, path):
        """Сохраняет MLP-адаптеры (веса и конфигурацию)"""
        os.makedirs(path, exist_ok=True)
        
        try:
            # Сохраняем веса адаптеров в правильном формате
            adapter_state = {}
            for name, adapter in self.adapters.items():
                adapter_state[name] = adapter.state_dict()
            
            # Сохраняем состояние адаптеров
            torch.save(adapter_state, os.path.join(path, "mlp_adapters.pt"))
            
            # Сохраняем конфигурацию адаптеров
            config = {
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "layers_to_adapt": self.layers_to_adapt,
                "trainable_params": self.trainable_params_count,
                "model_type": self.base_model.__class__.__name__,
                "base_model_name": getattr(self.base_model, "name_or_path", "unknown"),
            }
            
            with open(os.path.join(path, "mlp_adapters_config.json"), "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logging.info(f"MLP-адаптеры сохранены в {path}")
            return True
        except Exception as e:
            logging.error(f"Ошибка при сохранении адаптеров: {e}")
            logging.error(traceback.format_exc())
            return False
    
    def load_adapters(self, path):
        """Загружает адаптеры из указанного пути"""
        if not os.path.exists(path):
            logging.error(f"Путь {path} не существует")
            return False
        
        try:
            # Определяем устройство и тип данных базовой модели
            base_model_device = next(self.base_model.parameters()).device
            base_model_dtype = next(self.base_model.parameters()).dtype
            
            logging.info(f"Загружаем адаптеры на устройство: {base_model_device}, тип данных: {base_model_dtype}")
            
            # Проверяем формат файла адаптеров (поддержка старого и нового форматов)
            new_format_path = os.path.join(path, "mlp_adapters.pt")
            old_format_path = os.path.join(path, "mlp_adapters.bin")
            
            if os.path.exists(new_format_path):
                # Новый формат - словарь состояний для каждого адаптера
                adapter_state = torch.load(new_format_path, map_location=base_model_device)
                logging.info("Загружаем адаптеры в новом формате (.pt)")
                
                # Загружаем параметры для всех адаптеров
                for key, state_dict in adapter_state.items():
                    if key in self.adapters:
                        # Преобразуем веса адаптера в тип данных базовой модели
                        for param_key, param in state_dict.items():
                            state_dict[param_key] = param.to(dtype=base_model_dtype)
                        
                        # Загружаем состояние
                        self.adapters[key].load_state_dict(state_dict)
                        
                        # Дополнительно проверяем, что адаптер на правильном устройстве и имеет правильный тип данных
                        self.adapters[key] = self.adapters[key].to(device=base_model_device, dtype=base_model_dtype)
                    else:
                        logging.warning(f"Адаптер {key} не найден в текущей модели")
                    
            elif os.path.exists(old_format_path):
                # Старый формат - единый state_dict для всех адаптеров
                logging.info("Загружаем адаптеры в старом формате (.bin)")
                adapter_state_dict = torch.load(old_format_path, map_location=base_model_device)
                
                # Преобразуем веса в нужный тип данных
                for key, param in adapter_state_dict.items():
                    adapter_state_dict[key] = param.to(dtype=base_model_dtype)
                    
                # Загружаем state_dict
                self.adapters.to(device=base_model_device, dtype=base_model_dtype)
                self.adapters.load_state_dict(adapter_state_dict)
            else:
                logging.error(f"Файл с весами адаптеров не найден ни в новом (.pt), ни в старом (.bin) формате")
                return False
            
            # Проверка типов данных после загрузки
            adapter_param = next(self.adapters.parameters())
            if adapter_param.dtype != base_model_dtype:
                logging.warning(f"Типы данных не совпадают: модель {base_model_dtype}, адаптеры {adapter_param.dtype}")
                # Принудительно перемещаем все адаптеры на нужный тип данных
                self.adapters.to(dtype=base_model_dtype)
                logging.info(f"Адаптеры принудительно приведены к типу {base_model_dtype}")
            
            # Проверка устройства после загрузки
            if adapter_param.device != base_model_device:
                logging.warning(f"Устройства не совпадают: модель {base_model_device}, адаптеры {adapter_param.device}")
                # Принудительно перемещаем все адаптеры на нужное устройство
                self.adapters.to(device=base_model_device)
                logging.info(f"Адаптеры принудительно перемещены на устройство {base_model_device}")
            
            logging.info(f"MLP-адаптеры загружены из {path}")
            return True
        except Exception as e:
            logging.error(f"Ошибка при загрузке адаптеров: {e}")
            logging.error(traceback.format_exc())
            return False

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

# Функция для выбора GPU с наибольшим количеством свободной памяти
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
        use_cache=False,  # Отключаем кэширование ключей/значений для обучения
    )
    
    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Настройка PAD-токена для токенизатора
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logging.info("Модель и токенизатор успешно загружены")
    
    return model, tokenizer

# Создаем модель с MLP-адаптерами
def create_mlp_adapter_model(base_model, hidden_dim=128, dropout=0.1, num_layers=4, layers_to_adapt=None):
    """Создание модели с MLP-адаптерами"""
    logging.info("Создаем модель с MLP-адаптерами...")
    
    # Определяем тип данных и устройство базовой модели
    base_model_dtype = next(base_model.parameters()).dtype
    base_model_device = next(base_model.parameters()).device
    logging.info(f"Базовая модель использует dtype: {base_model_dtype}, устройство: {base_model_device}")
    
    # Замораживаем параметры базовой модели
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Создаем модель с MLP-адаптерами
    model = MLPAdapterModel(
        base_model=base_model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
        layers_to_adapt=layers_to_adapt
    )
    
    # Переносим адаптеры на правильное устройство и с правильным типом данных
    model.adapters.to(device=base_model_device, dtype=base_model_dtype)
    
    # Явно устанавливаем requires_grad=True для параметров адаптеров
    for name, param in model.named_parameters():
        if "adapters" in name:
            param.requires_grad = True
    
    # Проверяем, что адаптеры имеют правильный тип данных
    adapter_param = next(iter(model.adapters.parameters()))
    if adapter_param.dtype != base_model_dtype:
        logging.warning(f"Типы данных не совпадают: модель {base_model_dtype}, адаптеры {adapter_param.dtype}")
        model.adapters.to(dtype=base_model_dtype)
        logging.info(f"Адаптеры принудительно приведены к типу {base_model_dtype}")
    
    # Выводим информацию о модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Всего параметров: {total_params}")
    logging.info(f"Обучаемых параметров: {trainable_params}")
    logging.info(f"Процент обучаемых параметров: {trainable_params / total_params * 100:.4f}%")
    
    return model

# Функция для обучения модели с MLP-адаптерами
def train_mlp_adapter(
    train_dataset,
    val_dataset,
    model,
    tokenizer,
    output_dir=os.path.join(PROJECT_ROOT, "models", "mlp"),
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
    """Обучение модели с MLP-адаптерами"""
    
    # Инициализируем wandb для отслеживания процесса обучения
    if use_wandb:
        wandb.init(
            project="llama-mlp-adapters",
            name=f"mlp-dim{model.hidden_dim}-layers{len(model.layers_to_adapt)}",
            config={
                "model": BASE_MODEL,
                "hidden_dim": model.hidden_dim,
                "dropout": model.dropout,
                "layers_to_adapt": model.layers_to_adapt,
                "epochs": num_train_epochs,
                "batch_size": per_device_train_batch_size,
                "learning_rate": learning_rate
            }
        )
    
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
        # Отключаем автоматический выбор лучшей модели, так как eval_loss не вычисляется
        load_best_model_at_end=False,
        report_to="wandb" if use_wandb else "none",
        warmup_ratio=warmup_ratio,
        remove_unused_columns=False,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
        run_name=f"mlp-dim{model.hidden_dim}" if use_wandb else None,
        # Параметры для стабильности
        local_rank=-1,
        dataloader_num_workers=4,
        no_cuda=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,  # Отключаем для предотвращения конфликтов с нашими адаптерами
        torch_compile=False,  # Отключаем для снижения риска ошибок
        deepspeed=None,  # Явно отключаем DeepSpeed
        auto_find_batch_size=False,  # Отключаем автоматический поиск размера батча
    )
    
    # Создаем тренера
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
    
    # Сохраняем адаптеры
    logging.info("Сохраняем обученные MLP-адаптеры...")
    model.save_adapters(output_dir)
    
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

# Функция для загрузки модели с MLP-адаптерами
def load_model_with_mlp_adapters(adapters_path):
    """Загрузка модели с MLP-адаптерами"""
    logging.info("Загружаем модель с MLP-адаптерами...")
    
    try:
        # Загружаем базовую модель
        model, tokenizer = load_model_for_training()
        
        # Определяем тип данных и устройство базовой модели
        base_model_dtype = next(model.parameters()).dtype
        base_model_device = next(model.parameters()).device
        logging.info(f"Базовая модель использует dtype: {base_model_dtype}, устройство: {base_model_device}")
        
        # Проверяем наличие конфигурации адаптеров
        config_path = os.path.join(adapters_path, "mlp_adapters_config.json")
        if not os.path.exists(config_path):
            logging.error(f"Файл конфигурации адаптеров не найден: {config_path}")
            return None, None
        
        # Загружаем конфигурацию адаптеров
        with open(config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        
        # Создаем модель с MLP-адаптерами
        mlp_model = MLPAdapterModel(
            base_model=model,
            hidden_dim=adapter_config["hidden_dim"],
            dropout=adapter_config.get("dropout", 0.1),
            layers_to_adapt=adapter_config["layers_to_adapt"]
        )
        
        # Убедимся, что модель находится на выбранном устройстве и с правильным типом данных
        mlp_model.to(device=base_model_device, dtype=base_model_dtype)
        
        # Загружаем веса адаптеров
        success = mlp_model.load_adapters(adapters_path)
        if not success:
            logging.error("Ошибка при загрузке адаптеров")
            return None, None
        
        # Проверяем, что типы данных адаптеров соответствуют типу базовой модели
        adapter_param = next(iter(mlp_model.adapters.parameters()))
        if adapter_param.dtype != base_model_dtype:
            logging.warning(f"После загрузки типы данных не совпадают: модель {base_model_dtype}, адаптеры {adapter_param.dtype}")
            mlp_model.adapters.to(dtype=base_model_dtype)
            logging.info(f"Адаптеры принудительно приведены к типу {base_model_dtype}")
        
        # Проверяем, что устройства модели и адаптеров совпадают
        model_device = next(model.parameters()).device
        adapter_device = next(mlp_model.adapters.parameters()).device
        if model_device != adapter_device:
            logging.warning(f"Устройства модели и адаптеров не совпадают! Модель: {model_device}, адаптеры: {adapter_device}")
            # Принудительно перемещаем адаптеры на устройство модели
            mlp_model.adapters.to(model_device)
            logging.info(f"Адаптеры принудительно перемещены на устройство {model_device}")
        
        # Переводим модель в режим оценки
        mlp_model.eval()
        
        logging.info(f"Модель с MLP-адаптерами успешно загружена из {adapters_path}")
        
        return mlp_model, tokenizer
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели с адаптерами: {e}")
        logging.error(traceback.format_exc())
        return None, None

# Функция для генерации ответа с использованием MLP-адаптеров
def generate_with_mlp(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
):
    """Генерация ответа с использованием модели с MLP-адаптерами"""
    # Замеряем время
    start_time = time.time()
    
    # Формируем инструкцию для модели
    if "Инструкция:" not in prompt and "инструкция:" not in prompt.lower():
        instruction_prompt = f"Инструкция: {prompt}\n\nОтвет:"
    else:
        instruction_prompt = prompt
    
    logging.info(f"Генерируем ответ на запрос: {prompt[:100]}...")
    
    # Токенизируем входные данные
    inputs = tokenizer(instruction_prompt, return_tensors="pt").to(DEVICE)
    
    # Генерируем ответ
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # Штраф за повторения
            no_repeat_ngram_size=3   # Запрещаем повторять n-граммы
        )
    
    # Декодируем только сгенерированную часть (ответ)
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Измеряем время и токены
    time_taken = time.time() - start_time
    tokens_generated = len(generated_tokens)
    
    logging.info(f"Сгенерирован ответ: {response[:100]}...")
    logging.info(f"Время генерации: {time_taken:.2f} сек")
    logging.info(f"Сгенерировано токенов: {tokens_generated}")
    
    return response, time_taken, tokens_generated

# Функция для оценки модели на датасете Nerus
def evaluate_model_on_nerus(model, tokenizer, device, adapter_name=None):
    """Оценка модели на датасете Nerus"""
    from src.models.evaluate_base import evaluate_nerus
    
    logging.info(f"Оцениваем модель на датасете Nerus...")
    
    # Сохраняем текущую конфигурацию
    metrics = evaluate_nerus(model, tokenizer, device)
    
    return metrics

# Функция для оценки модели на датасете TERRa
def evaluate_model_on_terra(model, tokenizer, device, adapter_name=None):
    """Оценка модели на датасете TERRa"""
    from src.models.evaluate_base import evaluate_terra
    
    logging.info(f"Оцениваем модель на датасете TERRa...")
    
    # Оцениваем модель
    metrics = evaluate_terra(model, tokenizer, device)
    
    return metrics

# Функция для сравнения с базовой моделью
def compare_with_base_model(base_metrics, mlp_metrics):
    """Сравнение метрик базовой модели и модели с MLP-адаптерами"""
    logging.info("Сравнение метрик базовой модели и модели с MLP-адаптерами:")
    
    # Сравниваем метрики для TERRa
    logging.info("\nСравнение метрик для TERRa:")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        if metric in base_metrics["terra"] and metric in mlp_metrics["terra"]:
            change = mlp_metrics["terra"][metric] - base_metrics["terra"][metric]
            change_percent = (change / base_metrics["terra"][metric]) * 100 if base_metrics["terra"][metric] != 0 else float('inf')
            logging.info(f"{metric}: {base_metrics['terra'][metric]:.4f} -> {mlp_metrics['terra'][metric]:.4f} ({change_percent:+.2f}%)")
    
    # Сравниваем метрики для Nerus
    logging.info("\nСравнение метрик для Nerus:")
    for metric in ["precision", "recall", "f1"]:
        if metric in base_metrics["nerus"] and metric in mlp_metrics["nerus"]:
            change = mlp_metrics["nerus"][metric] - base_metrics["nerus"][metric]
            change_percent = (change / base_metrics["nerus"][metric]) * 100 if base_metrics["nerus"][metric] != 0 else float('inf')
            logging.info(f"{metric}: {base_metrics['nerus'][metric]:.4f} -> {mlp_metrics['nerus'][metric]:.4f} ({change_percent:+.2f}%)")

# Основная функция
def main(
    mlp_hidden_dim=128,
    mlp_dropout=0.1,
    num_layers=4,
    batch_size=2,
    grad_accum=4,
    epochs=2,
    learning_rate=5e-5,
    train_model=True,
    evaluate_base=True,
    evaluate_mlp=True,
    test_examples=True,
    use_wandb=False
):
    """Основная функция для обучения и оценки модели с MLP-адаптерами"""
    # Определяем пути для сохранения результатов и логов
    output_dir = os.path.join(PROJECT_ROOT, "models", "mlp")
    metrics = {"base": {}, "mlp": {}}
    
    # Выбираем лучшую GPU, если доступно несколько
    device = select_best_gpu()
    
    # Оцениваем базовую модель
    if evaluate_base:
        logging.info("\n--- Оценка базовой модели ---")
        model, tokenizer = load_model_for_training()
        model.eval()  # Переводим модель в режим оценки
        
        # Оцениваем на TERRa
        terra_metrics = evaluate_model_on_terra(model, tokenizer, device)
        metrics["base"]["terra"] = terra_metrics
        
        # Оцениваем на Nerus
        nerus_metrics = evaluate_model_on_nerus(model, tokenizer, device)
        metrics["base"]["nerus"] = nerus_metrics
        
        # Сохраняем метрики базовой модели
        logging.info("Сохраняем метрики базовой модели...")
        results_dir = os.path.join(PROJECT_ROOT, "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "base_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics["base"], f, ensure_ascii=False, indent=2)
    
    # Обучаем модель с MLP-адаптерами
    if train_model:
        logging.info("\n--- Обучение модели с MLP-адаптерами ---")
        
        # Загружаем данные
        train_dataset, val_dataset = load_nerus_data()
        if train_dataset is None or val_dataset is None:
            logging.error("Не удалось загрузить данные. Обучение невозможно.")
            return None
        
        # Загружаем базовую модель
        base_model, tokenizer = load_model_for_training()
        
        # Создаем модель с MLP-адаптерами
        mlp_model = create_mlp_adapter_model(
            base_model=base_model,
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout,
            num_layers=num_layers
        )
        
        # Подготавливаем данные для обучения
        train_dataset = prepare_dataset_for_training(train_dataset, tokenizer)
        val_dataset = prepare_dataset_for_training(val_dataset, tokenizer)
        
        # Обучаем модель
        mlp_model, tokenizer, train_metrics = train_mlp_adapter(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=mlp_model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            use_wandb=use_wandb
        )
    
    # Оцениваем модель с MLP-адаптерами
    if evaluate_mlp:
        logging.info("\n--- Оценка модели с MLP-адаптерами ---")
        
        # Загружаем модель с MLP-адаптерами, если нужно
        if not train_model or "mlp_model" not in locals():
            mlp_model, tokenizer = load_model_with_mlp_adapters(output_dir)
        
        if mlp_model is None:
            logging.error("Не удалось загрузить модель с MLP-адаптерами. Оценка невозможна.")
            return metrics
        
        # Оцениваем на TERRa
        terra_metrics = evaluate_model_on_terra(mlp_model, tokenizer, device)
        metrics["mlp"]["terra"] = terra_metrics
        
        # Оцениваем на Nerus
        nerus_metrics = evaluate_model_on_nerus(mlp_model, tokenizer, device)
        metrics["mlp"]["nerus"] = nerus_metrics
        
        # Сохраняем метрики модели с MLP-адаптерами
        logging.info("Сохраняем метрики модели с MLP-адаптерами...")
        with open(os.path.join(output_dir, "mlp_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics["mlp"], f, ensure_ascii=False, indent=2)
        
        # Сравниваем с базовой моделью
        if evaluate_base:
            compare_with_base_model(metrics["base"], metrics["mlp"])
    
    # Тестируем модель на примерах
    if test_examples and (train_model or evaluate_mlp):
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
            if "model" not in locals() or "model" in locals() and hasattr(model, "adapters"):
                model, tokenizer = load_model_for_training()
            
            for idx, example in enumerate(test_examples):
                logging.info(f"\nПример {idx+1}: {example}")
                response, time_taken, tokens = generate_with_mlp(model, tokenizer, 
                    f"Найди и классифицируй все именованные сущности в тексте, указав их тип (PER - персона, LOC - локация, ORG - организация): {example}")
                logging.info(f"Ответ базовой модели: {response}")
                logging.info(f"Время: {time_taken:.2f} сек, Токенов: {tokens}")
        
        # Тестируем с MLP-адаптерами
        logging.info("\nМодель с MLP-адаптерами:")
        # Загружаем модель с MLP-адаптерами, если нужно
        if "mlp_model" not in locals():
            mlp_model, tokenizer = load_model_with_mlp_adapters(output_dir)
        
        for idx, example in enumerate(test_examples):
            logging.info(f"\nПример {idx+1}: {example}")
            response, time_taken, tokens = generate_with_mlp(mlp_model, tokenizer, 
                f"Найди и классифицируй все именованные сущности в тексте, указав их тип (PER - персона, LOC - локация, ORG - организация): {example}")
            logging.info(f"Ответ модели с MLP-адаптерами: {response}")
            logging.info(f"Время: {time_taken:.2f} сек, Токенов: {tokens}")
    
    logging.info("\nРабота скрипта завершена!")
    return metrics


if __name__ == "__main__":
    import argparse
    
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description="Обучение и оценка модели с MLP-адаптерами")
    
    # Параметры MLP-адаптеров
    parser.add_argument("--hidden-dim", type=int, default=128, help="Размер скрытого слоя MLP-адаптера")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout для MLP-адаптера")
    parser.add_argument("--num-layers", type=int, default=4, help="Количество слоев с MLP-адаптерами")
    
    # Параметры обучения
    parser.add_argument("--batch-size", type=int, default=2, help="Размер батча для обучения")
    parser.add_argument("--grad-accum", type=int, default=4, help="Количество шагов для накопления градиента")
    parser.add_argument("--epochs", type=int, default=2, help="Количество эпох обучения")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Скорость обучения")
    
    # Флаги для управления процессом
    parser.add_argument("--train", action="store_true", help="Обучать модель")
    parser.add_argument("--evaluate-base", action="store_true", help="Оценить базовую модель")
    parser.add_argument("--evaluate-mlp", action="store_true", help="Оценить модель с MLP-адаптерами")
    parser.add_argument("--test-examples", action="store_true", help="Тестировать на примерах")
    parser.add_argument("--use-wandb", action="store_true", help="Использовать WandB для логирования")
    
    args = parser.parse_args()
    
    # Запускаем основную функцию с аргументами
    main(
        mlp_hidden_dim=args.hidden_dim,
        mlp_dropout=args.dropout,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_model=args.train,
        evaluate_base=args.evaluate_base,
        evaluate_mlp=args.evaluate_mlp,
        test_examples=args.test_examples,
        use_wandb=args.use_wandb
    ) 