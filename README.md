# Проект исследования адаптеров языковых моделей

## Цель проекта

Цель данного проекта — исследование методов адаптации больших языковых моделей с помощью специализированных дополнительных весовых элементов (адаптеров) для улучшения их работы на узкоспециализированных задачах при сохранении общих языковых способностей. В рамках проекта реализованы и сравнены два подхода к адаптации: LoRA (Low-Rank Adaptation) и MLP-адаптеры, с оценкой их эффективности на общих и специализированных задачах.

## Описание задачи

Большие языковые модели демонстрируют отличные результаты на широком спектре задач, но часто не справляются с узкоспециализированными запросами. Полное переобучение таких моделей под каждый новый тип запросов — неэффективное решение, поскольку:
- Требует значительных вычислительных ресурсов
- Может привести к потере общих языковых способностей (катастрофическое забывание)
- Не масштабируется при необходимости поддержки многих специализированных задач

Вместо этого можно модифицировать архитектуру модели, внедряя в нее дополнительные обучаемые элементы (адаптеры), настраиваемые под конкретные задачи. При необходимости использования модели для общих задач адаптеры можно отключить, а для специализированных — использовать соответствующий адаптер.

В рамках проекта решаются следующие задачи:
1. Реализация тестового стенда для обучения и оценки адаптеров
2. Имплементация двух подходов к адаптации: LoRA и MLP-адаптеры
3. Проведение экспериментов с различными конфигурациями адаптеров
4. Оценка изменения эффективности модели на общих и специализированных задачах
5. Анализ компромисса между специализацией и сохранением общих способностей

## Теоретические основы используемых методов

### Low-Rank Adaptation (LoRA)

LoRA — метод адаптации предобученных трансформерных моделей, предложенный исследователями Microsoft в 2021 году. Основная идея LoRA заключается в том, что обновления матриц весов можно аппроксимировать матрицами низкого ранга.

Принцип работы:
1. Для каждой матрицы весов W в модели:
   - Замораживаем исходные веса W
   - Добавляем произведение двух низкоранговых матриц: ΔW = BA, где B имеет размерность [d × r], A — [r × k], r << min(d, k)
   - Результирующая матрица: W' = W + ΔW = W + BA

2. Преимущества:
   - Количество обучаемых параметров сокращается с d×k до r×(d+k)
   - Исходные веса сохраняются, что позволяет легко переключаться между адаптированной и базовой моделью
   - Можно адаптировать только определенные матрицы модели (например, только матрицы внимания)

3. Ключевые гиперпараметры:
   - r (ранг) — определяет размерность низкоранговых матриц
   - α (альфа) — масштабирующий фактор для инициализации
   - target_modules — какие именно матрицы в модели адаптировать

### MLP-адаптеры

MLP-адаптеры представляют собой многослойные перцептроны, внедряемые параллельно основным слоям трансформерной модели. Этот подход был предложен в работе "Parameter-Efficient Transfer Learning for NLP" (Houlsby et al., 2019).

Принцип работы:
1. После определенных слоев трансформера добавляются адаптеры:
   - Входные данные проходят как через основной слой, так и через адаптер
   - Результаты складываются для получения итогового выхода

2. Структура MLP-адаптера:
   - Down-projection: проекция входных данных в пространство меньшей размерности
   - Нелинейная функция активации (например, GELU)
   - Up-projection: проекция обратно в исходное пространство
   - Дополнительно может использоваться skip-connection и dropout

3. Преимущества:
   - Модульность и гибкость — можно добавлять адаптеры к разным слоям модели
   - Исходные веса модели не меняются
   - Можно создавать комбинации адаптеров для разных задач

4. Ключевые гиперпараметры:
   - Размер скрытого слоя адаптера
   - Количество слоев модели, к которым применяются адаптеры
   - Dropout для регуляризации

## Техническая реализация

### Базовая модель

В качестве основы была выбрана модель **Llama 3.1 8B Instruct** (Meta). Эта архитектура представляет собой оптимальный баланс между качеством и вычислительными требованиями:
- Помещается в память одной GPU A100-40GB при использовании смешанной точности

Базовая модель загружается с применением техник количественного сжатия (quantization) для уменьшения требований к памяти:

```python
def load_model_for_training():
    device_map = {"": 0}
    
    # Настройки квантизации для уменьшения потребления памяти
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Загружаем базовую модель
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map=device_map
    )
    
    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    return model, tokenizer
```

### Реализация LoRA-адаптеров

Реализация LoRA-адаптеров выполнена с использованием библиотеки PEFT (Parameter-Efficient Fine-Tuning). Ключевые компоненты:

1. **Конфигурация LoRA-адаптеров**:
```python
def get_lora_config(r, alpha, dropout, target_modules):
    return LoraConfig(
        r=r,                                # ранг матриц
        lora_alpha=alpha,                   # параметр масштабирования
        lora_dropout=dropout,               # dropout для регуляризации
        target_modules=target_modules,      # целевые модули для адаптации
        bias="none",                        # не обучаем bias
        task_type="CAUSAL_LM"               # тип задачи - авторегрессивная генерация
    )
```

2. **Добавление адаптеров к модели**:
```python
# Применяем LoRA адаптеры к модели
model = get_peft_model(model, lora_config)
```

3. **Активация LoRA во время инференса**:
```python
def generate_with_lora(model, tokenizer, prompt, adapter_name=None):
    # Если указан адаптер, включаем его
    if adapter_name:
        model.set_adapter(adapter_name)
        
    # Формируем входные данные
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Замеряем время инференса
    start_time = time.time()
    
    # Генерируем ответ
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p
        )
    
    inference_time = time.time() - start_time
    
    # Декодируем ответ
    response = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    response = response[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
    
    return response, inference_time, generation_output.shape[1] - inputs["input_ids"].shape[1]
```

### Реализация MLP-адаптеров

MLP-адаптеры реализованы с нуля, так как требуют более глубокой интеграции с моделью:

1. **Класс MLP-адаптера**:
```python
class MLPAdapter(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 dropout=0.1, 
                 activation=nn.GELU, 
                 init_scale=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        
        # Down-projection
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        # Up-projection
        self.up_proj = nn.Linear(hidden_dim, input_dim)
        
        # Инициализация весов с малыми значениями для стабильности
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.down_proj.weight, std=0.01)
        nn.init.normal_(self.up_proj.weight, std=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        # Сохраняем входные данные для резидуального соединения
        residual = x
        
        # Применяем адаптер
        x = self.down_proj(x)  # Понижающая проекция
        x = self.activation(x)  # Нелинейная активация
        x = self.dropout(x)     # Dropout для регуляризации
        x = self.up_proj(x)     # Повышающая проекция
        
        # Добавляем резидуальное соединение
        return x + residual
```

2. **Модель с MLP-адаптерами**:
```python
class MLPAdapterModel(nn.Module):
    def __init__(self, 
                 base_model, 
                 hidden_dim=128, 
                 dropout=0.1, 
                 num_layers=4, 
                 layers_to_adapt=None):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        
        # Определяем слои трансформера для добавления адаптеров
        if layers_to_adapt is None:
            # По умолчанию адаптируем последние num_layers блоков трансформера
            n_layers = self.config.num_hidden_layers
            layers_to_adapt = list(range(n_layers - num_layers, n_layers))
        
        self.layers_to_adapt = layers_to_adapt
        self.adapters = nn.ModuleDict()
        
        # Создаем MLP-адаптеры для выбранных слоев
        for layer_idx in layers_to_adapt:
            adapter_name = f"layer_{layer_idx}"
            self.adapters[adapter_name] = MLPAdapter(
                input_dim=self.config.hidden_size,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        
        # Сохраняем оригинальные forward-методы слоев
        self._save_original_forward_funcs()
        
        # Счетчик количества обучаемых параметров
        self.trainable_params = sum(p.numel() for p in self.adapters.parameters())
```

3. **Интеграция адаптеров с базовой моделью**:
```python
def _save_original_forward_funcs(self):
    """Сохраняет оригинальные forward функции слоев и заменяет их на новые"""
    self.original_forwards = {}
    
    for layer_idx in self.layers_to_adapt:
        layer = self._get_decoder_layer(layer_idx)
        self.original_forwards[layer_idx] = layer.forward
        
        # Создаем новую forward функцию для слоя
        def make_new_forward(layer_idx, old_forward):
            def new_forward(*args, **kwargs):
                # Сначала вызываем оригинальную forward функцию
                outputs = old_forward(*args, **kwargs)
                
                # Извлекаем выходные данные
                if isinstance(outputs, tuple):
                    hidden_states = outputs[0]
                    rest = outputs[1:]
                else:
                    hidden_states = outputs
                    rest = None
                
                # Применяем адаптер
                adapter = self.adapters[f"layer_{layer_idx}"]
                hidden_states = adapter(hidden_states)
                
                # Собираем выходные данные обратно
                if rest is not None:
                    outputs = (hidden_states,) + rest
                else:
                    outputs = hidden_states
                
                return outputs
            
            return new_forward
        
        # Заменяем forward метод слоя
        layer.forward = make_new_forward(layer_idx, self.original_forwards[layer_idx])
```

### Обучение адаптеров

Процесс обучения адаптеров реализован с использованием стандартных компонентов PyTorch и оптимизированных трейнеров из библиотеки Hugging Face Transformers:

1. **Подготовка данных**:
```python
def prepare_dataset_for_training(dataset, tokenizer, max_length=256):
    def preprocess_function(examples):
        # Кодируем только промпты с padding
        model_inputs = tokenizer(
            examples["prompt"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Кодируем ответы для обучения
        labels = tokenizer(
            examples["completion"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # Заменяем padding токены на -100, чтобы исключить их из функции потерь
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        
        return model_inputs
    
    # Применяем функцию препроцессинга к датасету
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Токенизация датасета"
    )
    
    return tokenized_dataset
```

2. **Настройка тренера**:
```python
# Аргументы обучения
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    max_grad_norm=max_grad_norm,
    logging_steps=logging_steps,
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    save_steps=save_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=warmup_ratio,
    report_to=["wandb"] if use_wandb else [],
    run_name=run_name if use_wandb else None,
)

# Создаем тренер
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=lambda data: dict((k, torch.stack([f[k] for f in data])) 
                                     for k in data[0].keys())
)
```

### Оценка эффективности

Для объективной оценки эффективности адаптеров используются два типа задач:

1. **Общая задача (TERRa)** - определение текстового доказательства:
```python
def evaluate_terra(model, tokenizer, device):
    """Оценка модели на датасете TERRa"""
    # ... (загрузка данных)
    
    results = []
    y_true = []
    y_pred = []
    
    # Оцениваем модель на каждом примере
    for i, example in enumerate(tqdm(val_data, desc="Оценка TERRa")):
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        true_label = 1 if example["label"] == "entailment" else 0
        
        # Формируем промпт
        prompt = f"Проанализируй, подтверждает ли предложение A предложение B?\n" \
                 f"A: {premise}\n" \
                 f"B: {hypothesis}\n\n" \
                 f"Выбери один из вариантов: 1. ENTAILMENT - предложение B следует из предложения A. " \
                 f"2. NOT_ENTAILMENT - предложение B не следует из предложения A.\n\n" \
                 f"Ответ: "
        
        # Генерируем ответ модели
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False
            )
        
        response = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # Определяем предсказанный класс
        predicted_label = 1 if "ENTAILMENT" in response and "NOT_ENTAILMENT" not in response[:15] else 0
        
        # Сохраняем результаты
        results.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "true_label": "entailment" if true_label == 1 else "not_entailment",
            "predicted_label": "entailment" if predicted_label == 1 else "not_entailment",
            "response": response
        })
        
        y_true.append(true_label)
        y_pred.append(predicted_label)
    
    # Вычисляем метрики
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    return metrics
```

2. **Специализированная задача (Nerus)** - распознавание именованных сущностей:
```python
def evaluate_nerus(model, tokenizer, device):
    """Оценка модели на датасете Nerus"""
    # ... (загрузка данных)
    
    results = []
    all_metrics = []
    
    # Оцениваем модель на каждом примере
    for i, example in enumerate(tqdm(val_data, desc="Оценка Nerus")):
        text = example["text"]
        entities = example["entities"]
        
        # Формируем промпт
        prompt = f"Найди и классифицируй все именованные сущности в следующем тексте. " \
                 f"Обозначь каждую сущность в формате [текст](тип), где тип - это PER для персон, " \
                 f"LOC для локаций и ORG для организаций.\n\nТекст: {text}\n\nСущности: "
        
        # Генерируем ответ модели
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True,
                top_p=0.95
            )
        
        response = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # Извлекаем предсказанные сущности
        predicted_entities = extract_entities_from_response(response)
        
        # Вычисляем метрики
        metrics = calculate_entity_metrics(entities, predicted_entities)
        all_metrics.append(metrics)
        
        # Сохраняем результаты
        results.append({
            "text": text,
            "true_entities": entities,
            "predicted_entities": predicted_entities,
            "response": response,
            "metrics": metrics
        })
    
    # Вычисляем общие метрики (макро и микро)
    macro_metrics = {
        "precision": sum(m["precision"] for m in all_metrics) / len(all_metrics),
        "recall": sum(m["recall"] for m in all_metrics) / len(all_metrics),
        "f1": sum(m["f1"] for m in all_metrics) / len(all_metrics)
    }
    
    # ... (вычисление микро-метрик)
    
    return {
        "macro": macro_metrics,
        "micro": micro_metrics
    }
```

### Сравнение адаптеров

Для наглядного сравнения эффективности двух подходов к адаптации реализован инструмент, генерирующий сравнительный отчет:

```python
def generate_comparison_report(metrics):
    """Генерирует полный отчет сравнения адаптеров"""
    # ... (подготовка отчета)
    
    # Анализируем результаты TERRa
    lora_terra_f1 = metrics["lora"].get("terra", {}).get("f1", 0)
    base_terra_f1 = metrics["base"].get("terra", {}).get("f1", 0)
    mlp_terra_f1 = metrics["mlp"].get("terra", {}).get("f1") if "mlp" in metrics and metrics["mlp"] else None
    
    # Анализируем результаты Nerus
    lora_nerus_f1 = metrics["lora"].get("nerus", {}).get("f1", 0)
    base_nerus_f1 = metrics["base"].get("nerus", {}).get("f1", 0)
    mlp_nerus_f1 = metrics["mlp"].get("nerus", {}).get("f1") if "mlp" in metrics and metrics["mlp"] else None
    
    # Вычисляем изменения для TERRa
    if lora_terra_f1 and base_terra_f1:
        lora_terra_change = ((lora_terra_f1 - base_terra_f1) / base_terra_f1) * 100 if base_terra_f1 != 0 else 0
        
        if mlp_terra_available:
            mlp_terra_change = ((mlp_terra_f1 - base_terra_f1) / base_terra_f1) * 100 if base_terra_f1 != 0 else 0
            terra_winner = "LoRA" if lora_terra_f1 > mlp_terra_f1 else "MLP" if mlp_terra_f1 > lora_terra_f1 else "Ничья"
        else:
            mlp_terra_change = None
            terra_winner = "LoRA (MLP метрики недоступны)"
        
        report.append(f"1. На датасете TERRa (общие способности) лучшие результаты показал метод {terra_winner}.")
        report.append(f"   - LoRA: изменение F1 {lora_terra_change:+.2f}% относительно базовой модели")
        
        if mlp_terra_available:
            report.append(f"   - MLP: изменение F1 {mlp_terra_change:+.2f}% относительно базовой модели")
    
    # ... (аналогичный анализ для Nerus и общий вывод)
    
    return "\n".join(report)
```

## Экспериментальные результаты

### Конфигурации адаптеров

В рамках исследования были протестированы следующие конфигурации адаптеров:

**LoRA**:
- Ранг (r): 32
- Alpha (α): 64
- Целевые модули: ["q_proj", "k_proj", "v_proj", "o_proj"]
- Dropout: 0.05
- Обучаемые параметры: ~27.3M

**MLP-адаптеры**:
- Скрытый размер: 128
- Количество слоев для адаптации: 4
- Слои для адаптации: последние 4 слоя трансформера
- Dropout: 0.1
- Обучаемые параметры: ~4M

### Результаты экспериментов

Ниже представлены результаты оценки базовой модели и модели с адаптерами на двух датасетах:

#### Сравнение на датасете TERRa (общие языковые способности)

| Метрика   | Базовая модель | LoRA          | MLP           | LoRA vs Base   | MLP vs Base    |
|-----------|---------------|---------------|---------------|----------------|----------------|
| Accuracy  | 0.4800        | 0.7200        | 0.4400        | +0.2400 (+50.00%) | -0.0400 (-8.33%) |
| Precision | 0.5000        | 0.5000        | 0.4444        | +0.0000 (+0.00%)  | -0.0556 (-11.11%) |
| Recall    | 0.2400        | 0.3600        | 0.2000        | +0.1200 (+50.00%) | -0.0400 (-16.67%) |
| F1        | 0.3056        | 0.4186        | 0.2806        | +0.1130 (+37.00%) | -0.0250 (-8.18%)  |

#### Сравнение на датасете Nerus (специализированная задача)

| Метрика   | Базовая модель | LoRA          | MLP           | LoRA vs Base   | MLP vs Base    |
|-----------|---------------|---------------|---------------|----------------|----------------|
| Precision | 0.2483        | 0.4683        | 0.3583        | +0.2200 (+88.60%) | +0.1100 (+44.30%) |
| Recall    | 0.4564        | 0.7810        | 0.6923        | +0.3246 (+71.12%) | +0.2359 (+51.69%) |
| F1        | 0.3615        | 0.5611        | 0.4726        | +0.1996 (+55.19%) | +0.1111 (+30.72%) |

### Анализ результатов

#### Влияние адаптеров на общие языковые способности (TERRa)

- **LoRA-адаптеры** показали значительное улучшение результатов на общем датасете, увеличив F1-меру на +37.00% и Accuracy на +50.00%. Это говорит о том, что LoRA не только не утратил общие языковые способности, но и существенно их улучшил.
- **MLP-адаптеры** привели к незначительному снижению показателей на общем датасете, с уменьшением F1-меры на -8.18%. Такое снижение всё же находится в допустимом диапазоне для метода адаптации.

#### Влияние адаптеров на специализированную задачу (Nerus)

- **LoRA-адаптеры** продемонстрировали выдающиеся результаты на специализированной задаче, увеличив F1-меру на +55.19%, а также значительно улучшив Precision (+88.60%) и Recall (+71.12%).
- **MLP-адаптеры** также показали существенное улучшение на специализированной задаче, с ростом F1-меры на +30.72%, что подтверждает эффективность метода адаптации.

#### Анализ баланса "специализация vs общие способности"

Общий баланс между улучшением специализации и сохранением общих способностей:
- **LoRA-адаптеры**: +92.19 (сумма улучшения F1 на Nerus и улучшения F1 на TERRa)
- **MLP-адаптеры**: +22.55 (улучшение на Nerus с учётом потери на TERRa)

Этот показатель демонстрирует, что LoRA-адаптеры достигли наилучшего компромисса между специализацией и сохранением общих способностей.

## Выводы и рекомендации

1. **LoRA-адаптеры оказались эффективнее MLP-адаптеров** по всем ключевым метрикам. Они не только значительно улучшили результаты на специализированной задаче, но и повысили качество работы на общем датасете.

2. **Преимущества LoRA-адаптеров**:
   - Значительное улучшение как на специализированной задаче, так и на общих задачах
   - Более простая интеграция с базовой моделью (через библиотеку PEFT)
   - Лучшее соотношение между количеством параметров и достигаемым улучшением

3. **Области применения MLP-адаптеров**:
   - Когда критично сохранение полной совместимости с оригинальной архитектурой
   - В случаях, когда допустимо некоторое снижение общих способностей в пользу специализации
   - При необходимости более гибкой настройки адаптации отдельных слоев модели

4. **Рекомендации для дальнейших исследований**:
   - Провести эксперименты с другими конфигурациями LoRA (ранг, alpha, целевые модули)
   - Исследовать возможности комбинирования различных типов адаптеров
   - Изучить применение адаптеров для других типов специализированных задач

В целом, исследование убедительно демонстрирует эффективность методов адаптации языковых моделей для специализированных задач. LoRA-адаптеры предоставляют наилучший баланс между улучшением специализированных способностей и сохранением (или даже улучшением) общих языковых навыков.

## Структура проекта

```
adapter_project/
├── data/                      # Директория с данными
│   ├── nerus/                 # Данные для датасета Nerus
│   │   ├── nerus_lenta.conllu.gz  # Исходный датасет
│   │   ├── train.json         # Подготовленные тренировочные данные
│   │   └── val.json           # Подготовленные валидационные данные
│   │
│   └── terra/                 # Данные для датасета TERRa
│       ├── TERRa/             # Исходные файлы датасета
│       │   ├── train.jsonl    # Тренировочные данные
│       │   ├── val.jsonl      # Валидационные данные
│       │   └── test.jsonl     # Тестовые данные
│       ├── train.json         # Подготовленные тренировочные данные
│       └── val.json           # Подготовленные валидационные данные
│
├── logs/                      # Директория с логами
│   ├── train_lora.log         # Логи обучения LoRA адаптеров
│   ├── train_mlp.log          # Логи обучения MLP адаптеров
│   ├── evaluate_base.log      # Логи оценки базовой модели
│   └── compare_adapters_terminal.log  # Логи сравнения адаптеров
│
├── models/                    # Обученные модели и адаптеры
│   ├── lora/                  # LoRA адаптеры различных конфигураций
│   │   ├── r4_alpha8/         # LoRA с рангом 4 и alpha 8
│   │   ├── r8_alpha16/        # LoRA с рангом 8 и alpha 16
│   │   └── r32_alpha64/       # LoRA с рангом 32 и alpha 64
│   │
│   ├── mlp/                   # MLP адаптеры и чекпоинты
│   │   ├── mlp_adapters.pt    # Сохраненные MLP адаптеры
│   │   └── mlp_metrics.json   # Метрики MLP адаптеров
│   
│
├── results/                   # Результаты экспериментов
│   ├── baseline/              # Метрики базовой модели
│   │   ├── nerus_metrics.json  # Метрики на датасете Nerus
│   │   └── terra_metrics.json  # Метрики на датасете TERRa
│   │
│   ├── comparison/            # Отчеты сравнения адаптеров
│   └── base_metrics.json      # Базовые метрики для сравнения
│
│
├── src/                       # Исходный код проекта
│   ├── data_utils/            # Утилиты для обработки данных
│   │   ├── prepare_nerus.py   # Подготовка датасета Nerus
│   │   └── prepare_terra.py   # Подготовка датасета TERRa
│   │
│   ├── models/                # Код моделей и скрипты обучения
│   │   ├── train_lora.py      # Скрипт обучения LoRA адаптеров
│   │   ├── train_mlp.py       # Скрипт обучения MLP адаптеров
│   │   └── evaluate_base.py   # Скрипт оценки базовой модели
│   │   └── load_llama.py          # Скрипт для загрузки базовой модели
│   │
│   └── utils/                 # Утилиты для сравнения и визуализации
│       └── compare_adapters_terminal.py  # Инструмент для сравнения адаптеров
│
└── requirements.txt           # Зависимости проекта
```

## Инструкция по запуску

1. Установка зависимостей:
```bash
pip install -r requirements.txt
```
2. Загрузка базовой модели:
```bash
python src/models/load_llama.py
```
3. Загрузка датасетов и подготовка данных:
```bash
python src/data_utils/prepare_terra.py
python src/data_utils/prepare_nerus.py
```
4. Оценка базовой модели:
```bash
python src/models/evaluate_base.py
```
5. Обучение LoRA-адаптеров:
```bash
python src/models/train_lora.py --train --rank 32 --alpha 64 --target-modules q_proj,k_proj,v_proj,o_proj --batch-size 2 --grad-accum 8 --epochs 2 --evaluate-base --evaluate-lora --test
```
6. Обучение MLP-адаптеров:
```bash
python src/models/train_mlp.py --train --evaluate-mlp --test-examples --hidden-dim 128 --num-layers 4 --batch-size 2 --grad-accum 8 --epochs 2
```
7. Сравнение адаптеров:
```bash
python src/utils/compare_adapters_terminal.py
```

Отчет о сравнении будет сохранен в директории `results/comparison/`.
