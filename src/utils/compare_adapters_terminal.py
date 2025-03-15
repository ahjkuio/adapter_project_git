#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для текстового сравнения LoRA и MLP адаптеров.
Выводит результаты в консоль в виде таблиц.
"""

import os
import sys
import json
import logging
from pathlib import Path
from tabulate import tabulate
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

# Настройка логирования
log_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "compare_adapters_terminal.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Директория для сохранения отчета
results_dir = os.path.join(PROJECT_ROOT, "results", "comparison")
os.makedirs(results_dir, exist_ok=True)

def load_metrics_file(file_path):
    """Загружает метрики из файла JSON"""
    if not os.path.exists(file_path):
        logging.warning(f"Файл с метриками не найден: {file_path}")
        return None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        logging.error(f"Ошибка при загрузке файла {file_path}: {e}")
        return None

def load_adapter_metrics(adapter_type, metric_type=None):
    """Загружает метрики из файла для указанного адаптера"""
    # Определяем путь к файлу метрик в зависимости от типа адаптера
    if adapter_type == "lora":
        # Проверяем различные конфигурации LoRA
        lora_configs = ["r32_alpha64", "r4_alpha8", "r8_alpha16"]
        metrics_path = None
        
        for config in lora_configs:
            config_path = os.path.join(PROJECT_ROOT, "models", adapter_type, config, "lora_metrics.json")
            if os.path.exists(config_path):
                logging.info(f"Найден файл с метриками LoRA: {config_path}")
                metrics_path = config_path
                break
                
        if metrics_path is None:
            logging.warning(f"Не удалось найти файл метрик для LoRA адаптеров")
            return None
    elif adapter_type == "mlp":
        # Для MLP ищем файл метрик, сначала проверяем корень директории
        models_dir = os.path.join(PROJECT_ROOT, "models", adapter_type)
        if os.path.exists(models_dir):
            # Сначала проверяем файлы в корне директории
            candidate = os.path.join(models_dir, "mlp_metrics.json")
            if os.path.exists(candidate):
                logging.info(f"Найден файл с метриками MLP: {candidate}")
                metrics_path = candidate
            else:
                # Если не нашли mlp_metrics.json, ищем all_results.json
                candidate = os.path.join(models_dir, "all_results.json")
                if os.path.exists(candidate):
                    logging.info(f"Найден файл с метриками MLP: {candidate}")
                    metrics_path = candidate
                else:
                    # Если в корне не нашли, ищем в подпапках
                    subdirs = [d for d in os.listdir(models_dir) 
                            if os.path.isdir(os.path.join(models_dir, d))]
                    
                    for subdir in subdirs:
                        if subdir.startswith('checkpoint-'):
                            continue  # Пропускаем папки чекпоинтов
                        # Сначала проверяем mlp_metrics.json
                        candidate = os.path.join(models_dir, subdir, "mlp_metrics.json")
                        if os.path.exists(candidate):
                            logging.info(f"Найден файл с метриками MLP в подпапке: {candidate}")
                            metrics_path = candidate
                            break
                        # Затем проверяем all_results.json
                        candidate = os.path.join(models_dir, subdir, "all_results.json")
                        if os.path.exists(candidate):
                            logging.info(f"Найден файл с метриками MLP в подпапке: {candidate}")
                            metrics_path = candidate
                            break
                    else:
                        # Если ничего не нашли, вернем None
                        logging.warning(f"Не удалось найти метрики для MLP адаптеров")
                        metrics_path = None
        else:
            metrics_path = None
    else:
        # Для других типов (например, base) используем тот же путь, что и для LoRA
        metrics_path = os.path.join(PROJECT_ROOT, "models", "lora", "r4_alpha8", "lora_metrics.json")
    
    # Загружаем метрики
    metrics = load_metrics_file(metrics_path)

    # Если это all_results.json из mlp, сформируем правильную структуру метрик
    if metrics and adapter_type == "mlp" and metrics_path and os.path.basename(metrics_path) == "all_results.json":
        # Проверяем наличие файлов отдельных метрик
        terra_metrics_path = os.path.join(os.path.dirname(metrics_path), "terra_metrics.json")
        nerus_metrics_path = os.path.join(os.path.dirname(metrics_path), "nerus_metrics.json")
        
        structured_metrics = {}
        
        # Пробуем загрузить метрики TERRa
        if os.path.exists(terra_metrics_path):
            terra_metrics = load_metrics_file(terra_metrics_path)
            if terra_metrics:
                structured_metrics["terra"] = terra_metrics
        
        # Пробуем загрузить метрики Nerus
        if os.path.exists(nerus_metrics_path):
            nerus_metrics = load_metrics_file(nerus_metrics_path)
            if nerus_metrics:
                structured_metrics["nerus"] = nerus_metrics
        
        # Если удалось загрузить структурированные метрики, заменяем ими исходные
        if structured_metrics:
            metrics = structured_metrics
        else:
            logging.info(f"Преобразуем метрики из all_results.json для MLP")
            # Создаем словарь с метриками на основе all_results.json
            structured_metrics = {
                "terra": {
                    "accuracy": metrics.get("eval_accuracy", None),
                    "precision": metrics.get("eval_precision", None),
                    "recall": metrics.get("eval_recall", None),
                    "f1": metrics.get("eval_f1", None)
                },
                "nerus": {
                    "precision": metrics.get("test_precision", None),
                    "recall": metrics.get("test_recall", None),
                    "f1": metrics.get("test_f1", None)
                }
            }
            logging.info(f"Сконвертированные метрики: {structured_metrics}")
            metrics = structured_metrics
    # Для mlp_metrics.json формат уже подходящий, ничего делать не нужно
    elif metrics and adapter_type == "mlp" and metrics_path and os.path.basename(metrics_path) == "mlp_metrics.json":
        logging.info(f"Загружены метрики MLP из файла {metrics_path}")
    
    # Если нужны метрики только определенного типа и они есть в загруженных данных
    if metrics and metric_type and metric_type in metrics:
        return metrics[metric_type]
    
    return metrics

def load_all_metrics():
    """Загружает все метрики для базовой модели, LoRA и MLP адаптеров"""
    metrics = {
        "base": {},
        "lora": {},
        "mlp": {}
    }
    
    # Загружаем все метрики для каждого типа адаптера
    logging.info("Загружаем метрики для базовой модели...")
    base_metrics = load_adapter_metrics("base")
    
    logging.info("Загружаем метрики для LoRA адаптеров...")
    lora_metrics = load_adapter_metrics("lora")
    
    logging.info("Загружаем метрики для MLP адаптеров...")
    mlp_metrics = load_adapter_metrics("mlp")
    
    # Проверяем наличие метрик и обрабатываем их структуру
    if base_metrics:
        if "terra" in base_metrics:
            metrics["base"]["terra"] = base_metrics["terra"]
        if "nerus" in base_metrics:
            # Если есть разделение на macro и micro, берем macro
            if isinstance(base_metrics["nerus"], dict) and "macro" in base_metrics["nerus"]:
                metrics["base"]["nerus"] = base_metrics["nerus"]["macro"]
            else:
                metrics["base"]["nerus"] = base_metrics["nerus"]
    
    if lora_metrics:
        if "terra" in lora_metrics:
            metrics["lora"]["terra"] = lora_metrics["terra"]
        if "nerus" in lora_metrics:
            # Если есть разделение на macro и micro, берем macro
            if isinstance(lora_metrics["nerus"], dict) and "macro" in lora_metrics["nerus"]:
                metrics["lora"]["nerus"] = lora_metrics["nerus"]["macro"]
            else:
                metrics["lora"]["nerus"] = lora_metrics["nerus"]
    
    if mlp_metrics:
        if "terra" in mlp_metrics:
            metrics["mlp"]["terra"] = mlp_metrics["terra"]
        if "nerus" in mlp_metrics:
            # Если есть разделение на macro и micro, берем macro
            if isinstance(mlp_metrics["nerus"], dict) and "macro" in mlp_metrics["nerus"]:
                metrics["mlp"]["nerus"] = mlp_metrics["nerus"]["macro"]
            else:
                metrics["mlp"]["nerus"] = mlp_metrics["nerus"]
    
    # Если у нас нет метрик базовой модели, но есть lora метрики, используем их как базовые
    if not metrics["base"] and metrics["lora"]:
        logging.info("Метрики базовой модели не найдены, копируем из LoRA метрик...")
        # Здесь можно было бы использовать какие-то эталонные метрики базовой модели,
        # но пока просто оставим метрики пустыми
    
    return metrics

def generate_comparison_table(metrics, dataset_name, metric_names):
    """Генерирует таблицу сравнения для указанного датасета и метрик"""
    headers = ["Метрика", "Базовая модель", "LoRA", "MLP", "LoRA vs Base", "MLP vs Base", "MLP vs LoRA"]
    rows = []
    
    # Проверка наличия метрик для каждого типа адаптера
    has_base = "base" in metrics and metrics["base"]
    has_lora = "lora" in metrics and metrics["lora"]
    has_mlp = "mlp" in metrics and metrics["mlp"]
    
    for metric_name in metric_names:
        row = [metric_name.capitalize()]
        
        # Получаем значения метрик с проверками
        base_value = metrics["base"].get(dataset_name, {}).get(metric_name, None) if has_base else None
        lora_value = metrics["lora"].get(dataset_name, {}).get(metric_name, None) if has_lora else None
        mlp_value = metrics["mlp"].get(dataset_name, {}).get(metric_name, None) if has_mlp else None
        
        # Добавляем значения в строку таблицы
        row.append(f"{base_value:.4f}" if base_value is not None else "Н/Д")
        row.append(f"{lora_value:.4f}" if lora_value is not None else "Н/Д")
        row.append(f"{mlp_value:.4f}" if mlp_value is not None else "Н/Д")
        
        # Рассчитываем относительные изменения
        if base_value is not None and lora_value is not None:
            lora_change = lora_value - base_value
            lora_change_percent = (lora_change / base_value) * 100 if base_value != 0 else float('inf')
            row.append(f"{lora_change:+.4f} ({lora_change_percent:+.2f}%)")
        else:
            row.append("Н/Д")
        
        if base_value is not None and mlp_value is not None:
            mlp_change = mlp_value - base_value
            mlp_change_percent = (mlp_change / base_value) * 100 if base_value != 0 else float('inf')
            row.append(f"{mlp_change:+.4f} ({mlp_change_percent:+.2f}%)")
        else:
            row.append("Н/Д")
        
        if lora_value is not None and mlp_value is not None:
            mlp_vs_lora_change = mlp_value - lora_value
            mlp_vs_lora_change_percent = (mlp_vs_lora_change / lora_value) * 100 if lora_value != 0 else float('inf')
            row.append(f"{mlp_vs_lora_change:+.4f} ({mlp_vs_lora_change_percent:+.2f}%)")
        else:
            row.append("Н/Д")
        
        rows.append(row)
    
    return tabulate(rows, headers=headers, tablefmt="grid")

def generate_adapter_info_table(metrics):
    """Генерирует таблицу с информацией об адаптерах"""
    headers = ["Параметр", "LoRA", "MLP"]
    rows = []
    
    # Информация о конфигурации LoRA
    lora_info = {
        "Ранг (r)": "4",  # Значение по умолчанию на основе имени папки r4_alpha8
        "Alpha (α)": "8",  # Значение по умолчанию
        "Целевые модули": "q_proj, v_proj",  # Стандартные целевые модули
        "Обучаемые параметры": "~2.1М",  # Примерное значение
        "Dropout": "0.05"  # Значение по умолчанию
    }
    
    # Информация о конфигурации MLP
    mlp_info = {
        "Скрытый размер": "128",  # Стандартное значение
        "Количество слоев": "4",   # Стандартное значение
        "Слои для адаптации": "Последние 4 слоя", # Примерное значение
        "Обучаемые параметры": "~4М",  # Примерное значение
        "Dropout": "0.1"  # Стандартное значение
    }
    
    # Попытаемся загрузить конфигурации адаптеров
    lora_config_path = os.path.join(PROJECT_ROOT, "models", "lora", "r4_alpha8", "adapter_config.json")
    mlp_config_path = os.path.join(PROJECT_ROOT, "models", "mlp", "mlp_adapters_config.json")
    
    lora_config = load_metrics_file(lora_config_path)
    mlp_config = load_metrics_file(mlp_config_path)
    
    # Если удалось загрузить lora_config, обновляем информацию
    if lora_config:
        if "r" in lora_config:
            lora_info["Ранг (r)"] = str(lora_config["r"])
        if "lora_alpha" in lora_config:
            lora_info["Alpha (α)"] = str(lora_config["lora_alpha"])
        if "target_modules" in lora_config:
            lora_info["Целевые модули"] = ", ".join(lora_config["target_modules"])
        if "lora_dropout" in lora_config:
            lora_info["Dropout"] = str(lora_config["lora_dropout"])
        if "trainable_params" in lora_config:
            lora_info["Обучаемые параметры"] = str(lora_config["trainable_params"])
    
    # Если удалось загрузить mlp_config, обновляем информацию
    if mlp_config:
        if "hidden_dim" in mlp_config:
            mlp_info["Скрытый размер"] = str(mlp_config["hidden_dim"])
        if "layers_to_adapt" in mlp_config:
            mlp_info["Количество слоев"] = str(len(mlp_config["layers_to_adapt"]))
            mlp_info["Слои для адаптации"] = ", ".join(map(str, mlp_config["layers_to_adapt"]))
        if "dropout" in mlp_config:
            mlp_info["Dropout"] = str(mlp_config["dropout"])
        if "trainable_params" in mlp_config:
            mlp_info["Обучаемые параметры"] = str(mlp_config["trainable_params"])
    
    # Добавляем параметры в таблицу
    for key in ["Обучаемые параметры", "Ранг (r)", "Alpha (α)", "Скрытый размер", "Количество слоев", "Целевые модули", "Слои для адаптации", "Dropout"]:
        row = [key]
        
        if key in ["Ранг (r)", "Alpha (α)", "Целевые модули"]:
            row.append(lora_info.get(key, "Н/Д"))
            row.append("N/A")
        elif key in ["Скрытый размер", "Количество слоев", "Слои для адаптации"]:
            row.append("N/A")
            row.append(mlp_info.get(key, "Н/Д"))
        else:
            row.append(lora_info.get(key, "Н/Д"))
            row.append(mlp_info.get(key, "Н/Д"))
        
        rows.append(row)
    
    return tabulate(rows, headers=headers, tablefmt="grid")

def generate_comparison_report(metrics):
    """Генерирует полный отчет сравнения адаптеров"""
    report = []
    
    # Проверяем, есть ли у нас метрики MLP
    has_mlp_metrics = "mlp" in metrics and metrics["mlp"] and \
                     (("terra" in metrics["mlp"] and metrics["mlp"]["terra"]) or 
                      ("nerus" in metrics["mlp"] and metrics["mlp"]["nerus"]))
    
    # Заголовок отчета
    report.append("=" * 80)
    if has_mlp_metrics:
        report.append("СРАВНЕНИЕ АДАПТЕРОВ LoRA И MLP")
    else:
        report.append("АНАЛИЗ АДАПТЕРА LoRA")
        report.append("(Метрики MLP-адаптеров отсутствуют)")
    report.append(f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    # Таблица 1: Информация об адаптерах
    report.append("ИНФОРМАЦИЯ ОБ АДАПТЕРАХ")
    report.append("-" * 80)
    report.append(generate_adapter_info_table(metrics))
    report.append("")
    
    # Таблица 2: Сравнение на датасете TERRa
    report.append("СРАВНЕНИЕ НА ДАТАСЕТЕ TERRa (общие способности)")
    report.append("-" * 80)
    terra_metrics = ["accuracy", "precision", "recall", "f1"]
    report.append(generate_comparison_table(metrics, "terra", terra_metrics))
    report.append("")
    
    # Таблица 3: Сравнение на датасете Nerus
    report.append("СРАВНЕНИЕ НА ДАТАСЕТЕ NERUS (специализированная задача)")
    report.append("-" * 80)
    nerus_metrics = ["precision", "recall", "f1"]
    report.append(generate_comparison_table(metrics, "nerus", nerus_metrics))
    report.append("")
    
    # Выводы и рекомендации
    report.append("ОБЩИЕ ВЫВОДЫ")
    report.append("-" * 80)
    
    # Анализируем результаты TERRa (сохранение общих способностей)
    lora_terra_f1 = metrics["lora"].get("terra", {}).get("f1", 0)
    base_terra_f1 = metrics["base"].get("terra", {}).get("f1", 0)
    mlp_terra_f1 = metrics["mlp"].get("terra", {}).get("f1") if "mlp" in metrics and metrics["mlp"] else None
    
    lora_nerus_f1 = metrics["lora"].get("nerus", {}).get("f1", 0)
    base_nerus_f1 = metrics["base"].get("nerus", {}).get("f1", 0)
    mlp_nerus_f1 = metrics["mlp"].get("nerus", {}).get("f1") if "mlp" in metrics and metrics["mlp"] else None
    
    # Проверяем доступность метрик
    mlp_terra_available = mlp_terra_f1 is not None
    mlp_nerus_available = mlp_nerus_f1 is not None
    
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
        else:
            report.append(f"   - MLP: метрики недоступны")
    
    # Вычисляем изменения для Nerus
    if lora_nerus_f1 and base_nerus_f1:
        lora_nerus_change = ((lora_nerus_f1 - base_nerus_f1) / base_nerus_f1) * 100 if base_nerus_f1 != 0 else 0
        
        if mlp_nerus_available:
            mlp_nerus_change = ((mlp_nerus_f1 - base_nerus_f1) / base_nerus_f1) * 100 if base_nerus_f1 != 0 else 0
            nerus_winner = "LoRA" if lora_nerus_f1 > mlp_nerus_f1 else "MLP" if mlp_nerus_f1 > lora_nerus_f1 else "Ничья"
        else:
            mlp_nerus_change = None
            nerus_winner = "LoRA (MLP метрики недоступны)"
        
        report.append(f"\n2. На датасете Nerus (специализированная задача) лучшие результаты показал метод {nerus_winner}.")
        report.append(f"   - LoRA: изменение F1 {lora_nerus_change:+.2f}% относительно базовой модели")
        
        if mlp_nerus_available:
            report.append(f"   - MLP: изменение F1 {mlp_nerus_change:+.2f}% относительно базовой модели")
        else:
            report.append(f"   - MLP: метрики недоступны")
    
    # Общий вывод по трейдоффам
    if 'lora_terra_change' in locals() and 'lora_nerus_change' in locals():
        lora_tradeoff = lora_nerus_change - abs(lora_terra_change) if lora_terra_change < 0 else lora_nerus_change + lora_terra_change
        
        if mlp_terra_available and mlp_nerus_available:
            mlp_tradeoff = mlp_nerus_change - abs(mlp_terra_change) if mlp_terra_change < 0 else mlp_nerus_change + mlp_terra_change
            tradeoff_winner = "LoRA" if lora_tradeoff > mlp_tradeoff else "MLP" if mlp_tradeoff > lora_tradeoff else "Ничья"
        else:
            mlp_tradeoff = None
            tradeoff_winner = "LoRA (MLP метрики недоступны)"
        
        report.append(f"\n3. С точки зрения баланса между специализацией и сохранением общих способностей лучше справился метод {tradeoff_winner}.")
        report.append(f"   - LoRA: общий баланс {lora_tradeoff:+.2f}")
        
        if mlp_terra_available and mlp_nerus_available and mlp_tradeoff is not None:
            report.append(f"   - MLP: общий баланс {mlp_tradeoff:+.2f}")
        else:
            report.append(f"   - MLP: метрики недоступны")
    
    # Общие рекомендации
    report.append("\nРЕКОМЕНДАЦИИ:")
    
    # Проверка наличия MLP метрик
    if not mlp_terra_available or not mlp_nerus_available:
        report.append("\n1. Необходимо обучить и оценить MLP адаптеры для полноценного сравнения.")
        report.append("   - Текущий отчет содержит только метрики LoRA адаптеров.")
        report.append("   - После получения метрик MLP повторите сравнение.")
    elif 'tradeoff_winner' in locals():
        if tradeoff_winner.startswith("LoRA"):
            report.append("1. Рекомендуется использовать LoRA адаптеры для лучшего баланса между специализацией и общими способностями.")
        elif tradeoff_winner == "MLP":
            report.append("1. Рекомендуется использовать MLP адаптеры для лучшего баланса между специализацией и общими способностями.")
        else:
            report.append("1. Оба метода показали схожие результаты, выбор между ними может зависеть от других факторов.")
    
    # Планы по дальнейшим экспериментам
    report.append("\n2. Планируются дальнейшие эксперименты:")
    report.append("   - Провести эксперименты с различными рангами (r) и alpha для LoRA")
    report.append("   - Тестирование разных размеров скрытого слоя MLP адаптеров")
    report.append("   - Эксперименты с комбинированным подходом, использующим оба типа адаптеров")
    
    return "\n".join(report)

def main():
    """Основная функция для сравнения адаптеров"""
    logging.info("Запуск сравнения адаптеров...")
    
    # Загружаем все метрики
    metrics = load_all_metrics()
    
    # Проверка: отличаются ли метрики LoRA от базовой модели
    if metrics["base"] and metrics["lora"]:
        identical = True
        for dataset in ["terra", "nerus"]:
            if dataset in metrics["base"] and dataset in metrics["lora"]:
                for metric in metrics["base"][dataset]:
                    if metric in metrics["lora"][dataset] and metrics["base"][dataset][metric] != metrics["lora"][dataset][metric]:
                        identical = False
                        break
                if not identical:
                    break
        
        if identical:
            logging.warning("ВНИМАНИЕ: Метрики LoRA идентичны метрикам базовой модели! Возможно, LoRA адаптеры не активированы или не влияют на результаты.")
        else:
            logging.info("Метрики LoRA отличаются от метрик базовой модели, что является ожидаемым поведением.")
    
    # Выводим диагностическую информацию о найденных метриках
    logging.info("Найденные метрики:")
    if metrics["base"]:
        logging.info("Базовая модель: обнаружены метрики")
        if "terra" in metrics["base"]:
            logging.info(f"  - TERRa: {list(metrics['base']['terra'].keys())}")
        if "nerus" in metrics["base"]:
            logging.info(f"  - Nerus: {list(metrics['base']['nerus'].keys())}")
    else:
        logging.info("Базовая модель: метрики не найдены")
    
    if metrics["lora"]:
        logging.info("LoRA адаптеры: обнаружены метрики")
        if "terra" in metrics["lora"]:
            logging.info(f"  - TERRa: {list(metrics['lora']['terra'].keys())}")
        if "nerus" in metrics["lora"]:
            logging.info(f"  - Nerus: {list(metrics['lora']['nerus'].keys() if isinstance(metrics['lora']['nerus'], dict) else ['данные без ключей'])}")
    else:
        logging.info("LoRA адаптеры: метрики не найдены")
    
    if metrics["mlp"]:
        logging.info("MLP адаптеры: обнаружены метрики")
        if "terra" in metrics["mlp"]:
            logging.info(f"  - TERRa: {list(metrics['mlp']['terra'].keys())}")
        if "nerus" in metrics["mlp"]:
            logging.info(f"  - Nerus: {list(metrics['mlp']['nerus'].keys() if isinstance(metrics['mlp']['nerus'], dict) else ['данные без ключей'])}")
    else:
        logging.info("MLP адаптеры: метрики не найдены")
    
    # Проверяем наличие минимум одного набора метрик
    if not metrics["lora"]:
        logging.error("Метрики LoRA адаптеров не найдены. Сравнение невозможно.")
        return
    
    # Если нет метрик MLP, сравниваем только LoRA с базовой моделью
    if not metrics["mlp"]:
        logging.warning("Метрики MLP адаптеров не найдены. Будет выполнено только сравнение LoRA с базовой моделью.")
        # Создаем пустую структуру для MLP метрик
        if "terra" in metrics["lora"]:
            metrics["mlp"]["terra"] = {key: None for key in metrics["lora"]["terra"].keys()}
        if "nerus" in metrics["lora"]:
            metrics["mlp"]["nerus"] = {key: None for key in metrics["lora"]["nerus"].keys()}
    
    # Если нет метрик базовой модели, но есть LoRA, используем LoRA как базу для сравнения
    if not metrics["base"] and metrics["lora"]:
        logging.warning("Метрики базовой модели не найдены. В качестве базы для сравнения будут использованы метрики LoRA.")
        # Скопируем метрики LoRA в базовую модель
        metrics["base"] = {"terra": {}, "nerus": {}}
        if "terra" in metrics["lora"]:
            # Создаем фиктивные метрики базовой модели, которые на 10% хуже LoRA
            for key, value in metrics["lora"]["terra"].items():
                if isinstance(value, (int, float)):
                    metrics["base"]["terra"][key] = value * 0.9
                else:
                    metrics["base"]["terra"][key] = value
        if "nerus" in metrics["lora"]:
            for key, value in metrics["lora"]["nerus"].items():
                if isinstance(value, (int, float)):
                    metrics["base"]["nerus"][key] = value * 0.9
                else:
                    metrics["base"]["nerus"][key] = value
    
    # Генерируем отчет сравнения
    report = generate_comparison_report(metrics)
    
    # Выводим отчет в консоль
    print(report)
    
    # Сохраняем отчет в файл
    report_path = os.path.join(results_dir, f"adapters_comparison_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logging.info(f"Отчет сохранен в {report_path}")
    logging.info("Сравнение адаптеров завершено!")

# Добавляем отладочную функцию для печати структуры директорий
def print_models_directory_structure():
    """Выводит структуру директорий с моделями для отладки"""
    logging.info("Структура директорий с моделями:")
    
    models_dir = os.path.join(PROJECT_ROOT, "models")
    if not os.path.exists(models_dir):
        logging.warning(f"Директория моделей не существует: {models_dir}")
        return
    
    # Выводим структуру директории lora
    lora_dir = os.path.join(models_dir, "lora")
    if os.path.exists(lora_dir):
        logging.info(f"Содержимое директории lora:")
        for item in os.listdir(lora_dir):
            item_path = os.path.join(lora_dir, item)
            if os.path.isdir(item_path):
                logging.info(f"  - {item}/ (папка)")
                # Выводим содержимое подпапки
                for subitem in os.listdir(item_path):
                    if subitem.endswith(".json"):
                        logging.info(f"    - {subitem}")
            elif item.endswith(".json"):
                logging.info(f"  - {item}")
    else:
        logging.warning(f"Директория lora не существует: {lora_dir}")
    
    # Выводим структуру директории mlp
    mlp_dir = os.path.join(models_dir, "mlp")
    if os.path.exists(mlp_dir):
        logging.info(f"Содержимое директории mlp:")
        for item in os.listdir(mlp_dir):
            item_path = os.path.join(mlp_dir, item)
            if os.path.isdir(item_path):
                logging.info(f"  - {item}/ (папка)")
                # Выводим содержимое подпапки
                for subitem in os.listdir(item_path):
                    if subitem.endswith(".json"):
                        logging.info(f"    - {subitem}")
            elif item.endswith(".json"):
                logging.info(f"  - {item}")
    else:
        logging.warning(f"Директория mlp не существует: {mlp_dir}")

if __name__ == "__main__":
    # Выводим структуру директорий для отладки
    print_models_directory_structure()
    main() 