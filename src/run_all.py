import subprocess
import os
import sys
import time

def run_script(script_name):
    """Запускает указанный скрипт Python и выводит его статус."""
    print("-" * 50)
    print(f"Запуск: {script_name}")
    print("-" * 50)

    command = [sys.executable, script_name]

    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"[{script_name}] Успешно завершен.")
        if process.stdout:
            print("\n--- Вывод скрипта ---")
            # Выводим только последние 10 строк вывода, чтобы не загромождать консоль
            print('\n'.join(process.stdout.strip().split('\n')[-10:]))
            print("----------------------\n")

    except subprocess.CalledProcessError as e:
        print(f"\n[ОШИБКА] Скрипт {script_name} завершился с ошибкой (Код {e.returncode}).")
        print("\n--- STDERR ---")
        print(e.stderr)
        print("----------------\n")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n[ОШИБКА] Файл {script_name} не найден. Проверьте путь.")
        sys.exit(1)


if __name__ == "__main__":

    # Определяем корневую директорию проекта
    project_root = os.path.dirname(os.path.abspath(__file__))

    scripts_to_run = [
        'src/process_data.py',           # Обработка сырых данных, сохранение
        'src/filter_data.py',            # Фильтрация предобработанных данных, сохранение
        # 'src/fedot_experiments.py',    # Эксперименты с FEDOT
        # 'src/lightgbm_optuna.py',      # Обучение LightGBM + Optuna + TimeSeriesSplit (ручной пайплайн)
        'src/arima_sarima.py',           # Обучение SARIMA (ручной пайплайн)
        'src/manual_fe_mlp.py'           # Обучение MLP (ручной пайплайн)
    ]

    start_time = time.time()

    print("=" * 60)
    print("НАЧАЛО КОМПЛЕКСНОГО ЗАПУСКА ЭКСПЕРИМЕНТОВ")
    print("=" * 60)

    for script in scripts_to_run:
        run_script(script)

    end_time = time.time()
    total_time = (end_time - start_time) / 60

    print("=" * 60)
    print("ВСЕ ЭКСПЕРИМЕНТЫ УСПЕШНО ЗАВЕРШЕНЫ")
    print(f"Общее время выполнения: {total_time:.2f} минут")
    print("Метрики и прогнозы сохранены в папке 'models/'.")
    print("=" * 60)