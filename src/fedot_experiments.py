import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from plot_forecast import plot_forecast
import warnings

warnings.filterwarnings('ignore')

def prepare_data_for_fedot():
    print("\n--- Шаг 1: Подготовка данных для FEDOT ---")
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=7))

    data_input = InputData.from_csv_time_series(task=task,
                                                file_path='data/preprocessing/train_filtered.csv',
                                                target_column='sales')

    train_data, test_data = train_test_data_setup(data_input, validation_blocks=3)

    return task, train_data, test_data

def fedot_experiment(task, train_data, test_data, timeout: int, seed: int, exp_id):
    model_fedot = Fedot(problem='ts_forecasting',
                  task_params=task.task_params,
                  timeout=timeout,
                  preset='fast_train',
                  seed=seed)

    print(f"Запуск AutoML (timeout={timeout}min)...")
    pipeline_fedot = model_fedot.fit(train_data)

    # Сохраняем пайплайн
    pipeline_path = f"models/fedot_experiments/timeout_{timeout}/pipeline_timeout_{timeout}_seed_{seed}.json"
    pipeline_fedot.save(path=pipeline_path)
    print(f"Пайплайн сохранен: {pipeline_path}")

    # Прогноз
    print("Генерация прогноза...")
    forecast = model_fedot.predict(test_data, validation_blocks=3)

    # Метрики
    metrics = model_fedot.get_metrics(metric_names=['rmse', 'mae', 'r2'], validation_blocks=3)
    print(f"\nМЕТРИКИ (timeout={timeout}min):")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAE:  {metrics['mae']:.4f}")
    print(f"   R²:   {metrics['r2']:.4f}")

    # Сохраняем метрики в CSV
    metrics_df = pd.DataFrame([{
        'timeout': timeout,
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'r2': metrics['r2']
    }])
    metrics_path = f"models/fedot_experiments/timeout_{timeout}/metrics_timeout_{timeout}_seed_{seed}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Метрики сохранены: {metrics_path}")

    # Сохраняем прогноз
    forecast_df = pd.DataFrame({
        'forecast': forecast.flatten(),
        'actual': test_data.target
    })
    forecast_path = f"models/fedot_experiments/timeout_{timeout}/forecast_timeout_{timeout}_seed_{seed}.csv"
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Прогноз сохранен: {forecast_path}")

    # Графики
    print("Сохранение графиков...")

    # 1. Структура пайплайна
    pipeline_graph = f"models/fedot_experiments/timeout_{timeout}/pipeline_graph_timeout_{timeout}_seed_{seed}.png"
    pipeline_fedot.show(save_path=pipeline_graph)
    print(f"  Pipeline graph: {pipeline_graph}")

    # 2. Прогноз vs actual
    prediction_plot = f"models/fedot_experiments/timeout_{timeout}/prediction_plot_timeout_{timeout}_seed_{seed}.png"
    name_prediction_plot = f"FEDOT Forecast vs Actual (timeout={timeout}min)"
    plot_forecast(train_data.target, test_data.target, forecast.flatten(), prediction_plot, name_prediction_plot)
    print(f"  Prediction plot: {prediction_plot}")

    print(f"\nЭксперимент #{exp_id} ЗАВЕРШЕН! Все файлы в fedot_experiments/timeout_{timeout}/")
    print("   Структура пайплайна:")
    print(pipeline_fedot)

    return {
        'seed': seed,
        'timeout': timeout,
        'metrics': metrics,
        'pipeline_path': pipeline_path,
        'forecast_path': forecast_path,
        'metrics_path': metrics_path
    }


if __name__ == "__main__":
    print("ЗАПУСК FEDOT TIMEOUT EXPERIMENT")

    task, train_data, test_data = prepare_data_for_fedot()
    seeds = [20,30,42]
    timeouts = [1]

    results = []
    for seed in seeds:
        for i, timeout in enumerate(timeouts, 1):
            result = fedot_experiment(task, train_data, test_data, timeout, seed, i)
            results.append(result)

    # Итоговая таблица
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    summary_df = pd.DataFrame([r['metrics'] for r in results])
    summary_df['seed'] = [r['seed'] for r in results]
    summary_df['timeout'] = [r['timeout'] for r in results]
    summary_df = summary_df[['seed', 'timeout', 'rmse', 'mae', 'r2']].round(4)
    print(summary_df.to_string(index=False))

    # Сохраняем общую таблицу
    summary_path = "models/fedot_experiments/summary_all_timeouts.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nОбщая таблица сохранена: {summary_path}")

    print("\nВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("Результаты в папке: fedot_experiments/")
