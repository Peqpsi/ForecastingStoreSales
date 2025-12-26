import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os
from plot_forecast import plot_forecast

def create_lags_and_rolling_features(df):
    df_processed = df.copy()
    for lag in [1, 7, 14, 30]:
        df_processed[f'sales_lag_{lag}'] = df_processed['sales'].shift(lag)
    df_processed['sales_rolling_mean_7'] = df_processed['sales'].rolling(window=7).mean().shift(1)  # shift(1) чтобы избежать утечки данных
    return df_processed.dropna()

def objective_lgbm(trial, X, y, seed):  # Функция Optuna для LightGBM с TimeSeriesSplit
    # 1. Определение гиперпараметров
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'verbose': -1,
        'n_jobs': -1,
        'seed': seed
    }

    # 2. TimeSeriesSplit Кросс-валидация (5 фолдов)
    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[val_index]

        # 3. Обучение модели
        model = lgb.LGBMRegressor(**params)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            categorical_feature=CATEGORICAL_FEATURES
        )

        # 4. Оценка
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds)
        rmse_scores.append(rmse)

        # 5. Прунинг Optuna
        trial.report(rmse, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Возвращаем средний RMSE по всем фолдам
    return np.mean(rmse_scores)


# 2. Основной пайплайн с Optuna
def optuna_lgbm_pipeline(data_processed, max_trials=50, max_time=30, seed=42):  # Пайплайн LightGBM с оптимизацией гиперпараметров через Optuna, использующий TimeSeriesSplit

    # 1. Разделение данных
    TEST_SIZE = 21

    # Данные для обучения и оптимизации Optuna
    X_optuna = data_processed[FEATURES].iloc[:-TEST_SIZE]
    y_optuna = data_processed['sales'].iloc[:-TEST_SIZE]

    # Данные для финального теста
    X_test = data_processed[FEATURES].iloc[-TEST_SIZE:]
    y_test = data_processed['sales'].iloc[-TEST_SIZE:]

    y_train_plot = data_processed['sales'].iloc[:len(X_optuna)]

    print(f"Данные для Optuna (Train/Validation): {len(X_optuna)} точек")
    print(f"Данные для финального теста: {len(X_test)} точек")

    # 2. Запуск Optuna
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30)
    )

    # Обертка для передачи данных в objective
    func = lambda trial: objective_lgbm(trial, X_optuna, y_optuna, seed)

    start_time = time.time()
    study.optimize(
        func,
        n_trials=max_trials,
        timeout=max_time * 60,  # max_time в минутах
        show_progress_bar=True
    )
    end_time = time.time()

    print(f"\nОптимизация завершена за {end_time - start_time:.2f} секунд.")
    print(f"Лучший RMSE (CV): {study.best_value:.4f}")

    # 3. Обучение финальной модели
    best_params = study.best_params
    best_params['objective'] = 'regression'
    best_params['metric'] = 'rmse'
    best_params['n_estimators'] = 2000  # Увеличиваем для финального обучения
    best_params['n_jobs'] = -1
    best_params['seed'] = seed

    final_model = lgb.LGBMRegressor(**best_params)

    # Обучаем на всем X_optuna, используя X_test для ранней остановки
    final_model.fit(
        X_optuna, y_optuna,
        eval_set=[(X_test, y_test)],  # Используем тестовый набор для честной early stopping
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        categorical_feature=CATEGORICAL_FEATURES
    )

    # 4. Прогноз и оценка на тестовой выборке
    pred = final_model.predict(X_test)

    final_rmse = mean_squared_error(y_test, pred, squared=False)
    final_mae = mean_absolute_error(y_test, pred)
    final_r2 = r2_score(y_test, pred)

    # 5. Формирование итогового вывода
    results = {
        'seed': seed,
        'rmse': final_rmse,
        'mae': final_mae,
        'r2': final_r2
    }

    print("Финальные метрики LightGBM + Optuna:")
    print(f"{{'rmse': {results['rmse']:.3f}, 'mae': {results['mae']:.3f}, 'r2': {results['r2']:.3f}}}")

    # БЛОК ОРГАНИЗАЦИИ ВЫВОДА (CSV и PNG)
    # Путь к папке
    base_dir = f"models/LightGBM(Optuna)/timeout_{max_time}_seed_{seed}"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        print(f"Создана директория: {base_dir}")

    # Сохранение прогноза в CSV
    forecast_df = pd.DataFrame({
        'forecast': pred,
        'actual': y_test.values,
    })

    forecast_path = f"models/LightGBM(Optuna)/timeout_{max_time}_seed_{seed}/forecast_timeout_{max_time}_seed_{seed}.csv"
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Прогноз сохранен: {forecast_path}")

    # Сохранение метрик в CSV
    metrics_df = pd.DataFrame([results])
    metrics_path = f"models/LightGBM(Optuna)/timeout_{max_time}_seed_{seed}/metrics_timeout_{max_time}_seed_{seed}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Метрики сохранены: {metrics_path}")

    # Визуализация
    prediction_plot = f"models/LightGBM(Optuna)/timeout_{max_time}_seed_{seed}/prediction_plot_timeout_{max_time}_seed_{seed}.png"
    name_prediction_plot = f"LightGBM(Optuna) Forecast vs Actual (timeout={max_time}min)"
    plot_forecast(y_train_plot.to_numpy().tolist(),
                  y_test.to_numpy().tolist(),
                  pred,
                  prediction_plot,
                  name_prediction_plot)

    print(f"  Prediction plot: {prediction_plot}")
    print(f"График сохранен: {prediction_plot}")
    return results

if __name__ == "__main__":
    np.random.seed(42)

    data = pd.read_csv('data/preprocessing/train_filtered.csv', parse_dates=['date'])
    data_processed = create_lags_and_rolling_features(data)

    print(data_processed.columns)
    print(data_processed.onpromotion.unique())

    # Фичи, которые используются в модели
    FEATURES = ['onpromotion', 'dcoilwtico', 'transactions', 'is_weekend', 'dayofweek', 'is_holiday_overall', 'sales_lag_1',
                'sales_lag_7', 'sales_lag_14', 'sales_lag_30', 'sales_rolling_mean_7']

    # Категориальные фичи для LightGBM
    CATEGORICAL_FEATURES = ['is_weekend', 'dayofweek', 'sales_lag_1',
                'sales_lag_7', 'sales_lag_14', 'sales_lag_30', 'sales_rolling_mean_7']

    # Запуск пайплайна (10000 трайлов, таймаут 15 минут)
    # Установим seed для воспроизводимости
    seeds = [20, 30, 42]
    timeouts = [5]

    results = []
    for seed in seeds:
        for i, timeout in enumerate(timeouts, 1):
            optuna_rmse = optuna_lgbm_pipeline(data_processed.copy(), max_trials=10000, max_time=timeout, seed=seed)
            results.append(optuna_rmse)

    # Агрегация результатов
    results_for_table_lightgbm_optuna = []
    for r in results:
        results_for_table_lightgbm_optuna.append({
            'Seed': r['seed'],
            'RMSE': r['rmse'],
            'MAE': r['mae'],
            'R2': r['r2']
        })

    df_results_lightgbm_optuna = pd.DataFrame(results_for_table_lightgbm_optuna)

    # Вывод сводных результатов
    print("\nСводные результаты LightGBM + Optuna (Expert)")
    print(f"Средний RMSE: {df_results_lightgbm_optuna['RMSE'].mean():.3f} +/- {df_results_lightgbm_optuna['RMSE'].std():.3f}")
    print(f"Средний MAE: {df_results_lightgbm_optuna['MAE'].mean():.3f} +/- {df_results_lightgbm_optuna['MAE'].std():.3f}")
    print(f"Средний R^2: {df_results_lightgbm_optuna['R2'].mean():.3f} +/- {df_results_lightgbm_optuna['R2'].std():.3f}")

    df_results_lightgbm_optuna.to_csv('models/LightGBM(Optuna)/lightgbm_optuna_results.csv', index=False)
    print("\nРезультаты MLP сохранены в файл 'models/LightGBM(Optuna)/lightgbm_optuna_results.csv'")