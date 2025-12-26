import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_lags_and_rolling_features(df):
    df_processed = df.copy()
    for lag in [1, 7, 14, 30]:
        df_processed[f'sales_lag_{lag}'] = df_processed['sales'].shift(lag)
    df_processed['sales_rolling_mean_7'] = df_processed['sales'].rolling(window=7).mean().shift(1)  # shift(1) чтобы избежать утечки данных
    return df_processed.dropna()


def run_mlp_expert_pipeline(df_input, target_col='sales', epochs=50, forecast_length=21, seed=42): # Выполняет полный пайплайн для обучения и оценки MLP модели как экспертного пайплайна
    # Устанавливаем seed для воспроизводимости
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 1. Ручная Генерация Признаков (Feature Engineering)
    df_feat = create_lags_and_rolling_features (df_input)

    if 'date' in df_feat.columns:
        df_feat = df_feat.drop(columns=['date'])
    df_feat = df_feat.dropna()

    # Разделение на признаки (X) и целевую переменную (y)
    X = df_feat.drop(columns=[target_col])
    y = df_feat[target_col]

    # 2. Разделение на обучающую и тестовую выборки
    # Разделение по времени: последние forecast_length строк идут в тест

    X_train = X.iloc[:-forecast_length]
    X_test = X.iloc[-forecast_length:]
    y_train = y.iloc[:-forecast_length]
    y_test = y.iloc[-forecast_length:]

    # 3. Масштабирование данных
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    # 4. Построение и обучение MLP (Стандартный подбор параметров)
    input_dim = X_train_scaled.shape[1]

    model = Sequential()
    # Стандартная архитектура: 2-3 плотных слоя
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Выходной слой для регрессии

    model.compile(optimizer='adam', loss='mse')

    # Обучение (стандартные параметры: 1000 эпох, размер батча 32)
    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=epochs,
        batch_size=32,
        verbose=0
    )

    # 5. Прогнозирование и обратное масштабирование
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    # 6. Оценка метрик
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results = {
        'seed': seed,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred,
        'y_test': y_test.values  # Сохраняем фактические значения для построения графиков
    }

    return results



if __name__ == "__main__":
    all_mlp_results = []
    seeds_to_run = [20, 30, 42]
    forecast_length = 21
    data = pd.read_csv('data/preprocessing/train_filtered.csv', parse_dates=['date'])

    # Запускаем эксперименты для каждого seed
    for s in seeds_to_run:
        print(f"Запуск MLP-эксперимента с seed={s}...")
        experiment_result = run_mlp_expert_pipeline(
            df_input=data,
            target_col='sales',
            epochs = 50,
            forecast_length=forecast_length,
            seed=s
        )
        all_mlp_results.append(experiment_result)

    # Агрегация результатов
    results_for_table_mlp = []
    for r in all_mlp_results:
        results_for_table_mlp.append({
            'Seed': r['seed'],
            'RMSE': r['rmse'],
            'MAE': r['mae'],
            'R2': r['r2']
        })

    df_results_mlp = pd.DataFrame(results_for_table_mlp)

    # Вывод сводных результатов
    print("\nСводные результаты MLP (Expert)")
    print(f"Средний RMSE: {df_results_mlp['RMSE'].mean():.3f} +/- {df_results_mlp['RMSE'].std():.3f}")
    print(f"Средний MAE: {df_results_mlp['MAE'].mean():.3f} +/- {df_results_mlp['MAE'].std():.3f}")
    print(f"Средний R^2: {df_results_mlp['R2'].mean():.3f} +/- {df_results_mlp['R2'].std():.3f}")

    df_results_mlp.to_csv('models/MLP/mlp_expert_results.csv', index=False)
    print("\nРезультаты MLP сохранены в файл 'models/MLP/mlp_expert_results.csv'")

