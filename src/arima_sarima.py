import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

def arima_sarima_model():
    # 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
    df = pd.read_csv('data/preprocessing/train_filtered.csv', parse_dates=['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # SARIMA требует непрерывного временного ряда.
    df = df.asfreq('D')

    # Заполнение пропусков в целевой переменной и признаках
    df['sales'] = df['sales'].fillna(0)
    df['onpromotion'] = df['onpromotion'].fillna(0)
    df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill() # Цены на нефть
    df['transactions'] = df['transactions'].fillna(0)
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_holiday_overall'] = df['is_holiday_overall'].fillna(0)

    # Выбор целевой переменной и экзогенных факторов
    sales_ts = df['sales']
    exog_vars = df[['onpromotion', 'dcoilwtico', 'transactions', 'dayofweek', 'is_weekend', 'is_holiday_overall']]

    # 2. ПРОВЕРКА СТАЦИОНАРНОСТИ (ADF ТЕСТ)
    def check_stationarity(ts, title=""):
        result = adfuller(ts.dropna())
        print(f'--- ADF Тест: {title} ---')
        print(f'Статистика: {result[0]:.4f}')
        print(f'p-значение: {result[1]:.4f}')
        return 0 if result[1] < 0.05 else 1

    d = check_stationarity(sales_ts, "Исходный ряд")
    D = check_stationarity(sales_ts.diff(7).dropna(), "Сезонное дифференцирование (s=7)")

    print(f"\nРекомендуемые порядки: d={d}, D={D}")

    # 3. ВИЗУАЛИЗАЦИЯ ACF / PACF (ДЛЯ p, q, P, Q)
    path_ACF_PACF_plot = f"models/SARIMA/ACF_PACF_plot.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plot_acf(sales_ts.diff(7).dropna() if D==1 else sales_ts, ax=axes[0], lags=35)
    axes[0].set_title('Autocorrelation (ACF)')
    plot_pacf(sales_ts.diff(7).dropna() if D==1 else sales_ts, ax=axes[1], lags=35)
    axes[1].set_title('Partial Autocorrelation (PACF)')
    plt.tight_layout()
    plt.savefig(path_ACF_PACF_plot, bbox_inches='tight')

    # Исходя из графиков и структуры Store Sales выберем базовые параметры:
    p, q = 1, 1
    P, Q = 1, 1
    s = 7

    # 4. ОБУЧЕНИЕ МОДЕЛИ SARIMAX
    # Разделение на Hold-out выборку (последние 21 дней для теста)
    test_days = 21
    train_data = df.iloc[:-test_days]
    test_data = df.iloc[-test_days:]

    train_exog = exog_vars.iloc[:-test_days]
    test_exog = exog_vars.iloc[-test_days:]

    print(f"\nОбучение SARIMAX({p},{d},{q})({P},{D},{Q},7)...")

    start_time = time.time()
    model = SARIMAX(train_data['sales'],
                    exog=train_exog,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    results = model.fit(disp=False)

    end_time = time.time()
    print(f"\nОбучение завершено за {end_time - start_time:.2f} секунд.")

    print(results.summary())

    # 5. ПРОГНОЗ И РАСЧЕТ МЕТРИК
    # Получаем прогноз
    pred_res = results.get_prediction(start=test_data.index[0], end=test_data.index[-1], exog=test_exog)
    predictions = pred_res.predicted_mean
    predictions = np.maximum(0, predictions) # Убираем отрицательные значения

    forecast_df = pd.DataFrame({
        'forecast': predictions,
        'actual': test_data['sales'].values,
    })
    forecast_path = f"models/SARIMA/forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Прогноз сохранен: {forecast_path}")


    # Функция для расчета метрик
    def calculate_metrics(actual, forecast):
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        mae = mean_absolute_error(actual, forecast)
        r2 = r2_score(actual, forecast)
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

    metrics = calculate_metrics(test_data['sales'], predictions)

    # Сохранение метрик в CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_path = f"models/SARIMA/metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Метрики сохранены: {metrics_path}")

    print("\n" + "="*40)
    print("МЕТРИКИ КАЧЕСТВА (TEST):")
    print(f"RMSE:  {metrics['rmse']:.4f}")
    print(f"MAE:   {metrics['mae']:.4f}")
    print(f"R2:    {metrics['r2']:.4f}")
    print("="*40)

    # 6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
    path_prediction_plot = f"models/SARIMA/prediction_plot.png"
    plt.figure(figsize=(15, 7))
    plt.plot(train_data.index[-60:], train_data['sales'][-60:], label='Обучение (последние 60 дней)', color='blue', alpha=0.5)
    plt.plot(test_data.index, test_data['sales'], label='Фактические продажи', color='black', lw=2)
    plt.plot(test_data.index, predictions, label='Прогноз SARIMA', color='red', linestyle='--', lw=2)

    # Доверительный интервал
    pred_ci = pred_res.conf_int()
    plt.fill_between(test_data.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3, label='95% Дов. интервал')

    plt.title(f'Сравнение SARIMA с фактом\nR2: {metrics["r2"]:.4f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path_prediction_plot, bbox_inches='tight')

if __name__ == "__main__":
    arima_sarima_model()