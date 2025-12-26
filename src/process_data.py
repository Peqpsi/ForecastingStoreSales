import pandas as pd
import os

def process_data():
    # 1. Загрузка данных
    print("Загрузка данных...")
    try:
        df_train = pd.read_csv('data/initial/train.csv', parse_dates=['date'])
        df_stores = pd.read_csv('data/initial/stores.csv')
        df_holidays = pd.read_csv('data/initial/holidays_events.csv', parse_dates=['date'])
        df_oil = pd.read_csv('data/initial/oil.csv', parse_dates=['date'])
        df_transactions = pd.read_csv('data/initial/transactions.csv', parse_dates=['date'])
        print("Данные для обучения успешно загружены.")
    except FileNotFoundError as e:
        print(f"Ошибка: Не найден файл в папке 'data/'. {e}")
        exit()

    # 2. Подготовка праздников (очистка от дубликатов)
    print("Очистка праздников...")

    df_holidays_proc = df_holidays[df_holidays['transferred'] == False].copy()

    nat_holidays = df_holidays_proc[df_holidays_proc['locale'] == 'National'][['date', 'type']]
    nat_holidays = nat_holidays.drop_duplicates('date').rename(columns={'type': 'national_holiday_type'})

    reg_holidays = df_holidays_proc[df_holidays_proc['locale'] == 'Regional'][['date', 'locale_name', 'type']]
    reg_holidays = reg_holidays.rename(columns={'type': 'regional_holiday_type', 'locale_name': 'state'}).drop_duplicates(['date', 'state'])

    loc_holidays = df_holidays_proc[df_holidays_proc['locale'] == 'Local'][['date', 'locale_name', 'type']]
    loc_holidays = loc_holidays.rename(columns={'type': 'local_holiday_type', 'locale_name': 'city'}).drop_duplicates(['date', 'city'])

    # 3. Объединение данных
    print("Объединение данных...")

    # Работаем только с тренировочным набором
    df = df_train.copy()

    # 3.1 Добавляем магазины
    df = df.merge(df_stores, on='store_nbr', how='left')

    # 3.2 Добавляем праздники
    df = df.merge(nat_holidays, on='date', how='left')
    df = df.merge(reg_holidays, on=['date', 'state'], how='left')
    df = df.merge(loc_holidays, on=['date', 'city'], how='left')

    for col in ['national_holiday_type', 'regional_holiday_type', 'local_holiday_type']:
        df[col] = df[col].fillna('No Holiday')

    # 3.3 Добавляем нефть
    df = df.merge(df_oil, on='date', how='left')

    # 3.4 Добавляем транзакции
    df = df.merge(df_transactions, on=['date', 'store_nbr'], how='left')

    print("Объединение завершено.")

    # 4. Генерация признаков
    print("Генерация признаков...")

    # Заполнение нефти
    df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill()

    # Временные признаки

    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Флаг праздника
    df['is_holiday_overall'] = ((df['national_holiday_type'] != 'No Holiday') | \
                                (df['regional_holiday_type'] != 'No Holiday') | \
                                (df['local_holiday_type'] != 'No Holiday')).astype(int)


    # Транзакции (заполняем нулями, если в какой-то день их не было)
    df['transactions'] = df['transactions'].fillna(0)

    # Кодирование категорий
    categorical_cols = ['city', 'state', 'type', 'cluster', 'national_holiday_type', 'regional_holiday_type',
                        'local_holiday_type']

    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes + 1

    # 5. Финализация
    # Оставляем sales, но удаляем и id, так как для обучения они не нужны напрямую
    df_final = df.drop(columns=['id'])

    print(df_final.head(15))
    print(f"Подготовка завершена. Итоговая таблица: {df_final.shape}")
    # 6. Сохранение
    if not os.path.exists('data/preprocessing'):
        os.makedirs('data/preprocessing')

    df_final.to_csv('data/preprocessing/train_processed.csv', index=False)


if __name__ == "__main__":
    process_data()