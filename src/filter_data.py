import os
import pandas as pd
import numpy as np
from process_data import process_data

def filter_data(target_store_id: int, target_family_name: str):
    """
    Загружает предобработанный датасет,
    фильтрует его для конкретного магазина и семейства товаров,
    удаляет константные столбцы и сохраняет результат в CSV файл.
    """
    print(
        f"Начинаем фильтрацию и сохранение данных для магазина ID: {target_store_id}, семейства: '{target_family_name}'")

    # 1. Получение полного предобработанного датасета
    print("1. Загрузка ранее предобработанных данных")
    try:
        df_full_processed = pd.read_csv('data/preprocessing/train_processed.csv', parse_dates=['date'])
    except FileNotFoundError as e:
        print("1. Предобработанные данные не найдены. Запускаем полную предобработку (это может занять время)...")
        df_full_processed = process_data()  # Вызов функции из data_processor.py

    # 2. Фильтрация по магазину и товару
    print(f"2. Фильтрация данных для магазина ID: {target_store_id} и семейства: '{target_family_name}'...")
    df_single_ts = df_full_processed[
        (df_full_processed['store_nbr'] == target_store_id) &
        (df_full_processed['family'] == target_family_name)].copy()

    if df_single_ts.empty:
        raise ValueError(
            f"Нет данных для магазина ID: {target_store_id} и семейства: '{target_family_name}' после фильтрации. Проверьте входные параметры.")

    # 3. Удаление константных столбцов
    print("3. Проверка и удаление константных столбцов...")
    # Устанавливаем дату как индекс, чтобы ее не учитывать при поиске константных столбцов
    df_single_ts_temp = df_single_ts.set_index('date').sort_index()

    df_single_ts = df_single_ts.drop(columns=['national_holiday_type', 'local_holiday_type'])

    constant_columns = [col for col in df_single_ts_temp.columns if df_single_ts_temp[col].nunique() == 1]

    # Исключаем 'sales' из списка для удаления, если она вдруг оказалась константой
    if 'sales' in constant_columns:
        constant_columns.remove('sales')

    if constant_columns:
        print(f"   Обнаружены константные столбцы, которые будут удалены: {constant_columns}")
        # Удаляем из исходного df_single_ts (до установки индекса)
        df_single_ts = df_single_ts.drop(columns=constant_columns)
    else:
        print("   Константных столбцов не обнаружено.")

    # 4. Финализация (установка даты как индекса)
    df_single_ts = df_single_ts.set_index('date').sort_index()

    print("\nПредварительный просмотр отфильтрованной таблицы:")
    print(df_single_ts.head(15))
    print(f"Отфильтровка завершена. Итоговая таблица: {df_single_ts.shape}")

    # 5. Сохранение отфильтрованных данных
    # Сохраняем с индексом, чтобы дата была в файле
    df_single_ts.to_csv('data/preprocessing/train_filtered.csv', index=True)
    #print(f"Отфильтрованные данные сохранены в '{filtered_file_path}'")
    print("--- Фильтрация и сохранение завершены ---")
    #return filtered_file_path


if __name__ == "__main__":
    TARGET_STORE_ID = 1
    TARGET_FAMILY_NAME = 'BEVERAGES'
    filter_data(TARGET_STORE_ID, TARGET_FAMILY_NAME)

