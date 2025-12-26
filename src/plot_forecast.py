import numpy as np
import matplotlib.pyplot as plt
import os

def plot_forecast(y_train, y_test, y_pred, path_prediction_plot, name_prediction_plot, focus_last=100):  # Строит график фактических и прогнозных значений
    y_full = np.concatenate([y_train, y_test])
    n_total = len(y_full)
    split_point = len(y_train)

    x_indices = np.arange(n_total)

    # Определяем границы видимой области по X
    visible_start = max(0, n_total - focus_last)
    visible_end = n_total

    plt.figure(figsize=(18, 7))

    # 1. Фактические значения (полный ряд) - Зеленая линия
    plt.plot(x_indices, y_full, label='Actual values', color='green', linewidth=1.5)

    # 2. Прогноз (только тестовый период) - Синяя линия
    plt.plot(x_indices[split_point:], y_pred, label='Predicted', color='blue', linewidth=2.5)

    # 3. Вертикальная линия, показывающая разделение Train/Test
    plt.axvline(x=split_point, color='black', linestyle='-', linewidth=1)

    # Динамическое ограничение по оси Y (сверху и снизу)

    # 1. Выделяем значения в видимой области
    visible_actual = y_full[visible_start:visible_end]
    pred_start_index = max(0, visible_start - split_point)
    visible_pred = y_pred[pred_start_index:] if pred_start_index < len(y_pred) else np.array([])

    all_visible_values = np.concatenate([visible_actual, visible_pred])

    if len(all_visible_values) > 0:
        # Ограничение сверху: max + 500
        y_max = np.nanmax(all_visible_values)

        # Ограничение снизу: min - 50 (но не ниже 0)
        y_min = np.nanmin(all_visible_values)

        # Добавляем запас снизу, но ограничиваем нулем
        y_min_limit = max(0, y_min - 200)

        plt.ylim(y_min_limit, y_max + 200)
    else:
        # Если данных нет, оставляем автоматическое масштабирование
        plt.ylim(0, plt.ylim()[1])

    plt.title(name_prediction_plot, fontsize=16)
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True)
    plt.xlabel('Индекс времени (Time Index)', fontsize=12)
    plt.ylabel('Продажи (Sales)', fontsize=12)

    # Ограничение оси X для фокуса
    plt.xlim(visible_start, visible_end)

    plt.tight_layout()

    directory_to_create = os.path.dirname(path_prediction_plot)
    # Проверяем, существует ли директория
    if not os.path.exists(directory_to_create):
        # Если директории не существует, создаем ее
        os.makedirs(directory_to_create, exist_ok=True)
        print(f"Директория '{directory_to_create}' создана.")

    plt.savefig(path_prediction_plot, bbox_inches='tight')
    plt.close()