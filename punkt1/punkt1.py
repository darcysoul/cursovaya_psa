import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import os


def create_dataframe():
    """Создание DataFrame с данными по посещениям."""
    df = pd.DataFrame(columns=["Год", "Посещения, чел."])
    df.iloc[:, 0] = list(range(2018, 2023))
    df.iloc[:, 1] = [66328543, 62475975, 43695030, 52455343, 50910284]
    return df


def analyze_statistical_indicators(dataframe, mode):
    """Анализ статистических индикаторов: абсолютный прирост,
    темп роста и темп прироста.
    """
    copy_df = dataframe.copy()
    n = len(dataframe)
    delta_y = [None] * n
    growth_rate = [None] * n
    increase_rate = [None] * n

    for i in range(1, n):
        y = copy_df.iloc[i - 1, 1] if mode == "chain" else copy_df.iloc[0, 1]
        y_i = copy_df.iloc[i, 1]
        delta_y[i] = y_i - y
        growth_rate[i] = round((y_i / y) * 100, 1)
        increase_rate[i] = round(((y_i - y) / y) * 100, 1)

    copy_df["Абсолютный прирост"] = delta_y
    copy_df["Темп роста, %"] = growth_rate
    copy_df["Темп прироста, %"] = increase_rate
    return copy_df


def save_table_to_csv(dataframe, filename):
    """Сохранение таблицы в формате CSV."""
    filepath = os.path.join('punkt1', filename)
    dataframe.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"Таблица сохранена в {filepath}\n")


def print_table(dataframe):
    """Вывод таблицы в консоль."""
    table_data = dataframe.values.tolist()
    headers = dataframe.columns.tolist()
    print(tabulate(table_data, headers=headers, tablefmt='fancy_grid'))


def calculate_means(dataframe):
    """Расчет среднего уровня ряда и показателей изменения."""
    y_first = dataframe.iloc[0, 1]
    y_last = dataframe.iloc[-2, 1]
    n = len(dataframe) - 2

    y_mean = round(dataframe.iloc[:n + 1, 1].mean(), 2)
    y_mean_abs_increase = round((y_last - y_first) / n, 2)
    mean_growth_rate = round(((y_last / y_first) ** (1 / n)) * 100, 1)
    mean_increase_rate = round(mean_growth_rate - 100, 1)

    print(f'Средний уровень ряда: {int(np.ceil(y_mean))} (чел.)')
    print(f'Средний абсолютный прирост: {int(np.ceil(y_mean_abs_increase))} (чел.)')
    print(f'Средний темп роста: {mean_growth_rate}%')
    print(f'Средний темп прироста: {mean_increase_rate}%\n')

    return {
        "y_mean": y_mean,
        "y_mean_abs_increase": y_mean_abs_increase,
        "mean_growth_rate": mean_growth_rate,
        "mean_increase_rate": mean_increase_rate
    }


def forecast(dataframe, means):
    """Прогноз на 2022 год по различным методам."""
    y_first = dataframe.iloc[0, 1]
    y_last = dataframe.iloc[-2, 1]
    n = len(dataframe) - 2

    # Прогноз на основе среднего абсолютного прироста
    y2022_by_abs_inc = round(y_last + means["y_mean_abs_increase"], 1)

    # Прогноз на основе среднего темпа роста
    y2022_by_mean_growth = round(
        y_last * ((y_last / y_first) ** (1 / n)), 2
    )

    # Метод наименьших квадратов (МНК)
    total = len(dataframe) - 1
    t_mean = sum(range(1, total + 1)) / total
    t2_mean = sum([i * i for i in range(1, total + 1)]) / total
    y = dataframe.iloc[:n + 1, 1]
    y_mean = y.mean()
    ty_mean = sum([t * y.iloc[t - 1] for t in range(1, total + 1)]) / total

    b = (ty_mean - y_mean * t_mean) / (t2_mean - t_mean * t_mean)
    a = y_mean - b * t_mean

    # Прогноз на основе МНК
    y2022_by_mls = round(b * len(dataframe) + a, 2)

    print(f'Прогноз по среднему абсолютному приросту: {int(np.ceil(y2022_by_abs_inc))} (чел.)')
    print(f'Прогноз по среднему темпу роста: {int(np.ceil(y2022_by_mean_growth))} (чел.)')
    print(f'Прогноз по МНК: {int(np.ceil(y2022_by_mls))} (чел.)\n')

    return {
        "y2022_by_abs_inc": y2022_by_abs_inc,
        "y2022_by_mean_growth": y2022_by_mean_growth,
        "y2022_by_mls": y2022_by_mls
    }, a, b


def calculate_forecast_errors(dataframe, predicts):
    """Расчет относительной погрешности прогнозов."""
    y_fact = dataframe.iloc[-1, 1]

    sigma_abs_inc = round((abs(predicts["y2022_by_abs_inc"] - y_fact) / y_fact) * 100, 2)
    sigma_temp_growth = round((abs(predicts["y2022_by_mean_growth"] - y_fact) / y_fact) * 100, 2)
    sigma_mls = round((abs(predicts["y2022_by_mls"] - y_fact) / y_fact) * 100, 2)

    print(f"Относительная погрешность по среднему абсолют. приросту: {sigma_abs_inc}%")
    print(f"Относительная погрешность по среднему темпу роста: {sigma_temp_growth}%")
    print(f"Относительная погрешность по МНК: {sigma_mls}%\n")


def plot_visits_trend(dataframe, a, b):
    """Построение графика количества посещений по годам и
    линейного тренда.
    """
    years = dataframe['Год']
    attend_counts = dataframe.iloc[:, 1]
    t_values = np.arange(1, len(years) + 1)

    plt.figure(figsize=(12, 8))
    plt.scatter(years, attend_counts, color='blue', marker='o')

    for i, txt in enumerate(attend_counts):
        plt.annotate(txt, (years[i], attend_counts[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    linear_line = b * t_values + a
    plt.plot(years, linear_line, color='red', label=f'Прямая: y = {b}*t + {a}')

    plt.title('Количество посещений по годам')
    plt.xlabel('Год')
    plt.ylabel('Человек')
    plt.legend()
    plt.show()


def main():
    # Создаем DataFrame
    df = create_dataframe()

    # Выводим исходные данные
    print_table(df)
    save_table_to_csv(df, 'original_data.csv')

    # Базисный анализ
    df_base = analyze_statistical_indicators(df, "basis")
    print_table(df_base)
    save_table_to_csv(df_base, 'base_analysis.csv')

    # Цепной анализ
    df_chain = analyze_statistical_indicators(df, "chain")
    print_table(df_chain)
    save_table_to_csv(df_chain, 'chain_analysis.csv')

    # Расчет средних значений и прогнозов
    means = calculate_means(df)
    predicts, a, b = forecast(df, means)

    # Расчет погрешностей
    calculate_forecast_errors(df, predicts)

    # Построение графика
    plot_visits_trend(df, a, b)


if __name__ == "__main__":
    main()
