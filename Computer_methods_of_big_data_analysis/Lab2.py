import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

#настройки отображения данных в pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#чтение данных из excel
df = pd.read_excel('data.xlsx')

#замена специальных символов и преобразование столбцов в числовые значения
df[['x1', 'x2', 'x3', 'x4', 'x5']] = df[['x1', 'x2', 'x3', 'x4', 'x5']].replace('\u00a0', '', regex=True).apply(pd.to_numeric, errors='coerce')

#заполнение пропусков в столбце 'Округ'
df['Округ'] = df['Округ'].fillna(df['Округ'].mode()[0])

#заполнение пропусков в остальных числовых столбцах медианой
for col in ['x1', 'x2', 'x3', 'x4', 'x5']:
    df[col] = df[col].fillna(df[col].median())
    #вывод DataFrame
    print(df)

def delete_outliers(feature):
    #визуализация данных, удаляет выбросами
    _x = df.boxplot(column=feature)
    plt.close('all')
    #получение границ усов
    whiskers = []
    for line in _x.lines:
        if line.get_linestyle() == '-':
            whiskers.append(line.get_ydata())

    whiskers = whiskers[1:3]

    #определяем нижнюю и верхнюю границы усов
    lower_whisker = whiskers[0][1]
    upper_whisker = whiskers[1][1]

    #находим медиану
    median = np.median(df[feature])
    return np.where((df[feature] > upper_whisker) | (df[feature] < lower_whisker), median, df[feature])

#обработка всех признаков для удаления выбросов
features = ['x1', 'x2', 'x3', 'x4', 'x5']
for f in features:
   df[f] = delete_outliers(f)

#создание boxplot для всех признаков
plt.boxplot(df[features].values, whis=(0, 100))
plt.xticks(range(1, len(features) + 1), features)

#сохранение графика в файл
plt.savefig('boxplot.png')