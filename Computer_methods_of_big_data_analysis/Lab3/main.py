import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#загрузка данных
data = pd.read_csv('Titanic-Dataset.csv')

#экспертный анализ данных
print("Информация о датасете:")
print(data.info())
print("\nСтатистические характеристики:")
print(data.describe())
print("\nКоличество пропущенных значений:")
print(data.isnull().sum())

#удаление малозначащих данных
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#обработка категориальных признаков
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

#замена пропущенных значений
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)

#отделение целевой функции от датасета
X = data.drop('Survived', axis=1) #признаки
y = data['Survived'] #целевая переменная

#разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#линейная регрессия
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
linear_mse = mean_squared_error(y_test, y_pred_linear)
print(f'\nСреднеквадратичная ошибка линейной регрессии: {linear_mse}')

#логистическая регрессия
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
print(f'Точность логистической регрессии: {logistic_accuracy}')
print("Отчет классификации логистической регрессии:\n", classification_report(y_test, y_pred_logistic))

#лассо-регрессия
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
print(f'Среднеквадратичная ошибка Лассо-регрессии: {lasso_mse}')

#преобразование предсказаний Лассо в бинарный формат
y_pred_lasso_binary = np.where(y_pred_lasso >= 0.5, 1, 0)
lasso_accuracy = accuracy_score(y_test, y_pred_lasso_binary)
print(f'Точность Лассо-регрессии: {lasso_accuracy}')
