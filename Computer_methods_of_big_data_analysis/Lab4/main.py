import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

#загрузка данных
data = pd.read_csv('spaceship_titanic.csv')

#экспертный анализ данных
print("Информация о датасете:")
print(data.info())
print("\nСтатистические характеристики:")
print(data.describe())
print("\nКоличество пропущенных значений:")
print(data.isnull().sum())

#удаление малозначащих данных
data = data.drop(columns=['PassengerId'])

#замена пропущенных значений
data['CryoSleep'] = data['CryoSleep'].fillna(1).astype(int)

#приведение типов к boolean
data['CryoSleep'] = data['CryoSleep'].astype('boolean')
data['VIP'] = data['VIP'].astype('boolean')

#создание колонки FamilySize до преобразования категориальных переменных
data['FamilySize'] = data.groupby('Cabin')['Cabin'].transform('count') + data.groupby('Name')['Name'].transform('count') - 1

#обработка категориальных признаков
data = pd.get_dummies(data, columns=['HomePlanet', 'Destination', 'Cabin', 'Name'], drop_first=True)

#выделение целевой функции от датасета
X = data.drop(columns=['Transported'])
y = data['Transported']

#разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#нормализация данных
scaler = StandardScaler()
X_train[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = scaler.fit_transform(
    X_train[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
)
X_test[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = scaler.transform(
    X_test[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
)

#обучение моделей Random Forest и XGBoost для решения задачи бинарной классификации
#Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)
print(f'Random Forest Accuracy: {rf_accuracy:.4f}')

#XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_accuracy = xgb_model.score(X_test, y_test)
print(f'XGBoost Accuracy: {xgb_accuracy:.4f}')
