import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

# Загрузка данных
data = pd.read_csv('cleaned.csv')  # Замените на ваш файл с данными

# Предполагается, что у вас есть столбец 'target' с метками классов
X = data.drop(columns=['target'])  # Признаки
y = data['target']  # Целевой столбец

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Инициализация модели
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Обучение модели
model.fit(X_train, y_train)

# Оценка модели на тестовой выборке
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# Сохранение обученной модели
joblib.dump(model, 'trained_model.pkl')





# Загрузка очищенного датасета
cleaned_df = pd.read_csv('cleaned.csv')  # Замените на ваш путь к файлу

# Подготовка данных для предсказания (удаление целевого столбца, если он есть)
X_new = cleaned_df.drop(columns=['target'], errors='ignore')  # Удалите 'target', если есть

# Загрузка сохраненной модели
model = joblib.load('trained_model.pkl')

# Применение модели для предсказания вероятностей
cleaned_df['score'] = model.predict_proba(X_new)[:,1]  # Вероятности для класса 1

# Сохранение результатов в новый CSV файл
cleaned_df.to_csv('predicted_scores.csv', index=False)


# Подсчет количества признаков target с значением 1
count_target_1 = (cleaned_df['target'] == 1).sum()

# Подсчет количества признаков score с значением >= 0.5
count_score_ge_0_5 = (cleaned_df['score'] >= 0.5).sum()

# Вывод результатов
print(f'Количество признаков target с значением 1: {count_target_1}')
print(f'Количество признаков score с значением >= 0.5: {count_score_ge_0_5}')





# Для расчета ROC-AUC на тестовых данных
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Рассчет ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC: {roc_auc}')

# Отчет классификации
print(classification_report(y_test, y_pred))

# Строим график ROC
RocCurveDisplay.from_predictions(y_test, y_pred_proba)
plt.show()