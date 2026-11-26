# 코드_공유.py
# 작업자 A, B, Admin 이 함께 사용하는 파일입니다.

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

# 2013년 뉴욕 출발 항공편 표본 50,000건입니다.
# 항공편 정보에 기상 정보와 기체 정보를 미리 결합해 두었습니다.
df = pd.read_csv('flights_ml.csv')

NUM = ['month', 'day', 'sched_hour', 'distance', 'temp', 'humid',
       'wind_speed', 'precip', 'visib', 'seats', 'engines']
CAT = ['carrier', 'origin']

X = pd.get_dummies(df[NUM + CAT], columns=CAT)
X = X.fillna(X.median(numeric_only=True))
y = df['is_delayed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(X_train.shape, X_test.shape)


####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

tree_params = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 15],
    "min_samples_leaf": [1, 5, 20],
}

tree_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=tree_params,
    scoring="f1",
    cv=5,
)

tree_search.fit(X_train, y_train)
tree_model = tree_search.best_estimator_

tree_pred = tree_model.predict(X_test)
tree_f1 = f1_score(y_test, tree_pred)
tree_accuracy = accuracy_score(y_test, tree_pred)

print("DecisionTree 최적 파라미터:", tree_search.best_params_)
print(f"DecisionTree 교차검증 F1: {tree_search.best_score_:.4f}")
print(f"DecisionTree 테스트 F1: {tree_f1:.4f}")
print(f"DecisionTree 테스트 Accuracy: {tree_accuracy:.4f}")

tree_importance = pd.Series(
    tree_model.feature_importances_,
    index=X_train.columns,
)
tree_top10 = tree_importance.nlargest(10).sort_values()

ax = tree_top10.plot.barh(figsize=(9, 5))
ax.set_title("DecisionTree Feature Importance (Top 10)")
ax.set_xlabel("Feature importance")
plt.tight_layout()
plt.show()

####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''
