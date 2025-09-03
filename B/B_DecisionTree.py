from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from Utility.utils import TestModel , CM_Display

def B_DT_Train(X_train, X_val, X_test, y_train, y_val, y_test, class_names):
    model = DecisionTreeClassifier(random_state=0)
    params = {
        "criterion": ["entropy"],
        "min_samples_split": [23],  # 2..30
        "max_features": [12],       # 1..29
        "max_depth": [7],          # 1..29
    }
    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    cm = TestModel(y_test,y_pred,class_names)
    CM_Display(cm,class_names,'CM_B_DT')
    