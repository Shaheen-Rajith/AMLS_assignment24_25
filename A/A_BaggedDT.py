from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from Utility.utils import TestModel , CM_Display

def A_BaggingDT(X_train, X_val, X_test, y_train, y_val, y_test, class_names):
    '''
    
    '''
    base_model = DecisionTreeClassifier(random_state=0)
    bag_model = BaggingClassifier(estimator=base_model, random_state=0, n_jobs=-1)

    params = {
        "estimator__criterion": ["entropy"],
        "max_features": [12],
        "n_estimators": [25],
        "max_samples": [1.0],
        "bootstrap": [True],           
    }

    grid = GridSearchCV(
        estimator=bag_model,
        param_grid=params,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    cm = TestModel(y_test,y_pred,class_names)
    CM_Display(cm,class_names,'CM_A_Bag')