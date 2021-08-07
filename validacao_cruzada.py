import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold

def validacao_cruzada(X,y, model, n_splits=5):
    'Do split dataset and calculate cross_score'
    X = np.array(X)
    y = np.array(y)
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2017).split(X, y))
    cross_validation_score = []
    for j, (train_idx, test_idx) in enumerate(folds):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_holdout = X[test_idx]
        y_holdout = y[test_idx]

        print ("Fit %s fold %d" % (str(model).split('(')[0], j+1))
        model.fit(X_train, y_train)
        cross_score = cross_val_score(model, X_holdout, y_holdout, cv=2, scoring='roc_auc')
        print("    cross_score: %.5f" % cross_score.mean())
        cross_validation_score.append(cross_score)
    return cross_validation_score

def validacao_5x2(model, X, y):
    X = np.array(X)
    y = np.array(y)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=12345)
    model_score = cross_validate(model, X, y, cv = cv,
                        scoring=['accuracy','precision_macro', 'recall_macro', 'f1_macro'],
                        return_train_score=True)
    return model_score