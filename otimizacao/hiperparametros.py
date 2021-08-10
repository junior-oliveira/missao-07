from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
def melhores_hiperparametros(modelo, hiperparametros, X_train, X_vld, y_train, y_vld):
    grid_search = GridSearchCV(estimator=modelo,
                           param_grid=hiperparametros, scoring=['accuracy','precision_macro', 'recall_macro', 'f1_macro'],
                           refit='precision_macro',
                           cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=12345))

    grid_search.fit(X_train, y_train)
    y_true, y_pred = y_vld, grid_search.predict(X_vld)
    print(classification_report(y_true, y_pred))
    df=pd.DataFrame.from_dict(grid_search.best_params_,orient='index').transpose()
    return df

# conjunto de hiperparâmetros para a otimização
def conjunto_de_hiper_parametros():
    hyper_set = {}

    # Parâmetros SVM
    C = [0.1, 1.0 , 10] # Parâmetros para todos os kernel
    gamma = [0.1, 0.5, 2, 5] # Parêmtros para os kernels polinomial, rbf, sigmoide
    coef0 = [0.01, 0.05, 0.9, 2] # parâmetro para os kernel polinomial e sigmoide
    degree = [2, 4, 7, 10] # parâmetro para o kernel polinomial

    svm_params = [{'kernel': ['rbf'], 
                     'C': C},
                    {'kernel': ['poly'], 'C': C},
                     {'kernel': ['sigmoid'], 'C': C},
                    {'kernel': ['linear'], 'C': C}]
    # svm_params = [{'kernel': ['rbf']},
    #                 {'kernel': ['poly']},
    #                  {'kernel': ['sigmoid']},
    #                 {'kernel': ['linear']}]

    hyper_set['svm'] = svm_params 

    # Parâmetros K-NN
    knn_params = [{'n_neighbors': np.arange(1,38,2)}]

    hyper_set['knn'] = knn_params

    lvq_params = [{'prototypes_per_class': [1, 2, 3, 4]}]

    hyper_set['lvq'] = lvq_params

    # Parâmetros para o MLP

    neurons = [2, 5, 10, 20] # número de neurônios nas camadas escondidas
    hidden_layers = [1, 2, 3, 4] # número de camadas escondidas
    activation = ['identity', 'logistic', 'tanh', 'relu'] # Funções de ativação
    learning_rate_init = [0.0005,0.01,0.1,0.3] # taxa de aprendizagem
    hidden_layer_sizes = tuple(np.repeat(neurons, hidden_layers))

    mlp_params = [{'hidden_layer_sizes': hidden_layer_sizes,
                        'activation': activation, 
                        'learning_rate_init': learning_rate_init}]
    hyper_set['mlp'] = mlp_params

    # Árvore de decisão
    criterion = ['gini', 'entropy']
    max_depth = list(range(1, 10))
    min_samples_split = list(range(1, 10))
    min_samples_leaf = list(range(1, 5))


    tree_params = [{'criterion': criterion,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}]
    hyper_set['tree'] = tree_params


    hyper_set['tree']

    return hyper_set

# Teste de otimizador 
def best_hyper(modelo, hiperparametros, X_train, X_vld, y_train, y_vld):
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.svm import SVC

    print(__doc__)


    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            modelo, hiperparametros, scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_vld, clf.predict(X_vld)
        print(classification_report(y_true, y_pred))
        print()
