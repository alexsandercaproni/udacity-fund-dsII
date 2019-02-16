#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

# Imports do Scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Outros Imports
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
import matplotlib.pyplot as plt
from pprint import pprint

def explore_dataset():

    # Checando o dataset
    employees = data_dict.keys()
    employee_features = data_dict[employees[0]]

    # Numero de funcionarios no dataset
    print 'Numero total de data points:', len(employees)

    # Numero de features de cada funcionario
    print 'Numero de features de cada funcionario:', len(employee_features)

    # Checando o numero de POIS no dataset
    number_poi = 0
    poi_keys = []
    for i in data_dict:
        if data_dict[i]['poi'] == True:
            number_poi += 1
            poi_keys.append(data_dict[i]['poi'])
    print 'Numero de POIs: ', number_poi

    print 'Numero de non-POIs: ', len(employees) - number_poi

    # Checando se ha falta de informacao de salario
    non_salary = 0
    for i in data_dict:
        if data_dict[i]['salary'] == 'NaN':
            data_dict[i]['salary'] = 0
            non_salary += 1
    print 'Numero de funcionarios sem salario: ', non_salary

    # Checando se ha falta de informacao de salario
    non_bonus = 0
    for i in data_dict:
        if data_dict[i]['bonus'] == 'NaN':
            data_dict[i]['bonus'] = 0
            non_bonus += 1
    print 'Numero de funcionarios sem bonus: ', non_bonus

    # Busca Outliers
    ## Analisando os dados baseados em salario x bonus
    scatter_plot('salary', 'bonus')

    ## Verifica possiveis salarios discrepantes
    salario_outliers = []
    for key in data_dict:
        value = data_dict[key]['salary']
        if value != 'NaN':
            salario_outliers.append((key, int(value)))

    print 'Possiveis salarios discrepantes: '
    pprint(sorted(salario_outliers, key=lambda x: x[1], reverse=True)[:5])

    ## Remove do dataset
    data_dict.pop('TOTAL', 0)
    ### Baseado no PDF
    data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
    data_dict.pop("LOCKHART EUGENE E", 0)

    ## Checa se nao existe mais outliers
    scatter_plot('salary', 'bonus')

    # Cria nova feature
    create_features()

def create_features():
    # Feature que mede a fracao do salario pelo bonus
    for i in data_dict:
        salary = float(data_dict[i]['salary'])
        bonus = float(data_dict[i]['bonus'])
        if salary > 0 and bonus > 0:
            data_dict[i]['gain_fraction'] = data_dict[i]['salary'] / data_dict[i]['bonus']
        else:
            data_dict[i]['gain_fraction'] = 0

    # Adiciona nova feature
    features_list.extend(['gain_fraction'])

def choose_classifier(x):
    return {
        'naive_bayes': GaussianNB(),
        'decision_tree': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier(),
        'knn': KNeighborsClassifier(),
    }.get(x)


def classifier_with_pipeline(clf, num_k, f_list):
    data = featureFormat(my_dataset, f_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    # Tecnica de cross ideal para pequenos datasets
    shuffle = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)

    # Definicoes do Pipeline
    ## Normalizacao dos dados, utilizando MinMaxScaler
    ## Selecao das Features mais importantes, utilizando SelectKBest
    scaler = MinMaxScaler()
    kbest = SelectKBest(k=num_k)

    # Obtendo o algoritmo escolhido
    classifier = choose_classifier(clf)

    pipeline = Pipeline(steps=[('minmax_scaler', scaler), ('feature_selection', kbest), (clf, classifier)])

    # Seta os parametros focando em um melhor score f1
    parameters = []

    if clf == 'naive_bayes':
        parameters = dict()

    if clf == 'knn':
        parameters = dict(knn__n_neighbors=range(1, 8),
                          knn__algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'])

    if clf == 'random_forest':
        parameters = dict(random_forest__n_estimators=[25, 50],
                          random_forest__min_samples_split=[2, 3, 4],
                          random_forest__criterion=['gini', 'entropy'])

    if clf == 'decision_tree':
        parameters = dict(decision_tree__max_depth=[None, 5, 20],
                          decision_tree__min_samples_split=[2, 5, 10],
                          decision_tree__class_weight=['balanced'],
                          decision_tree__criterion=['gini', 'entropy'])

    # Obtem parametros otimizados
    gridSearch = GridSearchCV(pipeline, param_grid=parameters, scoring='f1', cv=shuffle)
    gridSearch.fit(features, labels)

    return gridSearch

def scatter_plot(axis_x, axis_y):
    # Create scatter plot, save to local path
    features = ['poi', axis_x, axis_y]
    data = featureFormat(data_dict, features)

    for point in data:
        x = point[1]
        y = point[2]
        if point[0]:
            plt.scatter(x, y, color="r")
        else:
            plt.scatter(x, y, color='b')

    plt.xlabel(axis_x)
    plt.ylabel(axis_y)
    plt.show()


# Lista de Features
# A primeira feature deve ser POI
features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'expenses', 'deferred_income',
                 'long_term_incentive', 'restricted_stock_deferred', 'shared_receipt_with_poi',
                 'loan_advances', 'from_messages', 'director_fees','total_stock_value',
                 'from_poi_to_this_person', 'from_this_person_to_poi', 'total_payments',
                 'exercised_stock_options', 'to_messages', 'restricted_stock']

# Carrega o conjunto de dados
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Explora conjunto de dados
explore_dataset()

# Salvando o conjunto de dados
my_dataset = data_dict

# Selecao de Features usando SelectKBest
## Extraindo features e labels do dataset para testes locais
data = featureFormat(my_dataset, features_list, sort_keys=True)

labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.30, random_state=42)

selector = SelectKBest(k=7)
selector.get_params()
selector.fit_transform(features_train, labels_train)
indices = selector.get_support(True)

final_list = ['poi']
for i in indices:
     print '%s, score: %f' % (features_list[i + 1], selector.scores_[i])
     final_list.append(features_list[i + 1])

# Acrescenta a nova feature (gain_fraction)
#final_list.append('gain_fraction')

# Atualizando a Feature List de acordo com as novas features
features_list = final_list

# Classificacao
## Tune your classifier to achieve better than .3 precision and recall using our testing script.
## Executando pipeline e dando tune nos parametros via GridSearchCV
## Algoritmos disponiveis: NaiveBayes (naive_bayes), Random Forest(random_forest) e Decision Tree (decision_tree)
cv = classifier_with_pipeline('naive_bayes', 7, features_list)

#print 'Melhores parametros: ', cv.best_params_
clf = cv.best_estimator_

# Valida o modelo com base na precisao, recall e F1-score
## Foi necessario mudar algumas linhas no codigo Tester.py
## Referencia:
## https://stackoverflow.com/questions/53899066/what-could-be-the-reason-for-typeerror-stratifiedshufflesplit-object-is-not

test_classifier(clf, my_dataset, features_list)

# Dump classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, features_list)