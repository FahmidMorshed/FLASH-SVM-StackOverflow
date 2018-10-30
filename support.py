import timeit
import pandas as pd

from tuner import DE_Tune_ML
from tuner_flash import FLASH_Tune
import numpy as np
from sklearn import svm, metrics
from sklearn.cluster import KMeans
from utility import study, preprocess_data, parse_classification_report, classifaction_report_csv
import os
from results import results_process
import experiment
def tune_learner(learner, train_X, train_Y, tune_X, tune_Y, goal,
                 target_class=None):
    """
    :param learner:
    :param train_X:
    :param train_Y:
    :param tune_X:
    :param tune_Y:
    :param goal:
    :param target_class:
    :return:
    """
    if not target_class:
        target_class = goal
    clf = learner(train_X, train_Y, tune_X, tune_Y, goal)
    tuner = DE_Tune_ML(clf, clf.get_param(), goal, target_class)
    return tuner.Tune()


def cross_val_DE_SVM(train_index, tune_index, trn_pd, tst_pd, lrnr, gl, my_F={}):# goal, F, isFlash=False):
    start_fold = timeit.default_timer()
    train_data = trn_pd.iloc[train_index]
    tune_data = trn_pd.iloc[tune_index]
    train_x = train_data.loc[:, "Output"].values
    train_y = train_data.loc[:, "LinkTypeId"].values
    tune_x = tune_data.loc[:, "Output"].values
    tune_y = tune_data.loc[:, "LinkTypeId"].values
    test_x = tst_pd.loc[:, "Output"].values
    test_y = tst_pd.loc[:, "LinkTypeId"].values
    params, evaluation = tune_learner(lrnr, train_x, train_y, tune_x,  # Runs DE
                                      tune_y, gl)
    clf = lrnr(train_x, train_y, test_x, test_y, gl)
    my_F = clf.learn(my_F, **params)
    stop_fold = timeit.default_timer()
    print("Fold finished. Time: ", stop_fold - start_fold)
    print("Eval: ", evaluation)
    return clf


def cross_val_FLASH_SVM(train_index, tune_index, trn_pd, tst_pd, lrnr, gl, my_F={}):# goal, F, isFlash=False):
    start_fold = timeit.default_timer()
    train_data = trn_pd.iloc[train_index]
    tune_data = trn_pd.iloc[tune_index]
    train_x = train_data.loc[:, "Output"].values
    train_y = train_data.loc[:, "LinkTypeId"].values
    tune_x = tune_data.loc[:, "Output"].values
    tune_y = tune_data.loc[:, "LinkTypeId"].values
    test_x = tst_pd.loc[:, "Output"].values
    test_y = tst_pd.loc[:, "LinkTypeId"].values
    params, evaluation = tune_learner_FLASH(lrnr, train_x, train_y, tune_x,  # Runs DE
                                      tune_y, gl)
    clf = lrnr(train_x, train_y, test_x, test_y, gl)
    my_F = clf.learn(my_F, **params)
    stop_fold = timeit.default_timer()
    print("Fold finished. Time: ", stop_fold - start_fold)
    print("Eval: ", evaluation)
    return clf



def tune_learner_FLASH(learner, train_X, train_Y, tune_X, tune_Y, goal,
                 target_class=None):
    """
    :param learner:
    :param train_X:
    :param train_Y:
    :param tune_X:
    :param tune_Y:
    :param goal:
    :param target_class:
    :return:
    """
    if not target_class:
        target_class = goal
    clf = learner(train_X, train_Y, tune_X, tune_Y, goal)
    tuner = FLASH_Tune(clf, clf.get_param(), goal, target_class)
    return tuner.Tune()





def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=1)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_
        # print(str(i+1) + ": " + str(origDisp))

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append(
            {'clusterCount': k, 'gap': gap}, ignore_index=True)

    # return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
    return gaps.argmax()











def results_SVM(predicted, test_Y, filename):
    # predicted = clf.predict(test_X)
    # labels: ["Duplicates", "DirectLink","IndirectLink", "Isolated"]
    report_gen = metrics.classification_report(
        test_Y, predicted, labels=["1", "2", "3", "4"], digits=3)
    print(report_gen)
    classifaction_report_csv(report_gen, filename)
    parsed_report = parse_classification_report(report_gen)
    return parsed_report



def get_IFA_score(actual, predicted):
    """
    get Initial False Alarm score
    :param actual: list of values
    :param predicted: ndarray of values
    :return: all_k: Panda DataFrame value of all IFA for each class
    """
    all_k_dict = []
    for cls in range(1, 5):
        k = 0
        for i, pred_cls in enumerate(predicted):
            if str(pred_cls) == str(cls):
                k += 1
                if str(actual[i]) == str(cls):
                    all_k_dict.append(k)
                    break;
    return all_k_dict


def get_acc(cm):
    """
    gets the accuracy for each class
    :param cm: Confusion Matrix
    :return: array of accouracy
    """
    out = []
    for i in range(4):
        out.append(cm[i][i] / 400)
    return out

def get_acc_str(acc_list):
    acc_str = "Accuracy: \n"
    for i, x in enumerate(acc_list):
        acc_str += str(i+1) + ": " + str(x) + "\n"
    return acc_str


def print_results(clfs, stop, start, filename="output.txt"):
    file_name = os.path.sep.join([filename])
    content = ""
    content += "\nITERATION NUMBER: " + str(experiment.iteration_num) + "\n============================================\n"
    for each in clfs:
        content += each.confusion
    print(content)
    content += "\nTime: " + str(stop - start)
    print("Model training time: ", stop - start)
    with open(file_name, "a+") as f:
        f.write(content)

    results_process.reports(file_name)

def print_results_simple(content, filename="output_simple.txt"):
    file_name = os.path.sep.join([filename])
    content = "\nITERATION NUMBER: " + str(experiment.iteration_num) + "\n============================================\n" + content

    print(content)
    with open(file_name, "a+") as f:
        f.write(content)

def total_summary(result_set, num_rows, start, end):
    weightedAvgs = [0, 0, 0]
    for l in result_set:
        avg_list = l['avg']
        for i in range(3):
            support_count = avg_list[3]
            weightedAvgs[i] += (avg_list[i] * support_count) / num_rows

    result = {}
    result['precision'] = weightedAvgs[0]
    result['recall'] = weightedAvgs[1]
    result['f1'] = weightedAvgs[2]
    print(result)
    print("Time: ", end - start)

def ifa_score_str(ifa):
    my_str = "IFA Score: \n"
    for i, x in enumerate(ifa):
        my_str += str(i+1) + ": " + str(x) + "\n"
    return my_str


def note_it(filename, content, same_run=True):
    file_name = os.path.sep.join([filename])

    if same_run is False:
        content = "\nITERATION NUMBER: " + str(experiment.iteration_num) + "\n============================================\n" + content
    print(content)
    with open(file_name, "a+") as f:
        f.write(content)