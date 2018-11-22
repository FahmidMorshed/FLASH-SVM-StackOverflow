import os
import random
import timeit
import warnings
import zipfile

import gensim
import wget
import numpy as np
from sklearn import svm, metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from convert import *
from support import *
from results import results_process
from learners import SK_SVM
from tuner import DE_Tune_ML
from tuner_flash import FLASH_Tune
from utility import study, preprocess_data, parse_classification_report, classifaction_report_csv
from multiprocessing import Queue
import multiprocessing as mp
import pandas as pd
from functools import partial
import process

###FAHID###
NUM_OF_ITERATION = 10
iteration_num = 0
DEF_FOLD = 10
@study
def run_FLASH_SVM(train_pd, test_pd, filename, onKmeans, repeats=1, fold=DEF_FOLD):
    """
    :param train_pd:Panda Dataframe, train data
    :param test_pd:Panda Dataframe, test data
    :param repeats:int, number of repeats
    :param fold: int,number of folds
    :return: None
    """


    learner = [SK_SVM][0]
    goal = {0: "PD", 1: "PF", 2: "PREC", 3: "ACC", 4: "F", 5: "G", 6: "Macro_F",
            7: "Micro_F"}[6]
    print("Goal: ", goal)
    F = {}
    clfs = []
    start_flash_svm = timeit.default_timer()
    for i in range(repeats):  # repeat n times here
        skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=0)

        SVM_cross_val_FLASH_SVM = partial(cross_val_FLASH_SVM, trn_pd=train_pd, tst_pd=test_pd, lrnr=learner, gl=goal, my_F=F)
        num_cpu = mp.cpu_count()
        print("num of cpu: ", num_cpu)
        unpack_y = train_pd.loc[:, "LinkTypeId"].values
        with mp.Pool(num_cpu) as p:
            clf = p.starmap_async(SVM_cross_val_FLASH_SVM, skfolds.split(train_pd, unpack_y)).get()
        for x in clf:
            clfs.append(x)

    stop_flash_svm = timeit.default_timer()
    if onKmeans is False:
        print_results(clfs, stop_flash_svm, start_flash_svm, filename)
    return clfs


####Shuvodeep###

def run_kmeans(train_pd, test_pd, filename, isFlash=False):
    csv_filename = filename.split(".")[0]
    avg_filename = csv_filename + "_avg.csv"
    csv_filename += ".csv"

    train_X = train_pd.loc[:, "Output"].tolist()

    start = timeit.default_timer()
    numClusters = optimalK(pd.DataFrame(train_X))
    stop = timeit.default_timer()
    note_str = "Found optimal k: " + str(numClusters) + "\nTime: " + str(stop - start) + "\n"
    note_it(filename, note_str, False)
    clf = KMeans(n_clusters=numClusters,
                 init='k-means++', max_iter=200, n_init=1)

    start0 = timeit.default_timer()
    clf.fit(train_X)
    stop0 = timeit.default_timer()

    svm_models = []  # maintain a list of svms
    train_pd['clabel'] = clf.labels_

    start1 = timeit.default_timer()
    for l in range(numClusters):
        cluster_training_time = timeit.default_timer()
        cluster = train_pd.loc[train_pd['clabel'] == l]
        if isFlash:
            clfs = run_FLASH_SVM(cluster, test_pd, filename, True)
        else:
            clfs = run_tuning_SVM(cluster, test_pd, filename, True)
        svm_models.append(clfs)
        cluster_training_time_end = timeit.default_timer()
        note_it(filename, "Cluster Time: " + str(cluster_training_time_end - cluster_training_time) + "\n", True)
    stop1 = timeit.default_timer()
    note_str = "Train Time: " + str(stop1 - start1) + "\n"
    note_it(filename, note_str)

    svm_results = []  # maintain a list of svm results
    test_X = test_pd.loc[:, "Output"].tolist()
    predicted = clf.predict(test_X)
    test_pd['clabel'] = predicted

    avg_predicted = []
    avg_cluster_Y = []
    for i in range(len(svm_models[l])):
        total_predicted = []
        total_cluster_Y = []
        for l in range(numClusters):
            cluster = test_pd.loc[test_pd['clabel'] == l]
            svm_model = svm_models[l][i]
            cluster_X = cluster.loc[:, "Output"].tolist()
            cluster_Y = cluster.loc[:, "LinkTypeId"].tolist()
            total_cluster_Y = np.append(total_cluster_Y, cluster_Y)
            avg_cluster_Y = np.append(avg_cluster_Y, cluster_Y)
            predicted_C = svm_model.learner.predict(cluster_X)
            total_predicted = np.append(total_predicted, predicted_C)
            avg_predicted = np.append(avg_predicted, predicted_C)
        svm_results.append(results_SVM(total_predicted, total_cluster_Y, csv_filename))  # store all the SVM result report in a dictionary
    svm_results.append(results_SVM(avg_predicted, avg_cluster_Y, avg_filename))
    # # call the helper method to summarize the svm results
    stop_end = timeit.default_timer()

    note_str = "Test Time: " + str(stop_end - stop1) + "\n"
    note_it(filename, note_str)
    total_summary(svm_results, test_pd.shape[0], start, stop_end)





@study
def run_tuning_SVM(train_pd, test_pd, filename, onKmeans, repeats=1, fold=DEF_FOLD):
    """
    :param train_pd:Panda Dataframe, train data
    :param test_pd:Panda Dataframe, test data
    :param repeats:int, number of repeats
    :param fold: int,number of folds
    :return: None
    """

    learner = [SK_SVM][0]
    goal = {0: "PD", 1: "PF", 2: "PREC", 3: "ACC", 4: "F", 5: "G", 6: "Macro_F",
            7: "Micro_F"}[6]
    print("Goal: ", goal)
    F = {}
    clfs = []
    start_tune_svm = timeit.default_timer()
    for i in range(repeats):  # repeat n times here
        skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=0)

        SVM_cross_val_DE_SVM = partial(cross_val_DE_SVM, trn_pd=train_pd, tst_pd=test_pd, lrnr=learner, gl=goal, my_F=F)
        num_cpu = mp.cpu_count()
        print("num of cpu: ", num_cpu)
        unpack_y = train_pd.loc[:, "LinkTypeId"].values
        with mp.Pool(num_cpu) as p:
            clf = p.starmap_async(SVM_cross_val_DE_SVM, skfolds.split(train_pd, unpack_y)).get()
        for x in clf:
            clfs.append(x)

    stop_tune_svm = timeit.default_timer()
    if onKmeans is False:
        print_results(clfs, stop_tune_svm, start_tune_svm, filename)
    return clfs




@study
def run_SVM_baseline(train_pd, test_pd, filename):
    """
    Run SVM+word embedding experiment !
    This is the baseline method!
    :param: train_pd Panda DataFrame training_set
    :param: test_pd Panda DataFrame testing_set
    :return:None
    """
    train_x = train_pd.loc[:, "Output"].tolist()
    train_y = train_pd.loc[:, "LinkTypeId"].tolist()
    test_x = test_pd.loc[:, "Output"].tolist()
    test_y = test_pd.loc[:, "LinkTypeId"].tolist()

    clf = svm.SVC(kernel="rbf", gamma=0.005)

    other_scores = {}

    start_svm = timeit.default_timer()
    clf.fit(train_x, train_y)
    stop_svm = timeit.default_timer()
    print("Model Training Time: ", stop_svm - start_svm)
    start_test = timeit.default_timer()
    predicted = clf.predict(test_x)
    stop_test = timeit.default_timer()
    print("Model testing time: ", stop_test - start_test)

    result = metrics.classification_report(test_y, predicted, labels=["1", "2", "3", "4"], digits=3)

    other_scores['IFA'] = get_IFA_score(test_y, predicted)
    other_scores['Accuracy'] = get_acc(metrics.confusion_matrix(test_y, predicted, labels=["1", "2", "3", "4"]))
    other_scores_pd = pd.DataFrame.from_dict(other_scores, orient='index', columns=['1', '2', '3', '4'])
    other_scores_pd = other_scores_pd.T

    result += "\n"
    result += other_scores_pd.to_string()

    result += "\n"
    result += "Time: " + str(stop_test - start_svm) + "\n"

    print_results_simple(result, filename)



# +++++++++++++++++ Katie's Code +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def prepare_word2vec():
    print("Downloading pretrained word2vec models")
    url = "https://zenodo.org/record/807727/files/word2vecs_models.zip"
    file_name = wget.download(url)
    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall()



def run_java_dataset():
    print("RUNNING JAVA DATSET")

    word_src = "word2vecs_models"
    warnings.filterwarnings("ignore")
    if not os.path.exists(word_src):
        prepare_word2vec()
    elif len(os.listdir(word_src)) == 0:
        os.rmdir(word_src)
        prepare_word2vec()
    for x in range(NUM_OF_ITERATION):
        experiment.iteration_num = x + 1

        random.seed(x)
        np.random.seed(x)
        myword2vecs = [os.path.join(word_src, i) for i in os.listdir(word_src)
                       if "syn" not in i]
        random_file = random.randrange(0, len(myword2vecs))

        # Preprocess data into test and train set
        start = timeit.default_timer()
        train_pd, test_pd = preprocess_data(myword2vecs[x])
        stop = timeit.default_timer()
        print("\nData Preprocessing time: ", stop - start)

        run_SVM_baseline(train_pd, test_pd, "ResultsJava" + os.path.sep + "_output_svm.txt")
        print("\nRun completed for baseline model: SVM--------------------------------------------------")
        run_tuning_SVM(train_pd, test_pd, "ResultsJava" + os.path.sep + "_output_de_svm.txt", False)
        print("\nRun completed for baseline model: DE SVM--------------------------------------------------")
        run_FLASH_SVM(train_pd, test_pd, "ResultsJava" + os.path.sep + "_output_flash_svm.txt", False)
        print("\nRun completed for FLASH model: FLASH SVM--------------------------------------------------")

        run_kmeans(train_pd, test_pd, "ResultsJava" + os.path.sep + "_output_kmeans_de_svm.txt", isFlash=False)
        print("Run completed for Kmeans DE model--------------------------------------------------")
        run_kmeans(train_pd, test_pd, "ResultsJava" + os.path.sep + "_output_kmeans_flash_svm.txt", isFlash=True)
        print("Run completed for Kmean Flash model--------------------------------------------------")


def run_python_dataset():
    print("RUNNING PYTHON DATSET")
    word_src = "PythonWord2vecs"
    if not os.path.exists(word_src) or len(os.listdir(word_src)) == 0:
        process.generate_word2vec('ProcessedDataPython' + os.path.sep, 'PythonWord2vecs' + os.path.sep, 10)
    for x in range(NUM_OF_ITERATION):
        experiment.iteration_num = x + 1
        random.seed(x)
        np.random.seed(x)

        myword2vecs = [os.path.join(word_src, i) for i in os.listdir(word_src)
                       if "." not in i]
        random_file = random.randrange(0, len(myword2vecs))

        # Preprocess data into test and train set
        start = timeit.default_timer()
        myword2vec = gensim.models.Word2Vec.load(myword2vecs[x])
        train_pd, test_pd = process.get_processed_train_test_after_word2vec(myword2vec)
        stop = timeit.default_timer()
        print("\nData Preprocessing time: ", stop - start)

        run_SVM_baseline(train_pd, test_pd, "ResultsPython" + os.path.sep + "_output_svm.txt")
        print("\nRun completed for baseline model: SVM--------------------------------------------------")
        run_tuning_SVM(train_pd, test_pd, "ResultsPython" + os.path.sep + "_output_de_svm.txt", False)
        print("\nRun completed for baseline model: DE SVM--------------------------------------------------")
        run_FLASH_SVM(train_pd, test_pd, "ResultsPython" + os.path.sep + "_output_flash_svm.txt", False)
        print("\nRun completed for FLASH model: FLASH SVM--------------------------------------------------")

        run_kmeans(train_pd, test_pd, "ResultsPython" + os.path.sep + "_output_kmeans_de_svm.txt", isFlash=False)
        print("Run completed for Kmeans DE model--------------------------------------------------")
        run_kmeans(train_pd, test_pd, "ResultsPython" + os.path.sep + "_output_kmeans_flash_svm.txt", isFlash=True)
        print("Run completed for Kmean Flash model--------------------------------------------------")


def preprocess_python():
    process.process_dataset('StackoverflowPython' + os.path.sep, 'ProcessedDataPython' + os.path.sep)
    process.generate_word2vec('ProcessedDataPython' + os.path.sep, 'PythonWord2vecs' + os.path.sep, 10)
    myword2vec = gensim.models.Word2Vec.load('PythonWord2vecs' + os.path.sep + 'word2vec1')
    process.get_processed_train_test_after_word2vec(myword2vec)


if __name__ == "__main__":
    # run_java_dataset()
    # run_python_dataset()

    viz_kmean_results("ResultsPython" + os.path.sep + "_output_kmeans_de_svm.txt",
                      "ResultsPython" + os.path.sep + "_output_kmeans_flash_svm.txt", "clustering_stats.txt")

    viz_regular_results("ResultsPython" + os.path.sep + "_output_de_svm.txt",
                      "ResultsPython" + os.path.sep + "_output_flash_svm.txt", "regular_stats.txt")
