from __future__ import print_function, division

from copy import deepcopy

import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from newabcd import ABCD
from collections import Counter
__author__ = 'Suvodeep'


class Learners(object):
    def __init__(self, clf, train_x, train_y, predict_x, predict_y, goal):
        """

        :param clf: classifier, SVM, etc...
        :param train_x: training data, independent variables.
        :param train_y: training labels, dependent variables.
        :param predict_x: testing data, independent variables.
        :param predict_y: testingd labels, dependent variables.
        :param goal: the objective of your tuning, F, recision,....
        """
        self.train_X = train_x.tolist()
        self.train_Y = train_y.tolist()
        self.predict_X = predict_x
        self.predict_Y = predict_y
        self.goal = goal
        self.param_distribution = self.get_param()
        self.learner = clf
        self.confusion = None
        self.params = None
        self.scores = None
        self.confustin_matix = None

    def learn(self, F, **kwargs):
        """
        :param F: a dict, holds all scores, can be used during debugging
        :param kwargs: a dict, all the parameters need to set after tuning.
        :return: F, scores.
        """
        self.learner.set_params(**kwargs)
        clf = self.learner.fit(self.train_X, self.train_Y)
        predict_result = []
        predict_Y = []
        for predict_X, actual in zip(self.predict_X, self.predict_Y):
            try:
                _predict_result = clf.predict(predict_X.reshape(1, -1))
                predict_result.append(_predict_result[0])
                predict_Y.append(actual)
            except:
                print("one pass")
                continue
        self.scores = self._Abcd(predict_result, predict_Y, F)
        self.confusion = classification_report(predict_Y, predict_result, digits=3)
        self.params = kwargs
        return self.scores

    def _Abcd(self, predicted, actual, F):
        """

        :param predicted: predicted results(labels)
        :param actual: actual results(labels)
        :param F: previously got scores
        :return: updated scores.
        """

        def calculate(scores):
            for i, v in enumerate(scores):
                F[uni_actual[i]] = F.get(uni_actual[i], []) + [v]
            freq_actual = [count_actual[one] / len(actual) for one in uni_actual]
            F["mean"] = F.get("mean", []) + [np.mean(scores)]
            F["mean_weighted"] = (F.get("mean_weighted", []) + [
                np.sum(np.array(scores) * np.array(freq_actual))])
            return F

        def micro_cal(goal="Micro_F"):
            TP, FN, FP = 0, 0, 0
            for each in confusion_matrix_all_class:
                TP += each.TP
                FN += each.FN
                FP += each.FP
            PREC = TP / (TP + FP)
            PD = TP / (TP + FN)
            F[goal] = F.get(goal, []) + [round(2.0 * PREC * PD / (PD + PREC), 3)]
            return F

        def macro_cal(goal="Macro_F"):
            PREC_sum, PD_sum = 0, 0
            for each in confusion_matrix_all_class:
                PD_sum += each.stats()[0]
                PREC_sum += each.stats()[2]
            PD_avg = PD_sum / len(uni_actual)
            PREC_avg = PREC_sum / len(uni_actual)
            F[goal] = F.get(goal, []) + [
                round(2.0 * PREC_avg * PD_avg / (PREC_avg + PD_avg), 3)]
            return F

        _goal = {0: "PD", 1: "PF", 2: "PREC", 3: "ACC",
                 4: "F", 5: "G", 6: "Macro_F", 7: "Micro_F"}
        abcd = ABCD(actual, predicted)
        # print(abcd)
        uni_actual = list(set(actual))
        # print(uni_actual)
        count_actual = Counter(actual)
        if "Micro" in self.goal or "Macro" in self.goal:
            confusion_matrix_all_class = [each for each in abcd()]
            self.confustin_matix = confusion_matrix_all_class
            gate = confusion_matrix_all_class
            # pdb.set_trace() # to trace

            ordered_confusion_matrix = deepcopy(gate)
            # for x in ordered_confusion_matrix:
            #     if x.indx == "4":
            #         gate[0] = x
            #     elif x.indx == "3":
            #         gate[1] = x
            #     elif x.indx == "2":
            #         gate[2] = x
            #     elif x.indx == "1":
            #         gate[3] = x
            #
            #
            # if len(gate) == 4 and (gate[0].indx != "4" and gate[1].indx != "3" and
            #                        gate[2].indx != "1" and gate[3].indx != "2"):  # changed 0 to 3
            #     raise ValueError("confusion matrix has the wrong order of class")
            if "Micro" in self.goal:
                return micro_cal()
            else:
                return macro_cal()
        else:
            score_each_klass = [k.stats()[_goal[self.goal]] for k in
                                abcd()]
            return calculate(score_each_klass)


class SK_SVM(Learners):
    def __init__(self, train_x, train_y, predict_x, predict_y, goal):
        clf = SVC()
        super(SK_SVM, self).__init__(clf, train_x, train_y, predict_x, predict_y,
                                     goal)

    def get_param(self):
        tunelst = {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                   "C": [1, 50],
                   "coef0": [0.0, 1],
                   "gamma": [0.0, 1],
                   "random_state": [1, 1]}
        return tunelst

if __name__ == "__main__":
    pass
