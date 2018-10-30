from __future__ import print_function, division
import random
from sklearn import tree
import pandas as pd
import timeit

# __author__ = 'Fahmid'

DEF_BUDGET = 5
DEF_SIZE = 5
DEF_SAMPLE_SPACE_SIZE = 50000

class FLASH(object):
    """
    :parameter
    ===========
    :param learner: data minier to be used to predict
    :param paras_distribution: dictionary type, key is the name, value is a
    list showing range
    :param train_data: training data sets, panda.DataFrame type
    :param tune_data: tuning data sets, panda.DataFrame type
    :param goal: tuning goal, can be "PD, PF, F, PREC, G" ect
    :param uneval_config: get all the possible configuration for the learner
    :param repeats: num of repeats,
    :param size: initial size of the eval_config.
    :param budget: number of times it adjusts
    :param fitness: NOT SURE
    :param configsize: NOT SURE num of config to use from infinite set to make a finite set in the aquisition function
    """



    def __init__(self, params_distribution, goal, target_class,
                 budget=DEF_BUDGET, size=DEF_SIZE, sample_space_size=DEF_SAMPLE_SPACE_SIZE):
        self.budget = budget
        self.size = size
        self.sample_space_size = sample_space_size
        self.params_distribution = params_distribution
        self.goal = goal
        self.target_class = target_class
        self.evaluation = 0
        self.scores = {}
        self.best_config = []
        self.bestscore = []
        self.sample_space = []

    def generate_configs(self):
        candidate = {}
        for key, val in self.params_distribution.items():
            if isinstance(val[0], float):
                candidate[key] = round(random.uniform(val[0], val[1]), 3)
            elif isinstance(val[0], bool):
                candidate[key] = random.random() <= 0.5
            elif isinstance(val[0], str):
                candidate[key] = random.choice(val)
            elif isinstance(val[0], int):
                candidate[key] = int(random.uniform(val[0], val[1]))
            elif isinstance(val[0], list) and isinstance(val[0][0], int):
                candidate[key] = [int(random.uniform(each[0], each[1])) for each in
                                  val]
            else:
                raise ValueError("type of params distribution is wrong!")
        if "random_state" in self.params_distribution.keys():
            candidate["random_state"] = 1  ## set random seed here
        return candidate

    def best(self):
        raise NotImplementedError("Please implement best")


    def evaluate_once(self, **kwargs):
        raise NotImplementedError("Please implement evaluate_once")

    def get_target_score(self, score_dict):
        raise NotImplementedError("Please implement get_target_score")

    def trim(self, n, x):
        if isinstance(self.params_distribution[n][0], float):
            return max(self.params_distribution[n][0],
                       min(round(x, 2), self.params_distribution[n][1]))
        elif isinstance(self.params_distribution[n][0], int):
            return max(self.params_distribution[n][0],
                       min(int(x), self.params_distribution[n][1]))
        else:
            raise ValueError("wrong type here in parameters")

    def Tune(self):

        def isBetter(new, old):
            return new < old if self.goal == "PF" else new > old

        def acquisition_fn(uneval_configs, cart_model):
            predicted = cart_model.predict(uneval_configs)

            val = 0
            id = 0
            for i, x in enumerate(predicted):
                if self.goal == "PF":
                    if val > x:
                        val = x
                        id = i
                else:
                    if val < x:
                        val = x
                        id = i
            return id

        def get_values_only(list_of_dict):
            ret_list = []
            for x in list_of_dict:
                ret_list.append(list(x.values()))
            return ret_list

        def convert_to_floats(list_of_list):
            ret_array = []
            for x in list_of_list:
                temp = []
                for y in x:
                    if (isinstance(y, str)):
                        temp.append(get_val(y))
                    elif isinstance(y, bool):
                        if y:
                            temp.append(1)
                        else:
                            temp.append(0)
                    else:
                        temp.append(y)
                ret_array.append(temp)
            return ret_array

        def get_val(string_class):
            for i, x in enumerate(self.params_distribution['kernel']):
                if(x in string_class):
                    return i

        def get_str(val):
            return self.params_distribution['kernel'][val]


        #TODO convert to pandas, make a new col for the kernel with numbers, take only the numeric cols to calculate

        self.sample_space = [self.generate_configs() for _ in range(self.sample_space_size)]

        sample_space_list = convert_to_floats(get_values_only(self.sample_space))

        eval_configs_x = [self.generate_configs() for _ in range(self.size)]
        eval_configs_x_list = convert_to_floats(get_values_only(eval_configs_x))

        eval_configs_y = []

        start = timeit.default_timer()
        for x in eval_configs_x:
            temp = self.evaluate_once(**x)
            eval_configs_y.append(list(temp.values()))
        stop = timeit.default_timer()
        print ("Initial eval time: ", stop - start)
        self.best_config = eval_configs_x[0]
        temp = []
        for x in eval_configs_y:
            temp.append(x[0])
        eval_configs_y = temp
        best_y = eval_configs_y[0]

        while self.budget > 0:
            start = timeit.default_timer()
            self.evaluation += 1

            cart_model = tree.DecisionTreeRegressor()
            cart_model.fit(eval_configs_x_list, eval_configs_y)

            id = acquisition_fn(sample_space_list, cart_model)
            next_best_config_x = self.sample_space[id]
            next_best_config_x_value = sample_space_list[id]
            next_best_config_y = self.evaluate_once(**next_best_config_x)
            next_best_config_y = list(next_best_config_y.values())[0]
            eval_configs_x.append(next_best_config_x)
            eval_configs_x_list.append(next_best_config_x_value)
            eval_configs_y.append(next_best_config_y)

            if isBetter(next_best_config_y[0], best_y[0]):
                best_y = next_best_config_y
                self.best_config = next_best_config_x
                self.budget += 1
            self.budget -= 1
            stop = timeit.default_timer()
        return self.best_config, self.evaluation


class FLASH_Tune(FLASH):
    def __init__(self, learner, params_distribution, goal, target_class,
                 budget=DEF_BUDGET, size=DEF_SIZE, config_size=DEF_SAMPLE_SPACE_SIZE):
        self.learner = learner
        super(FLASH_Tune, self).__init__(params_distribution, goal, target_class,
                                         budget, size, config_size)

    def evaluate_once(self, **new):
        return self.learner.learn({}, **new)

    def get_target_score(self, score_dict):
        temp = {}
        for key, val in score_dict.items():
            if key == self.target_class:
                temp[key] = val[0]  # value, not list
        return temp

