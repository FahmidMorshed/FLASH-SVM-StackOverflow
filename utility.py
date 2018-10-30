from __future__ import print_function, division
import datetime
import os
import pickle
import sys
import re
import time
import pandas as pd
import pdb
import random
import collections
import gensim
import multiprocessing as mp
from model import PaperData
from functools import partial


def study(f):
    def wrapper(*lst):
        # rseed() # reset the seed to our default
        what = f.__name__  # print the function name
        doc = f.__doc__  # print the function doc
        if doc:
            doc = re.sub(r"\n[ \t]*", "\n# ", doc)
        # print when this ran
        show = datetime.datetime.now().strftime
        print("\n###", what, "#" * 50)
        print("#", show("%Y-%m-%d %H:%M:%S"))
        if doc: print("#", doc)
        t1 = time.time()
        res = f(*lst)  # run the function
        t2 = time.time()  # show how long it took to run
        print("\n" + ("-" * 72))
        # showd(The)       # print the options
        print("# Runtime: %.3f secs" % (t2 - t1))
        return res
    return wrapper


### Coercion  #####################################
def atom(x):
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


### Command line processing ########################
def cmd(com="life(seed=1)"):
    "Convert command line to a function call."
    if len(sys.argv) < 2:
        return

    def strp(x):
        return isinstance(x, basestring)

    def wrap(x):
        return "'%s'" % x if strp(x) else str(x)

    words = map(wrap, map(atom, sys.argv[2:]))
    return sys.argv[1] + '(' + ','.join(words) + ')'

### Supporting experiment ###########################
def load_vec(d, data, use_pkl=False, file_name=None):
    if use_pkl:
        if os.path.isfile(file_name):
            with open(file_name, "rb") as my_pickle:
                return pickle.load(my_pickle)
    else:
        # print("call get_document_vec")
        return d.get_document_vec(data, file_name)



def preprocess_data(word2vec_src):
    """
    Preprocess word2vec and gets the training and testing set of data
    :param word2vec_src: the source file of word2vec
    :return: train_X, train_Y, test_X, test_Y
    """
    print("# word2vec:", word2vec_src)
    word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
    data = PaperData(word2vec=word2vec_model)

    train_pd = load_vec(data, data.train_data, use_pkl=False)
    test_pd = load_vec(data, data.test_data, use_pkl=False)
    return train_pd, test_pd



def classifaction_report_csv(report, filename):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('     ')
        row['class'] = row_data[2]
        row['precision'] = float(row_data[3].strip())
        row['recall'] = float(row_data[4])
        row['f1_score'] = float(row_data[5])
        row['support'] = float(row_data[6].strip())
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(filename, mode='a', index=False)

"""
Parse a sklearn classification report into a dict keyed by class name
and containing a tuple (precision, recall, fscore, support) for each class
Reference: https://gist.github.com/julienr/6b9b9a03bd8224db7b4f
"""
def parse_classification_report(clfreport):
    lines = clfreport.split('\n')
    # Remove empty lines
    lines = list(filter(lambda l: not len(l.strip()) == 0, lines))

    # Starts with a header, then score for each class and finally an average
    header = lines[0]
    cls_lines = lines[1:-1]
    avg_line = lines[-1]

    assert header.split() == ['precision', 'recall', 'f1-score', 'support']
    assert avg_line.split()[0] == 'avg'

    # class names can have spaces - figure the width of the class field
    # using indentation of the precision header
    cls_field_width = len(header) - len(header.lstrip())

    # Now, collect all the class names and score in a dict
    def parse_line(l):
        """Parse a line of classification_report"""
        cls_name = l[:cls_field_width].strip()
        precision, recall, fscore, support = l[cls_field_width:].split()
        precision = float(precision)
        recall = float(recall)
        fscore = float(fscore)
        support = int(support)
        return (cls_name, precision, recall, fscore, support)

    data = collections.OrderedDict()
    for l in cls_lines:
        ret = parse_line(l)
        cls_name = ret[0]
        scores = ret[1:]
        data[cls_name] = scores

    data['avg'] = parse_line(avg_line)[1:]  # average
    return data