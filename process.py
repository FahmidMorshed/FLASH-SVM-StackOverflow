from __future__ import division
import multiprocessing
import random
import timeit
from functools import partial
from joblib import Parallel, delayed
import gensim
import pandas
import os
from bs4 import BeautifulSoup
import string
import numpy as np
from nltk import WordNetLemmatizer, sent_tokenize, pos_tag, wordpunct_tokenize
from nltk.corpus import stopwords


MIN_HOP = 5
MAX_HOP = 7
WORD2VEC_TRAINING_SAMPLES = 100000
MAX_DEPTH = 100
MAX_SEARCH_BREATH = 100

def tokenize(document):
    lemmatizer = WordNetLemmatizer()
    "Break the document into sentences"
    for sent in sent_tokenize(document):
        "Break the sentence into part of speech tagged tokens"
        for token, tag in pos_tag(wordpunct_tokenize(sent)):

            "Apply preprocessing to the token"
            token = token.lower()  # Convert to lower case
            token = token.strip()  # Strip whitespace and other punctuations
            token = token.strip('_')  # remove _ if any
            token = token.strip('*')  # remove * if any

            "If stopword, ignore."
            if token in stopwords.words('english'):
                continue

            "If punctuation, ignore."
            if all(char in string.punctuation for char in token):
                continue

            "If number, ignore."
            if token.isdigit():
                continue

            # Lemmatize the token and yield
            # Note: Lemmatization is the process of looking up a single word form
            # from the variety of morphologic affixes that can be applied to
            # indicate tense, plurality, gender, etc.
            lemma = lemmatizer.lemmatize(token)
            # all_lema.append(lemma)
            yield lemma


def remove_unnecessary_data_from_qus(source_filename, destination_filename, overwrite=False):
    """
    reads a csv file of stackoverflow questions, remove unnecessary columns, remove html tags then creates a csv with
    <id, text>
    :param source_filename: string of the file name
    :param destination_filename: string of the file name
    :param overwrite: bool should overwrite or skip
    :return: dataset: pandas Dataframe having <id, text>

    """
    start = timeit.default_timer()
    if overwrite is False and os.path.isfile(destination_filename):
        return pandas.read_csv(destination_filename)
    dataset_pd = pandas.read_csv(source_filename, encoding='ANSI')

    dataset_pd['Question'] = dataset_pd['Title'] + " | " + dataset_pd['Body']
    dataset_pd = dataset_pd.drop(columns=['OwnerUserId', 'CreationDate', 'Score', 'Title', 'Body'])
    dataset_pd['Question'] = dataset_pd['Question'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())

    dataset_pd.to_csv(destination_filename, index=False)
    end = timeit.default_timer()
    print("remove_unnecessary_data_from_qus: ", end - start)
    return dataset_pd


def remove_unnecessary_data_from_ans(source_filename, destination_filename, overwrite=False):
    """
    reads a csv file of stackoverflow answers, remove unnecessary columns, remove html tags then creates a csv with
    <id, text>
    :param source_filename: string of the file name
    :param destination_filename: string of the file name
    :param overwrite: bool should overwrite the file or skip
    :return: dataset: pandas Dataframe having <id, text>
    """
    start = timeit.default_timer()
    if overwrite is False and os.path.isfile(destination_filename):
        return pandas.read_csv(destination_filename)
    dataset_pd = pandas.read_csv(source_filename, encoding='ANSI')
    dataset_pd['Answer'] = dataset_pd['Body']
    dataset_pd = dataset_pd.drop(columns=['OwnerUserId', 'CreationDate', 'Score', 'Body'])
    dataset_pd['Answer'] = dataset_pd['Answer'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
    dataset_pd.to_csv(destination_filename, index=False)
    end = timeit.default_timer()
    print("remove_unnecessary_data_from_ans: ", end - start)
    return dataset_pd


def convert_post_links(src, dest, overwrite=False):
    """
    converts the xml to a csv. Also, changes link type 1 to 2 (direct link) and link type 3 to 1 (duplicate link), this
    change is necessary to make it simillar to Xu's work.
    :param src: string with source file name of the xml
    :param dest: string of the destination filename as csv
    :param overwrite: bool to force overwrite
    :return: panda Dataframe having <id, PostId, RelatedPostId, LinkType>
    """
    start = timeit.default_timer()
    if overwrite is False and os.path.isfile(dest):
        return pandas.read_csv(dest)

    with open(src, mode='r', encoding='ANSI') as file:
        line_count = 0
        post_ids = []
        related_post_ids = []
        link_types = []
        for line in file.readlines():
            tokens = line.split(" ")
            if (len(tokens)) != 9:
                continue
            post_ids.append(tokens[5].split("\"")[1])
            related_post_ids.append(tokens[6].split("\"")[1])
            link_types.append(tokens[7].split("\"")[1])

            line_count += 1

        related_post_pd = pandas.DataFrame({
            'PostId': post_ids,
            'RelatedPostId': related_post_ids,
            'LinkType': link_types})

        related_post_pd.loc[related_post_pd['LinkType'] == '1', 'LinkType'] = 2
        related_post_pd.loc[related_post_pd['LinkType'] == '3', 'LinkType'] = 1

        related_post_pd.to_csv(dest, index=False)

        end = timeit.default_timer()
        print("convert_post_links: ", end - start)
        return related_post_pd


def create_dataset(qus_src, link_src, dest, overwrite=False):
    start = timeit.default_timer()
    if overwrite is False and os.path.isfile(dest):
        all_examples = pandas.read_csv(dest)
        print("Duplicates(1): ", len(all_examples.loc[all_examples['LinkType'] == 1]))
        print("Direct(2): ", len(all_examples.loc[all_examples['LinkType'] == 2]))
        print("Indirect(3): ", len(all_examples.loc[all_examples['LinkType'] == 3]))
        print("Isolated(4): ", len(all_examples.loc[all_examples['LinkType'] == 4]))
        return all_examples

    qus_pd = pandas.read_csv(qus_src)
    link_pd = pandas.read_csv(link_src)

    available_link_pd = link_pd.loc[link_pd['PostId'].isin(qus_pd['Id'])]
    available_related_link_pd = available_link_pd.loc[available_link_pd['RelatedPostId'].isin(qus_pd['Id'])]

    all_duplicates = available_related_link_pd.loc[available_related_link_pd['LinkType'] == 1]
    all_directs = available_related_link_pd.loc[available_related_link_pd['LinkType'] == 2]

    # NEW ADDED TO FIX THE DATASET NOV 13
    print("Dup: ", len(all_duplicates))
    print("Before: ", len(all_directs))
    for i, row in all_duplicates.iterrows():
        all_directs = all_directs.drop(all_directs.loc[(all_directs['PostId'] == row['PostId']) & (
                all_directs['RelatedPostId'] == row['RelatedPostId'])].index)

    print("After: ", len(all_directs))
    # END

    all_indirects = pandas.DataFrame(columns=['PostId', 'RelatedPostId', 'LinkType'])
    all_isolated = pandas.DataFrame(columns=['PostId', 'RelatedPostId', 'LinkType'])

    count = 0
    indirect_count = 0
    isolated_count = 0

    print("Before converting 3 & 4")
    sample = all_directs.sample(frac=.7, replace=False, random_state=1)
    sample2 = all_directs.sample(frac=.7, replace=False, random_state=2)
    for i in range(len(sample)):
        if (count+1) % 10000 == 0:
            print("10k done")
        row1 = sample.iloc[count, :]
        row2 = sample2.iloc[count, :]
        a = row1['PostId']
        b = row2['PostId']
        #indirect = get_indirect(a, all_directs)
        isolated, related = get_isolated(a, b, all_directs, 0)
        if related is not None:
            all_indirects.loc[indirect_count] = [a, related, 3]
            indirect_count += 1
        if isolated:
            all_isolated.loc[isolated_count] = [a, b, 4]
            isolated_count += 1
        count += 1

    all_examples = pandas.concat(([all_duplicates, all_directs, all_indirects, all_isolated]))
    all_examples = all_examples.drop_duplicates(subset=['PostId', 'RelatedPostId'], keep='first')
    all_examples.to_csv(dest, index=False)

    print("Duplicates(1): ", len(all_examples.loc[all_examples['LinkType'] == 1]))
    print("Direct(2): ", len(all_examples.loc[all_examples['LinkType'] == 2]))
    print("Indirect(3): ", len(all_examples.loc[all_examples['LinkType'] == 3]))
    print("Isolated(4): ", len(all_examples.loc[all_examples['LinkType'] == 4]))

    end = timeit.default_timer()
    print("create_dataset: ", end - start)


# NEW FIX NOV 13
def get_isolated(a, b, direct_list, depth, related=None):
    if a == b:
        return False, related
    if depth > MAX_DEPTH:
        return False, related
    current = direct_list['RelatedPostId'].loc[direct_list['PostId'] == a]
    if current.empty:
        return True, related
    if len(current.values - depth) > MAX_SEARCH_BREATH:
        return False, related
    for val in current.values:
        if depth > MIN_HOP and (random.uniform(0, 1) < (1.0/((depth-MIN_HOP)))) and depth < (MAX_DEPTH - MIN_HOP - 1):
            related = val
        decision, related = get_isolated(val, b, direct_list, depth+1, related)
        if decision is False:
            return False, related
    return True, related
# END


def get_indirect(a, direct_list):

    current = direct_list['RelatedPostId'].loc[direct_list['PostId'] == a]
    current = current.values[0]

    related = direct_list['RelatedPostId'].loc[direct_list['PostId'] == current]
    if related.empty:
        return -1
    related = random.choice(related.values)

    for hop in range(random.randint(MIN_HOP, MAX_HOP)):
        # NEW FIX FOR DATASET a->c, c->a fix NOV 13
        if a == related:
            return -1
        # END

        current = related
        related = direct_list['RelatedPostId'].loc[direct_list['PostId'] == current]
        if related.empty:
            if hop < MIN_HOP:
                return -1
            else:
                return current
        related = random.choice(related.values)
    return related


def create_test_train(src, dest_dir, overwrite=True):
    start = timeit.default_timer()
    if overwrite is False and os.path.isfile(dest_dir + 'train.csv') and os.path.isfile(dest_dir + 'test.csv'):
        train = pandas.read_csv(dest_dir + 'train.csv')
        test = pandas.read_csv(dest_dir + 'test.csv')
        return train, test

    all_examples = pandas.read_csv(src)

    train1, test1 = get_test_train_for_link(680, all_examples, 1)
    train2, test2 = get_test_train_for_link(1500, all_examples, 2, 400)
    train3, test3 = get_test_train_for_link(1500, all_examples, 3, 400)
    train4, test4 = get_test_train_for_link(1500, all_examples, 4, 400)

    train = pandas.concat([train1, train2, train3, train4])
    test = pandas.concat([test1, test2, test3, test4])

    train.to_csv(dest_dir + 'train.csv', index=False)
    test.to_csv(dest_dir + 'test.csv', index=False)

    end = timeit.default_timer()
    print("create_test_train: ", end - start)
    return train, test


def get_test_train_for_link(train_num, all_examples, link_type, test_num=None):
    ex_list = all_examples.loc[all_examples['LinkType'] == link_type]
    train = ex_list.sample(train_num, replace="False", random_state=1)
    test = pandas.concat([ex_list, train])
    test = test.drop_duplicates(keep=False)
    if test_num is not None:
        test = test.sample(test_num, replace="False", random_state=1)
    return train, test


def process_dataset(unprocessed_dir, processed_dir):
    if os.path.isdir(unprocessed_dir) is False:
        print("ERROR: Directory mising for unprocessed files")
        return
    if os.path.isdir(processed_dir) is False:
        os.mkdir(processed_dir)

    ans = remove_unnecessary_data_from_ans(unprocessed_dir + 'Answers.csv', processed_dir + 'Answers.csv')

    qus = remove_unnecessary_data_from_qus(unprocessed_dir + 'Questions.csv', processed_dir + 'Questions.csv')
    convert_post_links(unprocessed_dir + 'PostLinks.xml',unprocessed_dir + 'AllPostLinks.csv')

    create_dataset(processed_dir + 'Questions.csv',
                   unprocessed_dir + 'AllPostLinks.csv',
                   unprocessed_dir + 'all_examples.csv')
    train, test = create_test_train(unprocessed_dir + 'all_examples.csv', processed_dir)

    return train, test, qus


def build_model(documents):
    model = gensim.models.Word2Vec(
        documents,
        size=200,
        window=10,
        min_count=2,
        workers=5)
    model.train(documents, total_examples=len(documents), epochs=10)
    return model


def generate_word2vec(unprocessed_dir, processed_dir, num_of_models):
    if os.path.isdir(processed_dir) is False:
        os.mkdir(processed_dir)
    qus = pandas.read_csv(unprocessed_dir + 'Questions.csv')
    ans = pandas.read_csv(unprocessed_dir + 'Answers.csv')
    qus['Text'] = qus['Question']
    ans['Text'] = ans['Answer']
    qus = qus.drop(columns=['Question'])
    ans = ans.drop(columns=['Answer', 'ParentId'])
    all_post = pandas.concat([qus, ans])

    # num_of_cpu = multiprocessing.cpu_count()
    # print(num_of_cpu)
    #
    # make_word2vec_wrapper = partial(make_word2vec, all_pst=all_post, prcc_dir=processed_dir)
    # start = time.time()
    # with multiprocessing.Pool(num_of_cpu) as p:
    #     p.starmap_async(make_word2vec_wrapper, get_loop_num()).get()
    # stop = time.time()
    # print("For multiprocessing: ", stop - start)

    start = timeit.default_timer()
    Parallel(n_jobs=3)(delayed(make_word2vec)(all_post, processed_dir, i) for i in range(num_of_models))
        # start_loop = timeit.default_timer()
        # make_word2vec()
        # end_loop = timeit.default_timer()
        # print("1 model done on: ", end_loop - start_loop)
    stop = timeit.default_timer()
    print("Regular: ", stop - start)


def make_word2vec(all_pst, prcc_dir, i):
    start = timeit.default_timer()
    sample_for_word2vec = all_pst.sample(WORD2VEC_TRAINING_SAMPLES, replace=False, random_state=i)
    print ("Sampling time: ", timeit.default_timer() - start)

    training_tokens = []
    start = timeit.default_timer()
    for x in sample_for_word2vec['Text']:
        training_tokens.append([y for y in gensim.utils.simple_preprocess(x)])#tokenize(x)]
    print("Tokenizing time: ", timeit.default_timer() - start)

    start = timeit.default_timer()
    my_model = build_model(training_tokens)
    my_model.save(prcc_dir + 'word2vec' + str(i))
    print("Building and saving: ", timeit.default_timer() - start)


def get_processed_train_test_after_word2vec(word2vec):
    train_pd, test_pd, qus_pd = process_dataset('StackoverflowPython' + os.path.sep, 'ProcessedDataPython' + os.path.sep)
    train_pd = make_output_vec(train_pd, word2vec, qus_pd)
    test_pd = make_output_vec(test_pd, word2vec, qus_pd)

    train_pd = train_pd.rename(columns={"LinkType": "LinkTypeId"})
    test_pd = test_pd.rename(columns={"LinkType": "LinkTypeId"})

    return train_pd, test_pd

def make_output_vec(data, word2vec, qus_pd):
    start = timeit.default_timer()
    data["PostIdVec"] = ""
    data["RelatedPostIdVec"] = ""
    itr = 0
    for index, row in data.iterrows():
        if(itr % 1000 == 0):
            print("1k done")
        itr += 1

        id = row['PostId']
        related_id = row['RelatedPostId']
        post_qus = qus_pd['Question'].loc[qus_pd['Id'] == id]
        post_qus = post_qus.values[0]
        related_post_qus = qus_pd['Question'].loc[qus_pd['Id'] == related_id]
        related_post_qus = related_post_qus.values[0]

        post_qus = [y for y in gensim.utils.simple_preprocess(post_qus)]
        related_post_qus = [y for y in gensim.utils.simple_preprocess(related_post_qus)]

        x = np.array(
            [word2vec[i] for i in post_qus if
             i in word2vec.wv.vocab])
        word_count_p = len(x)
        word_vecs_p = np.sum(x, axis=0)

        y = np.array(
            [word2vec[i] for i in related_post_qus if
             i in word2vec.wv.vocab])
        word_count_rp = len(y)
        word_vecs_rp = np.sum(y, axis=0)

        temp = word_vecs_p / word_count_p
        data.set_value(index, "PostIdVec", temp)
        data.set_value(index, "RelatedPostIdVec", word_vecs_rp / word_count_rp)

    data['Output'] = (data['PostIdVec'] + data['RelatedPostIdVec']) / 2

    end = timeit.default_timer()
    print("make_output_vec: ", end - start)
    return data

def refine_qus(src_dir, dest_dir):
    start = timeit.default_timer()
    dataset_pd = pandas.read_csv(src_dir + "Questions.csv", encoding='ANSI')
    ans_pd = pandas.read_csv(src_dir + "Answers.csv", encoding='ANSI')
    for i, row in dataset_pd.iterrows():

        id = row['Id']
        qus = row['Question']
        bodies = ans_pd['Answer'].loc[ans_pd['ParentId'] == id]
        for body in bodies:
            qus += " | " + body
        qus = qus.strip()

        dataset_pd.set_value(i, "Question", qus)
    dataset_pd.to_csv(dest_dir + "Posts.csv")
    print("refine_qus: ", timeit.default_timer() - start)

if __name__=='__main__':
    #refine_qus('ProcessedDataPython' + os.path.sep, 'ProcessedDataPython' + os.path.sep)

    process_dataset('StackoverflowPython' + os.path.sep, 'ProcessedDataPython' + os.path.sep)
    #generate_word2vec('ProcessedDataPython' + os.path.sep, 'PythonWord2vecs' + os.path.sep, 10)
    #myword2vec = gensim.models.Word2Vec.load('PythonWord2vecs' + os.path.sep + 'word2vec1')
    #get_processed_train_test_after_word2vec(myword2vec)


