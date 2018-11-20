import stats
import numpy as np

def viz_regular_results(de_src, flash_src, dest):

    print("\nREGULAR DE vs FLASH:\n")
    de_train_times = []
    de_test_times = []
    flash_train_times, flash_f1s = get_regular_time_f1(flash_src)
    de_train_times, de_f1s = get_regular_time_f1(de_src)

    print("TIME:")
    print_tiles(flash_train_times, "FLASH", de_train_times, "DE   ")
    print_effect_size(flash_train_times, de_train_times)
    print_significant_test(flash_train_times, de_train_times)

    print("\nF1 Scores:")
    print_tiles(flash_f1s, "FLASH", de_f1s, "DE   ")
    print_effect_size(flash_f1s, de_f1s)
    print_significant_test(flash_f1s, de_f1s)

def get_regular_time_f1(src):
    total_time = []
    f1_scores = []
    lines = open(src, "r").readlines()
    for line in lines:
        if "Time:" in line:
            total_time.append(float(line.split(": ")[1]))
        elif "avg" in line:
            processed_line = " ".join(line.split())
            f1_scores.append(float(processed_line.split(" ")[5]))
    return total_time, f1_scores





def viz_kmean_results(de_src, flash_src, dest):

    print("\nCLUSTERED DE vs FLASH:\n")

    de_train_times = []
    de_test_times = []
    flash_train_times, flash_test_times = get_test_train_times(flash_src)
    de_train_times, de_test_times = get_test_train_times(de_src)

    print("TIME:")
    print_tiles(flash_train_times, "FLASH", de_train_times, "DE   ")
    print_effect_size(flash_train_times, de_train_times)
    print_significant_test(flash_train_times, de_train_times)
    #
    # print("\nTEST TIME:")
    # print_tiles(flash_test_times, "FLASH", de_test_times, "DE   ")
    # print_effect_size(flash_test_times, de_test_times)
    # print_significant_test(flash_test_times, de_test_times)


    flash_f1s = get_macro_F1(flash_src.split(".")[0] + ".csv")
    de_f1s = get_macro_F1(de_src.split(".")[0] + ".csv")
    print("\nF1 Scores:")
    print_tiles(flash_f1s, "FLASH", de_f1s, "DE   ")
    print_effect_size(flash_f1s, de_f1s)
    print_significant_test(flash_f1s, de_f1s)

def get_test_train_times(src):
    test = []
    train = []
    lines = open(src, "r").readlines()
    for line in lines:
        if "Train Time:" in line:
            train.append(float(line.split(": ")[1]))
        elif "Test Time:" in line:
            test.append(float(line.split(": ")[1]))
    return (train, test)

def print_tiles(list1, name_of_list1, list2, name_of_list2):
    my_lo = min(list1)
    my_hi = max(list1)
    if min(list2) < my_lo:
        my_lo = min(list2)
    if max(list2) > my_hi:
        my_hi = max(list2)
    def show(lst):
        return stats.xtile(lst, lo=my_lo, hi=my_hi, width=25, show = lambda s: " %3.2f" % s)
    print (name_of_list1, show(list1), "| Median: ", round(stats.median(list1), 2))
    print (name_of_list2, show(list2), "| Median: ", round(stats.median(list2), 2))

def get_macro_F1(src):
    scores = []
    current_score = 0
    first_line =  True
    lines = open(src, "r").readlines()
    for line in lines:
        if first_line:
            first_line = False
            continue
        tokens = line.split(",")

        if tokens[0].isdigit():
            current_score += float(tokens[1])
        else:
            scores.append(round(current_score / 4.0, 3))
            current_score = 0
            continue
    scores.append(round(current_score / 4.0, 3))
    return scores

def print_effect_size(list1, list2):
    print("Effect Size: " + str(stats.cliffsDelta(list1, list2)))

def print_significant_test(list1, list2):
    print('Significant Test: Different' if stats.bootstrap(list1, list2) else 'Significant Test: Same')
