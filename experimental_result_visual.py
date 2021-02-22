import json

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    file = "./experimental_results/tf_start_5_no_weighting.txt"
    with open(file, "r") as read_file:
        res = json.load(read_file)

    file = "./experimental_results/tf_start_5_w_weighting.txt"
    with open(file, "r") as read_file:
        res_2 = json.load(read_file)

    index_list = np.array([5, 15, 25, 35, 45])

    bm = []
    tl = []
    tl_2 = []
    for index in index_list:
        res_i = res[str(index)]
        benchmark = res_i["benchmark"]
        bm.append(benchmark)
        print("benchmark", benchmark)
        tl_list = res_i["tl"]
        tl_mean = np.mean(tl_list)
        print("tl_mean", tl_mean)
        tl.append(tl_mean)

        res_2_i = res_2[str(index)]
        tl_list_2 = res_2_i["tl"]
        tl_mean_2 = np.mean(tl_list_2)
        print("tl_mean_2", tl_mean_2)
        tl_2.append(tl_mean_2)

    N = len(index_list)
    ind = np.arange(N)
    width = 0.27
    ax = plt.subplot(111)
    rects1 = ax.bar(ind, bm, width=width, color='b', align='center')
    rects2 = ax.bar(ind + width, tl, width=width, color='g', align='center')
    rects3 = ax.bar(ind + 2*width, tl_2, width=width, color='r', align='center')
    ax.set_ylim(0.2, 1)

    ax.set_ylabel('acc')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(index_list)
    ax.legend((rects1[0], rects2[0], rects3[0]), ('Tr', 'Tr_wF', 'Tr_wFW'))

    plt.show()
