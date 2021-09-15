import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_all_guest():

    g_250 = 0.6236
    g_500 = 0.634
    g_1000 = 0.6438
    g_2000 = 0.6526
    g_4000 = 0.6576
    g_6000 = 0.6616
    g_8000 = 0.6724
    g_10000 = 0.6774

    results = dict()
    results["250"] = dict()
    results["500"] = dict()
    results["1000"] = dict()
    results["2000"] = dict()
    results["4000"] = dict()
    results["6000"] = dict()
    results["8000"] = dict()
    results["10000"] = dict()
    results["12000"] = dict()
    results["20000"] = dict()

    results["250"]["g_acc"] = g_250
    results["500"]["g_acc"] = g_500
    results["1000"]["g_acc"] = g_1000
    results["2000"]["g_acc"] = g_2000
    results["4000"]["g_acc"] = g_4000
    results["6000"]["g_acc"] = g_6000
    results["8000"]["g_acc"] = g_8000
    results["10000"]["g_acc"] = g_10000
    return results


def get_all_guest_v2():

    g_250 = 0.6774
    g_500 = 0.6774
    g_1000 = 0.6774
    g_2000 = 0.6774
    g_4000 = 0.6774
    g_6000 = 0.6774
    g_8000 = 0.6774
    g_10000 = 0.6774

    results = dict()
    results["250"] = dict()
    results["500"] = dict()
    results["1000"] = dict()
    results["2000"] = dict()
    results["4000"] = dict()
    results["6000"] = dict()
    results["8000"] = dict()
    results["10000"] = dict()
    results["12000"] = dict()
    results["20000"] = dict()

    results["250"]["g_acc"] = g_250
    results["500"]["g_acc"] = g_500
    results["1000"]["g_acc"] = g_1000
    results["2000"]["g_acc"] = g_2000
    results["4000"]["g_acc"] = g_4000
    results["6000"]["g_acc"] = g_6000
    results["8000"]["g_acc"] = g_8000
    results["10000"]["g_acc"] = g_10000
    return results


def get_fed_mvt_result():
    # Num_overlap = 250

    all_acc_250_1 = 0.6184
    g_acc_250_1 = 0.60968
    h_acc_250_1 = 0.3376

    all_acc_250 = all_acc_250_1
    g_acc_250 = g_acc_250_1
    h_acc_250 = h_acc_250_1

    # Num_overlap = 500
    # 'all_acc': 0.6268, 'g_acc': 0.61328, 'h_acc': 0.38764,

    all_acc_500_1 = 0.6268
    g_acc_500_1 = 0.61328
    h_acc_500_1 = 0.38764

    all_acc_500 = all_acc_500_1
    g_acc_500 = g_acc_500_1
    h_acc_500 = h_acc_500_1

    # Num_overlap = 1000
    # 'all_acc': 0.6504, 'g_acc': 0.64012, 'h_acc': 0.41748,

    all_acc_1000_1 = 0.6504
    g_acc_1000_1 = 0.64012
    h_acc_1000_1 = 0.41748

    all_acc_1000_2 = 0.6504
    g_acc_1000_2 = 0.64012
    h_acc_1000_2 = 0.41748

    all_acc_1000 = (all_acc_1000_1 + all_acc_1000_2) / 2
    g_acc_1000 = (g_acc_1000_1 + g_acc_1000_2) / 2
    h_acc_1000 = (h_acc_1000_1 + h_acc_1000_2) / 2

    # Num_overlap = 2000
    # 'all_acc': 0.6602, 'g_acc': 0.6491199999999999, 'h_acc': 0.48360000000000003,

    all_acc_2000_1 = 0.6602
    g_acc_2000_1 = 0.6491199999999999
    h_acc_2000_1 = 0.48360000000000003

    all_acc_2000_2 = 0.6602
    g_acc_2000_2 = 0.6491199999999999
    h_acc_2000_2 = 0.48360000000000003

    all_acc_2000 = (all_acc_2000_1 + all_acc_2000_2) / 2
    g_acc_2000 = (g_acc_2000_1 + g_acc_2000_2) / 2
    h_acc_2000 = (h_acc_2000_1 + h_acc_2000_2) / 2

    # Num_overlap = 4000
    # 'all_acc': 0.6784, 'g_acc': 0.6687200000000001, 'h_acc': 0.50316,

    all_acc_4000_1 = 0.6784
    g_acc_4000_1 = 0.6687200000000001
    h_acc_4000_1 = 0.50316

    all_acc_4000_2 = 0.6784
    g_acc_4000_2 = 0.6687200000000001
    h_acc_4000_2 = 0.50316

    all_acc_4000_3 = 0.6784
    g_acc_4000_3 = 0.6687200000000001
    h_acc_4000_3 = 0.50316

    all_acc_4000 = (all_acc_4000_1 + all_acc_4000_2 + all_acc_4000_3) / 3
    g_acc_4000 = (g_acc_4000_1 + g_acc_4000_2 + g_acc_4000_3) / 3
    h_acc_4000 = (h_acc_4000_1 + h_acc_4000_2 + h_acc_4000_3) / 3

    # Num_overlap = 6000
    # 'all_acc': 0.705, 'g_acc': 0.6797599999999999, 'h_acc': 0.5485599999999999,

    all_acc_6000_1 = 0.705
    g_acc_6000_1 = 0.6797599999999999
    h_acc_6000_1 = 0.5485599999999999

    all_acc_6000 = all_acc_6000_1
    g_acc_6000 = g_acc_6000_1
    h_acc_6000 = h_acc_6000_1

    # Num_overlap = 8000
    # 'all_acc': 0.7144, 'g_acc': 0.69068, 'h_acc': 0.5768000000000001,

    all_acc_8000_1 = 0.7144
    g_acc_8000_1 = 0.69068
    h_acc_8000_1 = 0.5768000000000001

    all_acc_8000_2 = 0.7144
    g_acc_8000_2 = 0.69068
    h_acc_8000_2 = 0.5768000000000001

    all_acc_8000 = (all_acc_8000_1 + all_acc_8000_2) / 2
    g_acc_8000 = (g_acc_8000_1 + g_acc_8000_2) / 2
    h_acc_8000 = (h_acc_8000_1 + h_acc_8000_2) / 2

    # Num_overlap = 10000
    # 'all_acc': 0.7206, 'g_acc': 0.69872, 'h_acc': 0.5854,

    all_acc_8000_1 = 0.7206
    g_acc_8000_1 = 0.69872
    h_acc_8000_1 = 0.5854

    all_acc_10000 = all_acc_8000_1
    g_acc_10000 = g_acc_8000_1
    h_acc_10000 = h_acc_8000_1

    results = dict()
    results["250"] = dict()
    results["500"] = dict()
    results["1000"] = dict()
    results["2000"] = dict()
    results["4000"] = dict()
    results["6000"] = dict()
    results["8000"] = dict()
    results["10000"] = dict()
    results["12000"] = dict()
    results["20000"] = dict()

    results["250"]["all_acc"] = all_acc_250
    results["250"]["g_acc"] = g_acc_250
    results["250"]["h_acc"] = h_acc_250

    results["500"]["all_acc"] = all_acc_500
    results["500"]["g_acc"] = g_acc_500
    results["500"]["h_acc"] = h_acc_500

    results["1000"]["all_acc"] = all_acc_1000
    results["1000"]["g_acc"] = g_acc_1000
    results["1000"]["h_acc"] = h_acc_1000

    results["2000"]["all_acc"] = all_acc_2000
    results["2000"]["g_acc"] = g_acc_2000
    results["2000"]["h_acc"] = h_acc_2000

    results["4000"]["all_acc"] = all_acc_4000
    results["4000"]["g_acc"] = g_acc_4000
    results["4000"]["h_acc"] = h_acc_4000

    results["6000"]["all_acc"] = all_acc_6000
    results["6000"]["g_acc"] = g_acc_6000
    results["6000"]["h_acc"] = h_acc_6000

    results["8000"]["all_acc"] = all_acc_8000
    results["8000"]["g_acc"] = g_acc_8000
    results["8000"]["h_acc"] = h_acc_8000

    results["10000"]["all_acc"] = all_acc_10000
    results["10000"]["g_acc"] = g_acc_10000
    results["10000"]["h_acc"] = h_acc_10000

    return results


def get_benchmark_result():

    # 250
    acc_mean_250 = 0.36112
    acc_stddev_250 = 0.006475306942531753

    # 500
    acc_mean_500 = 0.40028
    acc_stddev_500=0.004050876448375079

    # 1000
    acc_mean_1000 = 0.442
    acc_stddev_1000 = 0.005268775948927792

    # 2000
    acc_mean_2000 = 0.50384
    acc_stddev_2000=0.006680598775558947

    # 4000
    acc_mean_4000 = 0.5646
    acc_stddev_4000=0.008660715905743582

    # 6000
    acc_mean_6000 = 0.6074
    acc_stddev_6000 = 0.008660715905743582

    # 8000
    acc_mean_8000 = 0.6287
    acc_stddev_8000 = 0.008660715905743582

    # 10000
    acc_mean_10000 = 0.6501
    acc_stddev_10000 = 0.008660715905743582

    results = dict()
    results["250"] = dict()
    results["500"] = dict()
    results["1000"] = dict()
    results["2000"] = dict()
    results["4000"] = dict()
    results["6000"] = dict()
    results["8000"] = dict()
    results["10000"] = dict()
    results["12000"] = dict()
    results["20000"] = dict()

    results["250"]["all_acc"] = acc_mean_250
    results["500"]["all_acc"] = acc_mean_500
    results["1000"]["all_acc"] = acc_mean_1000
    results["2000"]["all_acc"] = acc_mean_2000
    results["4000"]["all_acc"] = acc_mean_4000
    results["6000"]["all_acc"] = acc_mean_6000
    results["8000"]["all_acc"] = acc_mean_8000
    results["10000"]["all_acc"] = acc_mean_10000

    # results["6000"]["all_acc"] = acc_mean_6000
    # results["8000"]["all_acc"] = acc_mean_8000
    # results["100000"]["all_acc"] = acc_mean_10000

    return results


def plot_series(metric_records, lengend_list, data_type=""):
    plt.rcParams['pdf.fonttype'] = 42

    # style_list = ["r", "b", "g", "k", "m", "y", "c"]
    # style_list = ["r", "g", "g--", "k", "k--", "y", "y--"]
    # style_list = ["r", "b", "g", "k", "r--", "b--", "g--", "k--"]
    # style_list = ["r", "b", "g", "r--", "b--", "g--", "r-.", "b-.", "g-."]
    # style_list = ["r", "b", "g", "r--", "b--", "g--", "r-.", "b-.", "g-."]

    # style_list = ["r", "b", "g", "k", "m", "y", "c"]
    style_list = ["orchid", "red", "green", "blue", "purple", "peru", "olive", "coral"]

    if len(lengend_list) == 3:
        style_list = ["r", "b", "b--"]

    if len(lengend_list) == 4:
        style_list = ["r", "b", "r--", "b--"]

    if len(lengend_list) == 6:
        # style_list = ["orchid", "r", "g", "b", "purple", "peru", "olive", "coral"]
        style_list = ["r", "g", "b", "r--", "g--", "b--"]
        # style_list = ["r", "r--", "b", "b--", "g", "g--"]
        # style_list = ["m", "r", "g", "b", "c", "y", "k"]

    if len(lengend_list) == 7:
        # style_list = ["m", "r", "g", "b", "r--", "g--", "b--"]
        style_list = ["orchid", "r", "g", "b", "r--", "g--", "b--"]

    if len(lengend_list) == 8:
        style_list = ["r", "b", "g", "k", "r--", "b--", "g--", "k--"]

    if len(lengend_list) == 9:
        style_list = ["r", "r--", "r:", "b", "b--", "b:", "g", "g--", "g:"]

    legend_size = 16
    markevery = 50
    markesize = 3

    plt.subplot(111)

    for i, metrics in enumerate(metric_records):
        plt.plot(metrics, style_list[i], markersize=markesize, markevery=markevery, linewidth=2.3)

    plt.xticks(np.arange(8), ("250", "500", "1000", "2000", "4000", "6000", "8000", "10000"), fontsize=13)
    # plt.xticks(np.arange(6), ("500", "1000", "2000", "4000"), fontsize=13)
    plt.yticks(fontsize=12)
    plt.xlabel("Number of labeled overlapping samples", fontsize=15)
    plt.ylabel("Test accuracy", fontsize=16)
    plt.title("CNN on CIFAR10", fontsize=16)
    plt.legend(lengend_list, fontsize=legend_size, loc='best')
    plt.show()


if __name__ == "__main__":

    benchmark_result = get_benchmark_result()
    guest = get_all_guest_v2()

    # fed_mvt = get_fed_image_as_guest_result_v2()
    # guest_acc = "g_image_acc"
    # guest_data_type = "image"

    fed_mvt = get_fed_mvt_result()
    guest_acc = "g_acc"
    guest_data_type = "text"

    fedmvt_all = []
    fedmvt_guest = []
    guest_all_samples = []
    vallina_VTL = []
    all_acc = "all_acc"
    # n_overlapping_samples_list = [500, 1000, 2000, 4000]
    n_overlapping_samples_list = [250, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    for n_overlap_samples in n_overlapping_samples_list:
        fedmvt_all.append(100*fed_mvt[str(n_overlap_samples)][all_acc])
        # fedmvt_guest.append(100*fed_mvt[str(n_overlap_samples)]["g_acc"])
        guest_all_samples.append(100*guest[str(n_overlap_samples)][guest_acc])
        vallina_VTL.append(100*benchmark_result[str(n_overlap_samples)][all_acc])

    print("guest_all_samples:", guest_all_samples)
    print("vallina_VTL:", vallina_VTL)
    print("fedmvt_guest:", fedmvt_guest)
    print("fedmvt_all:", fedmvt_all)

    # metric_records = [guest_all_samples, vallina_VTL, fedmvt_guest, fedmvt_all]
    # lengend_list = ["Vanilla-local", "Vanilla-VFL", "FedMVT-local", "FedMVT-VFL"]
    metric_records = [guest_all_samples, vallina_VTL, fedmvt_all]
    lengend_list = ["Local Model", "Vanilla-VFL", "FedMVT-VFL"]
    plot_series(metric_records, lengend_list, data_type=guest_data_type)
