import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_all_guest():

    # num_overlapping: 28428(250)
    # [acc]
    # img: 0.5832598062910648(0.005053809377148826), txt: 0.6984109959360509(0.0014963620455066768))

    g_image_acc_250_1 = 0.5832598062910648
    g_txt_acc_250_1 = 0.6984109959360509

    # num_overlapping: 28553(500)
    # [acc]
    # img: 0.583595431190036(0.0029089804778399744), txt: 0.6995516586998943(0.006017012178262104))

    g_image_acc_500_1 = 0.583595431190036
    g_txt_acc_500_1 = 0.6995516586998943

    # num_overlapping: 28803(1000)
    # [acc]
    # img: 0.5851651717779329(0.005761995633708087), txt: 0.7012191847037972(0.004530873743610926))

    g_image_acc_1000_1 = 0.5851651717779329
    g_txt_acc_1000_1 = 0.7012191847037972

    # num_overlapping: 29303(2000)
    # [acc]
    # img: 0.5889145496535796(0.0017125093067752663), txt: 0.7022754533899063(0.0031198268307702595))

    g_image_acc_2000_1 = 0.5889145496535796
    g_txt_acc_2000_1 = 0.7022754533899063

    # num_overlapping: 30303(4000)
    # [acc]
    # img: 0.5891293839626188(0.0005477221669899469), txt: 0.703252430313121(0.004997495811570172))

    g_image_acc_4000_1 = 0.5891293839626188
    g_txt_acc_4000_1 = 0.703252430313121

    # num_overlapping: 32303(8000)
    # [acc]
    # img: 0.5885999606137101(0.0015319241251969444), txt: 0.7031895160857189(0.0019591978115981365))

    g_image_acc_8000_1 = 0.5885999606137101
    g_txt_acc_8000_1 = 0.7031895160857189

    results = dict()
    results["250"] = dict()
    results["500"] = dict()
    results["1000"] = dict()
    results["2000"] = dict()
    results["4000"] = dict()
    results["8000"] = dict()
    results["12000"] = dict()
    results["20000"] = dict()

    results["250"]["g_image_acc"] = g_image_acc_250_1
    results["250"]["g_txt_acc"] = g_txt_acc_250_1

    results["500"]["g_image_acc"] = g_image_acc_500_1
    results["500"]["g_txt_acc"] = g_txt_acc_500_1

    results["1000"]["g_image_acc"] = g_image_acc_1000_1
    results["1000"]["g_txt_acc"] = g_txt_acc_1000_1

    results["2000"]["g_image_acc"] = g_image_acc_2000_1
    results["2000"]["g_txt_acc"] = g_txt_acc_2000_1

    results["4000"]["g_image_acc"] = g_image_acc_4000_1
    results["4000"]["g_txt_acc"] = g_txt_acc_4000_1

    results["8000"]["g_image_acc"] = g_image_acc_8000_1
    results["8000"]["g_txt_acc"] = g_txt_acc_8000_1

    return results


def get_fed_image_as_guest_result_v2():

    # Num_overlap = 250

    all_acc_250_1 = 0.6388270006623713
    g_acc_250_1 = 0.5711928704763052
    h_acc_250_1 = 0.5734569759739869

    # all_acc_250_2 = 0.6641777563677967
    # g_acc_250_2 = 0.5706750165592822
    # h_acc_250_2 = 0.6070933943517793

    all_acc_250 = (all_acc_250_1 + all_acc_250_1) / 2
    g_acc_250 = (g_acc_250_1 + g_acc_250_1) / 2
    h_acc_250 = (h_acc_250_1 + h_acc_250_1) / 2

    # Num_overlap = 500

    all_acc_500_1 = 0.6735713855603059
    g_acc_500_1 = 0.5867434816643584
    h_acc_500_1 = 0.6324561931715542

    # all_acc_500_2 = 0.6966941651110977
    # g_acc_500_2 = 0.5695924007948455
    # h_acc_500_2 = 0.6479556813391943

    all_acc_500 = (all_acc_500_1 + all_acc_500_1) / 2
    g_acc_500 = (g_acc_500_1 + g_acc_500_1) / 2
    h_acc_500 = (h_acc_500_1 + h_acc_500_1) / 2

    # Num_overlap = 1000

    all_acc_1000_1 = 0.6835069548985367
    g_acc_1000_1 = 0.5909211778165834
    h_acc_1000_1 = 0.6495092430902631

    all_acc_1000_2 = 0.6863973023423857
    g_acc_1000_2 = 0.5906222075028603
    h_acc_1000_2 = 0.6762570000602156

    all_acc_1000 = (all_acc_1000_1 + all_acc_1000_2) / 2
    g_acc_1000 = (g_acc_1000_1 + g_acc_1000_2) / 2
    h_acc_1000 = (h_acc_1000_1 + h_acc_1000_2) / 2

    # Num_overlap = 2000

    all_acc_2000_1 = 0.72141675197206
    g_acc_2000_1 = 0.5950042150900223
    h_acc_2000_1 = 0.6780452821099536

    all_acc_2000_2 = 0.713012585054495
    g_acc_2000_2 = 0.5968985909556211
    h_acc_2000_2 = 0.6821219967483592

    all_acc_2000 = (all_acc_2000_1 + all_acc_2000_2) / 2
    g_acc_2000 = (g_acc_2000_1 + g_acc_2000_2) / 2
    h_acc_2000 = (h_acc_2000_1 + h_acc_2000_2) / 2

    # Num_overlap = 4000

    all_acc_4000_1 = 0.7440838200758716
    g_acc_4000_1 = 0.5995324260853857
    h_acc_4000_1 = 0.6957788884205456

    all_acc_4000_2 = 0.7445613295598241
    g_acc_4000_2 = 0.602593761666767
    h_acc_4000_2 = 0.7088095381465648

    all_acc_4000 = (all_acc_4000_1 + all_acc_4000_2) / 2
    g_acc_4000 = (g_acc_4000_1 + g_acc_4000_2) / 2
    h_acc_4000 = (h_acc_4000_1 + h_acc_4000_2) / 2

    # Num_overlap = 8000

    all_acc_8000_1 = 0.7562108147166857
    g_acc_8000_1 = 0.6214565544649847
    h_acc_8000_1 = 0.7233455771662551

    all_acc_8000 = (all_acc_8000_1 + all_acc_8000_1) / 2
    g_acc_8000 = (g_acc_8000_1 + g_acc_8000_1) / 2
    h_acc_8000 = (h_acc_8000_1 + h_acc_8000_1) / 2

    results = dict()
    results["250"] = dict()
    results["500"] = dict()
    results["1000"] = dict()
    results["2000"] = dict()
    results["4000"] = dict()
    results["8000"] = dict()
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

    results["8000"]["all_acc"] = all_acc_8000
    results["8000"]["g_acc"] = g_acc_8000
    results["8000"]["h_acc"] = h_acc_8000

    return results


def get_image_as_guest_result():
    # Num_overlap = 250

    all_fscore_250_1 = 0.6134232227928682
    g_fscore_250_1 = 0.5250147530017818
    h_fscore_250_1 = 0.5409859431013918

    all_acc_250_1 = 0.6318814114614104
    g_acc_250_1 = 0.5395563671518342
    h_acc_250_1 = 0.5768193780546753

    all_auc_250_1 = 0.8429895063191439
    g_auc_250_1 = 0.8091318504725085
    h_auc_250_1 = 0.7298664292569238

    all_fscore_250_2 = 0.6540061575193339
    g_fscore_250_2 = 0.558055931795751
    h_fscore_250_2 = 0.5885883620974676

    all_acc_250_2 = 0.6641777563677967
    g_acc_250_2 = 0.5706750165592822
    h_acc_250_2 = 0.6070933943517793

    all_auc_250_2 = 0.8794059920549469
    g_auc_250_2 = 0.8375873328972725
    h_auc_250_2 = 0.768065205466795

    all_acc_250 = (all_acc_250_1 + all_acc_250_2) / 2
    g_acc_250 = (g_acc_250_1 + g_acc_250_2) / 2
    h_acc_250 = (h_acc_250_1 + h_acc_250_2) / 2

    # Num_overlap = 500

    all_fscore_500_1 = 0.6713664344686264
    g_fscore_500_1 = 0.5523118593292436
    h_fscore_500_1 = 0.6130181445589156

    all_acc_500_1 = 0.6834416456308072
    g_acc_500_1 = 0.5667758741070948
    h_acc_500_1 = 0.6278103013051185

    all_auc_500_1 = 0.8944461559989808
    g_auc_500_1 = 0.8386328383696634
    h_auc_500_1 = 0.7980657163290891

    all_fscore_500_2 = 0.6851562289426328
    g_fscore_500_2 = 0.5509076412710566
    h_fscore_500_2 = 0.6255959889088112

    all_acc_500_2 = 0.6966941651110977
    g_acc_500_2 = 0.5695924007948455
    h_acc_500_2 = 0.6479556813391943

    all_auc_500_2 = 0.8907631234450152
    g_auc_500_2 = 0.8301945394682937
    h_auc_500_2 = 0.8127470726870836

    all_acc_500 = (all_acc_500_1 + all_acc_500_2) / 2
    g_acc_500 = (g_acc_500_1 + g_acc_500_2) / 2
    h_acc_500 = (h_acc_500_1 + h_acc_500_2) / 2

    # Num_overlap = 1000

    all_fscore_1000_1 = 0.7044663084522023
    g_fscore_1000_1 = 0.5568511669472007
    h_fscore_1000_1 = 0.6422537064837037

    all_acc_1000_1 = 0.7112626886513777
    g_acc_1000_1 = 0.569031634352006
    h_acc_1000_1 = 0.657350018798002

    all_auc_1000_1 = 0.8983882223905765
    g_auc_1000_1 = 0.8443808020025388
    h_auc_1000_1 = 0.8154701085385124

    all_fscore_1000_2 = 0.6981982662157163
    g_fscore_1000_2 = 0.5594668507150353
    h_fscore_1000_2 = 0.6532616296897873

    all_acc_1000_2 = 0.712229782621786
    g_acc_1000_2 = 0.5739989161197085
    h_acc_1000_2 = 0.6715240561209129

    all_auc_1000_2 = 0.9086947742720456
    g_auc_1000_2 = 0.8357904187767374
    h_auc_1000_2 = 0.8382483952158573

    all_acc_1000 = (all_acc_1000_1 + all_acc_1000_2) / 2
    g_acc_1000 = (g_acc_1000_1 + g_acc_1000_2) / 2
    h_acc_1000 = (h_acc_1000_1 + h_acc_1000_2) / 2

    # Num_overlap = 2000

    all_fscore_2000_1 = 0.7257106274191737
    g_fscore_2000_1 = 0.5651474846245487
    h_fscore_2000_1 = 0.6775543442195744

    all_acc_2000_1 = 0.7331220795961115
    g_acc_2000_1 = 0.5767549277619637
    h_acc_2000_1 = 0.6942370696600247

    all_auc_2000_1 = 0.9073708548908177
    g_auc_2000_1 = 0.8375382013800211
    h_auc_2000_1 = 0.8473990192370977

    all_fscore_2000_2 = 0.7221669930976123
    g_fscore_2000_2 = 0.5543944370197538
    h_fscore_2000_2 = 0.6629598510194186

    all_acc_2000_2 = 0.7330643704461974
    g_acc_2000_2 = 0.5729644728126693
    h_acc_2000_2 = 0.6854579394231348

    all_auc_2000_2 = 0.9139101735472115
    g_auc_2000_2 = 0.8413363444049609
    h_auc_2000_2 = 0.8522735176588572

    all_acc_2000 = (all_acc_2000_1 + all_acc_2000_2) / 2
    g_acc_2000 = (g_acc_2000_1 + g_acc_2000_2) / 2
    h_acc_2000 = (h_acc_2000_1 + h_acc_2000_2) / 2

    # Num_overlap = 4000

    all_fscore_4000_1 = 0.7381622187545613
    g_fscore_4000_1 = 0.5678276945488054
    h_fscore_4000_1 = 0.6783223370410588

    all_acc_4000_1 = 0.7426438584241903
    g_acc_4000_1 = 0.5792986733981417
    h_acc_4000_1 = 0.7006069069230356

    all_auc_4000_1 = 0.9279494283435593
    g_auc_4000_1 = 0.8518436622808798
    h_auc_4000_1 = 0.8635534671644006

    all_fscore_4000_2 = 0.7436754659160413
    g_fscore_4000_2 = 0.5629008713616529
    h_fscore_4000_2 = 0.6901171329853784

    all_acc_4000_2 = 0.7445613295598241
    g_acc_4000_2 = 0.578793761666767
    h_acc_4000_2 = 0.7088095381465648

    all_auc_4000_2 = 0.922546522058239
    g_auc_4000_2 = 0.8415757239501259
    h_auc_4000_2 = 0.8692853388758707

    all_acc_4000 = (all_acc_4000_1 + all_acc_4000_2) / 2
    g_acc_4000 = (g_acc_4000_1 + g_acc_4000_2) / 2
    h_acc_4000 = (h_acc_4000_1 + h_acc_4000_2) / 2

    # Num_overlap = 8000

    all_fscore_8000_1 = 0.7472430866384748
    g_fscore_8000_1 = 0.5615854387193706
    h_fscore_8000_1 = 0.7098548404625721

    all_acc_8000_1 = 0.7502108147166857
    g_acc_8000_1 = 0.5824565544649847
    h_acc_8000_1 = 0.7233455771662551

    all_auc_8000_1 = 0.9320617413676138
    g_auc_8000_1 = 0.8461607688706456
    h_auc_8000_1 = 0.882205067177978

    all_acc_8000 = (all_acc_8000_1 + all_acc_8000_1) / 2
    g_acc_8000 = (g_acc_8000_1 + g_acc_8000_1) / 2
    h_acc_8000 = (h_acc_8000_1 + h_acc_8000_1) / 2

    results = dict()
    results["250"] = dict()
    results["500"] = dict()
    results["1000"] = dict()
    results["2000"] = dict()
    results["4000"] = dict()
    results["8000"] = dict()
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

    results["8000"]["all_acc"] = all_acc_8000
    results["8000"]["g_acc"] = g_acc_8000
    results["8000"]["h_acc"] = h_acc_8000

    return results


def get_fed_text_as_guest_result():
    # Num_overlap = 250

    all_acc_250_1 = 0.6876016137773228
    g_acc_250_1 = 0.7325344734148251
    h_acc_250_1 = 0.4942373697838261

    all_acc_250 = all_acc_250_1
    g_acc_250 = g_acc_250_1
    h_acc_250 = h_acc_250_1

    # Num_overlap = 500

    all_acc_500_1 = 0.7122899981935329
    g_acc_500_1 = 0.7337018124887096
    h_acc_500_1 = 0.5054856385861384

    all_acc_500 = all_acc_500_1
    g_acc_500 = g_acc_500_1
    h_acc_500 = h_acc_500_1

    # Num_overlap = 1000

    all_acc_1000_1 = 0.7298729451436141
    g_acc_1000_1 = 0.7366504546275668
    h_acc_1000_1 = 0.5272475462154513

    all_acc_1000_2 = 0.7256578551213344
    g_acc_1000_2 = 0.736364840127657
    h_acc_1000_2 = 0.5188655386282892

    all_acc_1000 = (all_acc_1000_1 + all_acc_1000_2) / 2
    g_acc_1000 = (g_acc_1000_1 + g_acc_1000_2) / 2
    h_acc_1000 = (h_acc_1000_1 + h_acc_1000_2) / 2

    # Num_overlap = 2000

    all_acc_2000_1 = 0.7389052809056422
    g_acc_2000_1 = 0.7405672306858554
    h_acc_2000_1 = 0.5291383151683026

    all_acc_2000_2 = 0.7401698079123261
    g_acc_2000_2 = 0.7404717287890649
    h_acc_2000_2 = 0.5443487685915577

    all_acc_2000 = (all_acc_2000_1 + all_acc_2000_2) / 2
    g_acc_2000 = (g_acc_2000_1 + g_acc_2000_2) / 2
    h_acc_2000 = (h_acc_2000_1 + h_acc_2000_2) / 2

    # Num_overlap = 4000

    all_acc_4000_1 = 0.7637743120370928
    g_acc_4000_1 = 0.7451100138495815
    h_acc_4000_1 = 0.5543608117059072

    all_acc_4000_2 = 0.7562473655687361
    g_acc_4000_2 = 0.7430721984705244
    h_acc_4000_2 = 0.5443608117059072

    all_acc_4000_3 = 0.7578731860059011
    g_acc_4000_3 = 0.7446880833383513
    h_acc_4000_3 = 0.5515746372011803

    all_acc_4000 = (all_acc_4000_1 + all_acc_4000_2 + all_acc_4000_3) / 3
    g_acc_4000 = (g_acc_4000_1 + g_acc_4000_2 + g_acc_4000_3) / 3
    h_acc_4000 = (h_acc_4000_1 + h_acc_4000_2 + h_acc_4000_3) / 3

    # Num_overlap = 8000

    all_acc_8000_1 = 0.7757572108147167
    g_acc_8000_1 = 0.7484434274703438
    h_acc_8000_1 = 0.5441440356476185

    all_acc_8000_2 = 0.7732883723730957
    g_acc_8000_2 = 0.7497337267417354
    h_acc_8000_2 = 0.5326187752152707

    all_acc_8000 = (all_acc_8000_1 + all_acc_8000_2) / 2
    g_acc_8000 = (g_acc_8000_1 + g_acc_8000_2) / 2
    h_acc_8000 = (h_acc_8000_1 + h_acc_8000_2) / 2

    results = dict()
    results["250"] = dict()
    results["500"] = dict()
    results["1000"] = dict()
    results["2000"] = dict()
    results["4000"] = dict()
    results["8000"] = dict()
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

    results["8000"]["all_acc"] = all_acc_8000
    results["8000"]["g_acc"] = g_acc_8000
    results["8000"]["h_acc"] = h_acc_8000

    return results


def get_benchmark_result():
    #
    # num_overlapping = 250
    #
    # [fscore]
    img_score_250_1 = 0.33873741197173846  # (0.003650730772286298)
    txt_score_250_1 = 0.43037354550225393  # (0.006843189381831484)
    img_txt_score_250_1 = 0.3324502822445149  # (0.009875998578196564)
    # [acc]
    img_acc_250_1 = 0.3156023416939685  # (0.006843189381831484)
    txt_acc_250_1 = 0.44770395832214405  # (0.028394895334533363)
    img_txt_acc_250_1 = 0.30235780654170474  # (0.011749767397919433)
    # [auc]
    img_auc_250_1 = 0.6291614036010047  # (0.014287966043243932)
    txt_auc_250_1 = 0.6572689708537157  # (0.015931498400320775)
    img_txt_auc_250_1 = 0.6193287116785913  # (0.008019100712660413)

    # [fscore]
    img_fscore_250_2 = 0.342465936415762  # (0.01110239175515212)
    txt_fscore_250_2 = 0.4215417883860839  # (0.013357795513473743)
    img_txt_fscore_250_2 = 0.3241096968479006  # (0.011884108624801555)
    # [acc]
    img_acc_250_2 = 0.3261516228096586  # (0.013357795513473743)
    txt_acc_250_2 = 0.43964593243812855  # (0.012018485082206216)
    img_txt_acc_250_2 = 0.2922743421448787  # (0.014005184098230618)
    # [auc]
    img_auc_250_2 = 0.6312880642547782  # (0.011100423436959894)
    txt_auc_250_2 = 0.6525203621451161  # (0.011209243547289631)
    img_txt_250_2 = 0.6136664015652427  # (0.01198842103656269)

    img_acc_250 = (img_acc_250_1 + img_acc_250_2) / 2
    txt_acc_250 = (txt_acc_250_1 + txt_acc_250_2) / 2
    img_txt_acc_250 = (img_txt_acc_250_1 + img_txt_acc_250_2) / 2

    #
    # num_overlapping: 500
    #
    # [fscore]
    img_fscore_500_1 = 0.44018428824448286  # (0.01820281185376018)
    txt_fscore_500_1 = 0.5774203520226476  # (0.016093835892552442)
    img_txt_fscore_500_1 = 0.47892790930340945  # (0.01212478366190498)
    # [acc]
    img_acc_500_1 = 0.46033621569364624  # (0.016093835892552442)
    txt_acc_500_1 = 0.6038347924163489  # (0.008983786694619965)
    img_txt_acc_500_1 = 0.49186315054514207  # (0.012183681546215318)
    # [auc]
    img_auc_500_1 = 0.718844488883853  # (0.018288350569117512)
    txt_auc_500_1 = 0.7719631889309893  # (0.01580929147441737)
    img_txt_auc_500_1 = 0.7446095975309405  # (0.013288739365226945)

    # [fscore]
    img_fscore_500_2 = 0.42977627054798867  # (0.009309102220943404)
    txt_fscore_500_2 = 0.5824910113685001  # (0.009914606803102207)
    img_txt_fscore_500_2 = 0.47381558775306154  # (0.008280458637844532)
    # [acc]
    img_acc_500_2 = 0.4500150538929367  # (0.009914606803102207)
    txt_acc_500_2 = 0.604287348708376  # (0.008961857467029372)
    img_txt_acc_500_2 = 0.4885289335822244  # (0.007096757039121392)
    # [auc]
    img_auc_500_2 = 0.7182647558877242  # (0.005732268144466972)
    txt_auc_500_2 = 0.7810871043619564  # (0.00819872312580587)
    img_txt_auc_500_2 = 0.7409539054413157  # (0.005696360128751877)

    img_acc_500 = (img_acc_500_1 + img_acc_500_2) / 2
    txt_acc_500 = (txt_acc_500_1 + txt_acc_500_2) / 2
    img_txt_acc_500 = (img_txt_acc_500_1 + img_txt_acc_500_2) / 2

    #
    # num_overlapping: 1000
    #
    # [fscore]
    img_fscore_1000_1 = 0.507510796675433  # (0.005608102517908071)
    txt_fscore_1000_1 = 0.6579595762698551  # (0.006193472301347079)
    img_txt_fscore_1000_1 = 0.6170687965109635  # (0.0033572633050433227)
    # [acc]
    img_acc_1000_1 = 0.5264729577313497  # (0.006193472301347079)
    txt_acc_1000_1 = 0.6682313765508352  # (0.0022443245793427437)
    img_txt_acc_1000_1 = 0.6336000859337236  # (0.00380429647506857)
    # [auc]
    img_auc_1000_1 = 0.7949122162954314  # (0.006026727446215345)
    txt_auc_1000_1 = 0.842363642334839  # (0.003667840100956873)
    img_txt_auc_1000_1 = 0.8612824744235832  # (0.004292115175783617)

    # [fscore]
    img_fscore_1000_2 = 0.5096055122569526  # (0.007099571064210442)
    txt_fscore_1000_2 = 0.668563838056482  # (0.006817466518093064)
    img_txt_fscore_1000_2 = 0.6290863579099366  # (0.00442608367832995)
    # [acc]
    img_acc_1000_2 = 0.5278737881616186  # (0.006817466518093064)
    txt_acc_1000_2 = 0.6788101403022822  # (0.003754091998880072)
    img_txt_acc_1000_2 = 0.6446919973505149  # (0.005633500632184011)
    # [auc]
    img_auc_1000_2 = 0.7940663904734522  # (0.005115465363423301)
    txt_auc_1000_2 = 0.8500902277703967  # (0.006002882682465373)
    img_txt_auc_1000_2 = 0.864811148132706  # (0.0026255839426057503)

    img_acc_1000 = (img_acc_1000_1 + img_acc_1000_2) / 2
    txt_acc_1000 = (txt_acc_1000_1 + txt_acc_1000_2) / 2
    img_txt_acc_1000 = (img_txt_acc_1000_1 + img_txt_acc_1000_2) / 2

    #
    # num_overlapping: 2000
    #
    # [fscore]
    img_fscore_2000_1 = 0.5436306744389627  # (0.005260529798674754)
    txt_fscore_2000_1 = 0.6802350900391725  # (0.004674633633648177)
    img_txt_fscore_2000_1 = 0.6665031132160582  # (0.008270120023453708)
    # [acc]
    img_acc_2000_1 = 0.5619850690155218  # (0.004674633633648177)
    txt_acc_2000_1 = 0.6860303990547291  # (0.007114097930868852)
    img_txt_acc_2000_1 = 0.6786078736774263  # (0.007478810904812235)
    # [auc]
    img_auc_2000_1 = 0.8261278656846567  # (0.006064062583232691)
    txt_auc_2000_1 = 0.8658708383322111  # (0.0061146675277696004)
    img_txt_auc_2000_1 = 0.8897121889659075  # (0.0015931941774739056)

    # [fscore]
    img_fscore_2000_2 = 0.54124263781125  # (0.003792710938468023)
    txt_fscore_2000_2 = 0.6970670365623101  # (0.003499016979842725)
    img_txt_fscore_2000_2 = 0.6724436168623074  # (0.003488631134031452)
    # [acc]
    img_acc_2000_2 = 0.5608960077075932  # (0.003499016979842725)
    txt_acc_2000_2 = 0.6937556452098512  # (0.0030558201292763356)
    img_txt_acc_2000_2 = 0.6870958029746493  # (0.0033987623189945977)
    # [auc]
    img_auc_2000_2 = 0.8240461449755113  # (0.003941192608315244)
    txt_auc_2000_2 = 0.869400916689351  # (0.005232911169978839)
    img_txt_auc_2000_2 = 0.8912522130978896  # (0.0057816626338550005)

    img_acc_2000 = (img_acc_2000_1 + img_acc_2000_2) / 2
    txt_acc_2000 = (txt_acc_2000_1 + txt_acc_2000_2) / 2
    img_txt_acc_2000 = (img_txt_acc_2000_1 + img_txt_acc_2000_2) / 2

    #
    # num_overlapping: 4000
    #
    # [fscore] #
    img_fscore_4000_1 = 0.5626314053859504  # (0.0013752741432987634)
    txt_fscore_4000_1 = 0.6945269453748061  # (0.0020033803279445603)
    img_txt_fscore_4000_1 = 0.6982655615062296  # (0.0032993990184270803)
    # [acc]
    img_acc_4000_1 = 0.5785595359578926  # (0.0020033803279445603)
    txt_acc_4000_1 = 0.6947848971480746  # (0.003610462982834915)
    img_txt_acc_4000_1 = 0.7087491272356196  # (0.0025538188463342937)
    # [auc]
    img_auc_4000_1 = 0.8452129492753746  # (0.004318185293767963)
    txt_auc_4000_1 = 0.8771485267295936  # (0.0043184206390867)
    img_txt_auc_4000_1 = 0.9066225336843454  # (0.004355673451492476)

    # [fscore]
    img_fscore_4000_2 = 0.5564053761191464  # (0.0048878450465077985)
    txt_fscore_4000_2 = 0.7011160915107221  # (0.005451741087094873)
    img_txt_fscore_4000_2 = 0.6979842819362547  # (0.006519145522882329)
    # [acc]
    img_acc_4000_2 = 0.5733606310591919  # (0.005451741087094873)
    txt_acc_4000_2 = 0.70066839284639  # (0.005007242514670066)
    img_txt_acc_4000_2 = 0.7095441681218764  # (0.0063423435542382944)
    # [auc]
    img_auc_4000_2 = 0.8443016636836788  # (0.004178601863499376)
    txt_auc_4000_2 = 0.8802471749584944  # (0.00242474916782574)
    img_txt_auc_4000_2 = 0.90935466213983  # (0.0034706962354960225)

    img_acc_4000 = (img_acc_4000_1 + img_acc_4000_2) / 2
    txt_acc_4000 = (txt_acc_4000_1 + txt_acc_4000_2) / 2
    img_txt_acc_4000 = (img_txt_acc_4000_1 + img_txt_acc_4000_2) / 2

    #
    # num_overlapping: 8000
    #
    # [fscore]
    img_fscore_8000_1 = 0.5743232691870341  # (0.004067425994027846)
    txt_fscore_8000_1 = 0.698687959623908  # (0.002887504185015258)
    img_txt_fscore_8000_1 = 0.7080032193258413  # (0.003946452967707246)
    # [acc]
    img_acc_8000_1 = 0.5809528975777432  # (0.002887504185015258)
    txt_acc_8000_1 = 0.6946989634244589  # (0.00449917620453629)
    img_txt_acc_8000_1 = 0.7135194693592566  # (0.005224340179482907)
    # [auc]
    img_auc_8000_1 = 0.8522003795634395  # (0.0017581832986817477)
    txt_auc_8000_1 = 0.8823625347232815  # (0.0027204844173574975)
    img_txt_auc_8000_1 = 0.9116374667999038  # (0.003044841121222076)

    # [fscore]
    img_fscore_8000_2 = 0.5671696916973695  # (0.0046324931718979626)
    txt_fscore_8000_2 = 0.6975100276415194  # (0.005218352460616722)
    img_txt_fscore_8000_2 = 0.7065441019362254  # (0.006853017746642202)
    # [acc]
    img_acc_8000_2 = 0.5786965737339675  # (0.005218352460616722)
    txt_acc_8000_2 = 0.7002830131872102  # (0.005829364973206709)
    img_txt_acc_8000_2 = 0.7155295959535136  # (0.0026803438407217105)
    # [auc]
    img_auc_8000_2 = 0.8504946540412007  # (0.0031623765396537997)
    txt_auc_8000_2 = 0.8839427752833334  # (0.003715420837702678)
    img_txt_auc_8000_2 = 0.9157449889641456  # (0.002656512806806591)

    img_acc_8000 = (img_acc_8000_1 + img_acc_8000_2) / 2
    txt_acc_8000 = (txt_acc_8000_1 + txt_acc_8000_2) / 2
    img_txt_acc_8000 = (img_txt_acc_8000_1 + img_txt_acc_8000_2) / 2

    #
    # num_overlapping: 12000
    #
    # [fscore]
    img_fscore_12000_1 = 0.5730318892152264  # (0.004641328681605091)
    txt_fscore_12000_1 = 0.6986252984244452  # (0.003861583210311216)
    img_txt_fscore_12000_1 = 0.7095717163263595  # (0.003744453376485432)

    # [acc]
    img_acc_12000_1 = 0.5808163166657715  # (0.003861583210311216)
    txt_acc_12000_1 = 0.6975777431655835  # (0.002440852460766873)
    img_txt_acc_12000_1 = 0.7184969654653849  # (0.006643349374572847)

    # [auc]
    img_auc_12000_1 = 0.8534699823505945  # (0.004897877984297234)
    txt_auc_12000_1 = 0.8831294443815526  # (0.0017568264632639293)
    img_txt_auc_12000_1 = 0.9124015975829798  # (0.003473060963672515)

    # [fscore]
    img_fscore_12000_2 = 0.5762812167940599  # (0.004450099700834028)
    txt_fscore_12000_2 = 0.7006415365589961  # (0.003655516707960591)
    img_txt_fscore_12000_2 = 0.7124536415989889  # (0.012002470174454872)
    # [acc]
    img_acc_12000_2 = 0.5813821882338774  # (0.003655516707960591)
    txt_acc_12000_2 = 0.7009815138194737  # (0.0030733310509741575)
    img_txt_acc_12000_2 = 0.7197928584331909  # (0.0049768233797826515)
    # [auc]
    img_auc_12000_2 = 0.8540674621748572  # (0.003976252514569345)
    txt_auc_12000_2 = 0.8876560062053359  # (0.002193369214465236)
    img_txt_auc_12000_2 = 0.9177054708518749  # (0.00309189640433882)

    img_acc_12000 = (img_acc_12000_1 + img_acc_12000_2) / 2
    txt_acc_12000 = (txt_acc_12000_1 + txt_acc_12000_2) / 2
    img_txt_acc_12000 = (img_txt_acc_12000_1 + img_txt_acc_12000_2) / 2

    #
    # num_overlapping: 20000
    #
    # [fscore]
    img_fscore_20000_1 = 0.5747863818400566  # (0.0059530405531615415)
    txt_fscore_20000_1 = 0.6932696606053672  # (0.0021336302687833294)
    img_txt_fscore_20000_1 = 0.7138193925651469  # (0.007062449663183293)
    # [acc]
    img_acc_20000_1 = 0.5813053332617218  # (0.0021336302687833294)
    txt_acc_20000_1 = 0.6935066330092916  # (0.004223502641615633)
    img_txt_acc_20000_1 = 0.7198210054245662  # (0.006593138590666575)
    # [auc]
    img_auc_20000_1 = 0.8525440095561994  # (0.0022380838735905485)
    txt_auc_20000_1 = 0.8795721789543679  # (0.004855595378581716)
    img_txt_auc_20000_1 = 0.9135074968477914  # (0.003918787734931345)

    # [fscore]
    img_fscore_20000_2 = 0.5804661973086859  # (0.003249356818326284)
    txt_fscore_20000_2 = 0.7008466322451994  # (0.002202408717402757)
    img_txt_fscore_20000_2 = 0.7162512664928997  # (0.006197446675996066)
    # [acc]
    img_acc_20000_2 = 0.5816101643885109  # (0.002202408717402757)
    txt_acc_20000_2 = 0.699981935328476  # (0.004013737128985887)
    img_txt_acc_20000_2 = 0.7213705064129583  # (0.003790310765167476)
    # [auc]
    img_auc_20000_2 = 0.8548589401166721  # (0.002254900773210017)
    txt_auc_20000_2 = 0.8900181307216481  # (0.0038683059867802836)
    img_txt_auc_20000_2 = 0.9182949033688548  # (0.0025060309832712247)

    img_acc_20000 = (img_acc_20000_1 + img_acc_20000_2) / 2
    txt_acc_20000 = (txt_acc_20000_1 + txt_acc_20000_2) / 2
    img_txt_acc_20000 = (img_txt_acc_20000_1 + img_txt_acc_20000_2) / 2

    results = dict()
    results["250"] = dict()
    results["500"] = dict()
    results["1000"] = dict()
    results["2000"] = dict()
    results["4000"] = dict()
    results["8000"] = dict()
    results["12000"] = dict()
    results["20000"] = dict()

    results["250"]["all_acc"] = img_txt_acc_250
    results["250"]["g_acc"] = img_acc_250
    results["250"]["h_acc"] = txt_acc_250

    results["500"]["all_acc"] = img_txt_acc_500
    results["500"]["g_acc"] = img_acc_500
    results["500"]["h_acc"] = txt_acc_500

    results["1000"]["all_acc"] = img_txt_acc_1000
    results["1000"]["g_acc"] = img_acc_1000
    results["1000"]["h_acc"] = txt_acc_1000

    results["2000"]["all_acc"] = img_txt_acc_2000
    results["2000"]["g_acc"] = img_acc_2000
    results["2000"]["h_acc"] = txt_acc_2000

    results["4000"]["all_acc"] = img_txt_acc_4000
    results["4000"]["g_acc"] = img_acc_4000
    results["4000"]["h_acc"] = txt_acc_4000

    results["8000"]["all_acc"] = img_txt_acc_8000
    results["8000"]["g_acc"] = img_acc_8000
    results["8000"]["h_acc"] = txt_acc_8000

    results["12000"]["all_acc"] = img_txt_acc_12000
    results["12000"]["g_acc"] = img_acc_12000
    results["12000"]["h_acc"] = txt_acc_12000

    results["20000"]["all_acc"] = img_txt_acc_20000
    results["20000"]["g_acc"] = img_acc_20000
    results["20000"]["h_acc"] = txt_acc_20000

    return results


def plot_series(metric_records, lengend_list, scenario="", data_type=""):
    plt.rcParams['pdf.fonttype'] = 42

    # style_list = ["r", "b", "g", "k", "m", "y", "c"]
    # style_list = ["r", "g", "g--", "k", "k--", "y", "y--"]
    # style_list = ["r", "b", "g", "k", "r--", "b--", "g--", "k--"]
    # style_list = ["r", "b", "g", "r--", "b--", "g--", "r-.", "b-.", "g-."]
    # style_list = ["r", "b", "g", "r--", "b--", "g--", "r-.", "b-.", "g-."]

    # style_list = ["r", "b", "g", "k", "m", "y", "c"]
    style_list = ["orchid", "red", "green", "blue", "purple", "peru", "olive", "coral"]

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

    plt.xticks(np.arange(6), ("250", "500", "1000", "2000", "4000", "8000"), fontsize=13)
    plt.yticks(fontsize=12)
    plt.xlabel("Number of labeled overlapping samples", fontsize=15)
    plt.ylabel("Test accuracy", fontsize=16)
    plt.title(scenario + " Party A with " + data_type, fontsize=16)
    plt.legend(lengend_list, fontsize=legend_size, loc='best')
    plt.show()


if __name__ == "__main__":

    benchmark_result = get_benchmark_result()
    guest = get_all_guest()

    fed_mvt = get_fed_image_as_guest_result_v2()
    guest_acc = "g_image_acc"
    guest_data_type = "image"
    scenario = "Scenario-1:"

    # fed_mvt = get_fed_text_as_guest_result()
    # guest_acc = "g_txt_acc"
    # guest_data_type = "text"
    # scenario = "Scenario-2:"

    fedmvt_all = []
    fedmvt_guest = []
    guest_all_samples = []
    vallina_VTL = []
    all_acc = "all_acc"
    n_overlapping_samples_list = [250, 500, 1000, 2000, 4000, 8000]
    for n_overlap_samples in n_overlapping_samples_list:
        # fedmvt_all.append(fed_mvt[str(n_overlap_samples)][all_acc])
        # fedmvt_guest.append(fed_mvt[str(n_overlap_samples)]["g_acc"])
        # guest_all_samples.append(guest[str(n_overlap_samples)][guest_acc])
        # vallina_VTL.append(benchmark_result[str(n_overlap_samples)][all_acc])

        fedmvt_all.append(100*fed_mvt[str(n_overlap_samples)][all_acc])
        fedmvt_guest.append(100*fed_mvt[str(n_overlap_samples)]["g_acc"])
        guest_all_samples.append(100*guest[str(n_overlap_samples)][guest_acc])
        vallina_VTL.append(100*benchmark_result[str(n_overlap_samples)][all_acc])

    print("guest_all_samples:", guest_all_samples)
    print("vallina_VTL:", vallina_VTL)
    print("fedmvt_guest:", fedmvt_guest)
    print("fedmvt_all:", fedmvt_all)

    metric_records = [guest_all_samples, vallina_VTL, fedmvt_guest, fedmvt_all]
    lengend_list = ["Vanilla-local", "Vanilla-VFL", "FedMVT-local", "FedMVT-VFL"]
    plot_series(metric_records, lengend_list, scenario=scenario, data_type=guest_data_type)
