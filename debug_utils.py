import numpy as np


def analyze_estimated_labels(nl_ested_lbls_for_att_ested_guest_reprs,
                             nl_ested_lbls_for_lr_ested_guest_reprs,
                             nl_ested_lbls_for_lr_host_reprs,
                             nl_ested_lbls_for_fed_ested_guest_reprs):

    print("=" * 20 + " analyze_estimated_labels " + "=" * 20)
    print("[INFO] nl_ested_lbls_for_att_ested_guest_reprs shape:", nl_ested_lbls_for_att_ested_guest_reprs.shape)
    print("[INFO] nl_ested_lbls_for_lr_ested_guest_reprs shape:", nl_ested_lbls_for_lr_ested_guest_reprs.shape)
    print("[INFO] nl_ested_lbls_for_lr_host_reprs shape:", nl_ested_lbls_for_lr_host_reprs.shape)
    print("[INFO] fed_host_non_overlap_ested_lbls shape:", nl_ested_lbls_for_fed_ested_guest_reprs.shape)

    present_count = 0
    for att_lbl, lr_host_lbl, lr_lbl, fed_lbl in zip(nl_ested_lbls_for_att_ested_guest_reprs,
                                                     nl_ested_lbls_for_lr_host_reprs,
                                                     nl_ested_lbls_for_lr_ested_guest_reprs,
                                                     nl_ested_lbls_for_fed_ested_guest_reprs):
        fed_lbl_idx = np.argmax(fed_lbl)
        lr_host_lbl_idx = np.argmax(lr_host_lbl)
        lr_lbl_idx = np.argmax(lr_lbl)
        att_lbl_idx = np.argmax(att_lbl)

        fed_lbl_prob = np.max(fed_lbl)
        lr_host_lbl_prob = np.max(lr_host_lbl)
        lr_lbl_prob = np.max(lr_lbl)
        att_lbl_prob = np.max(att_lbl)
        # print("fed_lbl:", fed_lbl)
        # print("lr_lbl:", lr_lbl)
        # if fed_lbl_idx == lr_lbl_idx and lr_lbl_prob > 0.6 and fed_lbl_prob > 0.6:
        if fed_lbl_idx == lr_lbl_idx and present_count < 20:
            # if present_count < 20:
            # print("lr_lbl:", lr_lbl)
            # print("fed_lbl:", fed_lbl)
            print("att         lr_host         lr_guest         fed")
            print("[{0}]:{1:0.6f}, [{2}]:{3:0.6f}, [{4}]:{5:0.6f},  [{6}]:{7:0.6f}".format(att_lbl_idx,
                                                                                           att_lbl_prob,
                                                                                           lr_host_lbl_idx,
                                                                                           lr_host_lbl_prob,
                                                                                           lr_lbl_idx,
                                                                                           lr_lbl_prob,
                                                                                           fed_lbl_idx,
                                                                                           fed_lbl_prob))
            present_count += 1

    print("total number of equal predictions: {0}/{1}".format(present_count,
                                                              len(nl_ested_lbls_for_fed_ested_guest_reprs)))
    print("=" * 47)
