import torch
import torch.nn.functional as F


def get_shared_reprs_loss(Ug_overlap_comm, Uh_overlap_comm):
    shared_reprs_loss = F.mse_loss(Ug_overlap_comm, Uh_overlap_comm, reduction="mean")
    return shared_reprs_loss


def get_orth_reprs_loss(uniq_reprs, comm_reprs):
    num_samples = torch.tensor(uniq_reprs.shape[0]).float()
    loss = torch.norm(torch.matmul(uniq_reprs, torch.transpose(comm_reprs, 0, 1))) / num_samples
    return loss


# def get_orth_reprs_loss(uniq_reprs, comm_reprs):
# return F.cosine_similarity(uniq_reprs, comm_reprs, dim=-1).mean()
#     loss = F.mse_loss(uniq_reprs, comm_reprs, reduction="mean")
#     return loss


def get_label_estimation_loss(pred_soft_lbls, true_lbls):
    """
    pred_soft_lbls is the output from a softmax layer (num_examples x num_classes)
    true_lbls is labels (num_examples x num_classes). Note that y is one-hot encoded vector.
    """
    log_likelihood = -torch.sum(torch.log(pred_soft_lbls + 1e-8) * true_lbls, dim=1)
    loss = torch.mean(log_likelihood)
    return loss


def get_alignment_loss(ested_repr, repr):
    return F.mse_loss(ested_repr, repr, reduction="mean")
    # return - F.cosine_similarity(ested_repr, repr, dim=-1).mean()

    # @staticmethod
    # def get_shared_reprs_loss(Ug_overlap_comm, Uh_overlap_comm, W_hg=None):
    #     num_samples = tf.cast(tf.shape(input=Ug_overlap_comm)[0], tf.float32)
    #     if W_hg is None:
    #         shared_reprs_loss = tf.nn.l2_loss(Ug_overlap_comm - Uh_overlap_comm) / num_samples
    #     else:
    #         transformed_Uh = tf.matmul(Uh_overlap_comm, W_hg)
    #         shared_reprs_loss = tf.nn.l2_loss(Ug_overlap_comm - transformed_Uh)
    #     print("shared_reprs_loss shape", shared_reprs_loss.shape)
    #     return shared_reprs_loss
    #
    # @staticmethod
    # def get_orth_reprs_loss(uniq_reprs, comm_reprs):
    #     num_samples = tf.cast(tf.shape(input=uniq_reprs)[0], tf.float32)
    #     loss = tf.nn.l2_loss(tf.matmul(uniq_reprs, tf.transpose(a=comm_reprs))) / num_samples
    #     return loss
    #
    # @staticmethod
    # def get_label_estimation_loss(pred_soft_lbls, true_lbls):
    #     """
    #     pred_soft_lbls is the output from a softmax layer (num_examples x num_classes)
    #     true_lbls is labels (num_examples x num_classes). Note that y is one-hot encoded vector.
    #     """
    #     log_likelihood = -tf.reduce_sum(input_tensor=tf.math.log(pred_soft_lbls + 1e-8) * true_lbls, axis=1)
    #     loss = tf.reduce_mean(input_tensor=log_likelihood)
    #     return loss
    #
    # @staticmethod
    # def get_alignment_loss(ested_repr, repr):
    #     num_samples = tf.cast(tf.shape(input=ested_repr)[0], tf.float32)
    #     # return tf.nn.l2_loss(ested_repr - repr)
    #     return tf.nn.l2_loss(ested_repr - repr) / num_samples
