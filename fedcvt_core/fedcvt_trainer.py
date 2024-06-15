import csv
import json
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from dataset.data_utils import ForeverDataIterator
from fedcvt_core.fedcvt_parties import VFTLGuest, VFLHost
from fedcvt_core.fedcvt_repr_estimator import concat_reprs
from fedcvt_core.fedcvt_utils import aggregate, get_fed_input_dim
from fedcvt_core.param import FederatedTrainingParam
from fedcvt_core.semi_ssl_on_labeled_guest import SemiSupervisedLearningV2
from models.mlp_models import SoftmaxRegression
from models.regularization import EarlyStoppingCheckPoint
from utils import convert_to_1d_labels, f_score_v2


class VerticalFederatedTransferLearning(object):

    def __init__(self,
                 vftl_guest: VFTLGuest,
                 vftl_host: VFLHost,
                 fed_training_param: FederatedTrainingParam,
                 debug=False):
        self.vftl_guest = vftl_guest
        self.vftl_host = vftl_host
        self.fed_training_param = fed_training_param
        self.n_class = self.vftl_guest.n_classes
        self.stop_training = False
        self.fed_lr = None
        self.guest_lr = None
        self.host_lr = None
        self.debug = debug
        self.device = fed_training_param.device
        self.aggregation_mode = fed_training_param.aggregation_mode

        self.criteria = None
        self.fed_optimizer = None
        self.guest_optimizer = None
        self.host_optimizer = None
        self.scheduler_list = None

        self.semi_sl = None

    def save_model(self, file_path):
        print("[INFO] TODO: save model")

    def save_info(self, file_path, eval_info):
        field_names = list(eval_info.keys())
        with open(file_path + ".csv", "a") as logfile:
            logger = csv.DictWriter(logfile, fieldnames=field_names)
            logger.writerow(eval_info)

    def build(self):
        learning_rate = self.fed_training_param.learning_rate
        weight_decay = self.fed_training_param.weight_decay
        fed_input_dim = self.fed_training_param.fed_input_dim
        fed_hidden_dim = self.fed_training_param.fed_hidden_dim
        guest_input_dim = self.fed_training_param.guest_input_dim
        host_input_dim = self.fed_training_param.host_input_dim
        guest_hidden_dim = self.fed_training_param.guest_hidden_dim
        host_hidden_dim = self.fed_training_param.host_hidden_dim
        fed_reg_lambda = self.fed_training_param.fed_reg_lambda
        guest_reg_lambda = self.fed_training_param.guest_reg_lamba
        loss_weight_dict = self.fed_training_param.loss_weight_dict
        sharpen_temp = self.fed_training_param.sharpen_temperature
        is_hetero_repr = self.fed_training_param.is_hetero_repr
        aggregation_mode = self.fed_training_param.aggregation_mode
        epoch = self.fed_training_param.epoch

        fed_input_dim = get_fed_input_dim(host_input_dim=host_input_dim,
                                          guest_input_dim=guest_input_dim,
                                          aggregation_mode=self.aggregation_mode)

        print("[INFO] #===> Build Federated Model.")

        print("[INFO] # ================ Hyperparameter Info ================")
        print("[INFO] epoch: {0}".format(epoch))
        print("[INFO] learning_rate: {0}".format(learning_rate))
        print("[INFO] weight_decay: {0}".format(weight_decay))
        print("[INFO] guest_hidden_dim: {0}".format(guest_hidden_dim))
        print("[INFO] host_hidden_dim: {0}".format(host_hidden_dim))
        print("[INFO] fed_hidden_dim: {0}".format(fed_hidden_dim))
        print("[INFO] guest_input_dim: {0}".format(guest_input_dim))
        print("[INFO] host_input_dim: {0}".format(host_input_dim))
        print("[INFO] fed_input_dim: {0}".format(fed_input_dim))
        print("[INFO] fed_reg_lambda: {0}".format(fed_reg_lambda))
        print("[INFO] guest_reg_lambda: {0}".format(guest_reg_lambda))
        print("[INFO] sharpen_temp: {0}".format(sharpen_temp))
        print("[INFO] is_hetero_repr: {0}".format(is_hetero_repr))
        print("[INFO] aggregation_mode: {0}".format(aggregation_mode))
        for key, val in loss_weight_dict.items():
            print("[INFO] {0}: {1}".format(key, val))
        print("[INFO] ========================================================")

        fed_hidden_dim = None
        guest_hidden_dim = None
        host_hidden_dim = None

        self.fed_lr = SoftmaxRegression(1).to(self.device)
        self.fed_lr.build(input_dim=fed_input_dim, output_dim=self.n_class, hidden_dim=fed_hidden_dim)
        self.fed_lr = self.fed_lr.to(self.device)

        self.guest_lr = SoftmaxRegression(2).to(self.device)
        self.guest_lr.build(input_dim=guest_input_dim, output_dim=self.n_class, hidden_dim=guest_hidden_dim)
        self.guest_lr = self.guest_lr.to(self.device)

        self.host_lr = SoftmaxRegression(3).to(self.device)
        self.host_lr.build(input_dim=host_input_dim, output_dim=self.n_class, hidden_dim=host_hidden_dim)
        self.host_lr = self.host_lr.to(self.device)

        print("[INFO] fed top model:")
        print("[INFO] fed_lr:", self.fed_lr)
        print("[INFO] guest top model:")
        print("[INFO] guest_lr:", self.guest_lr)
        print("[INFO] host top model:")
        print("[INFO] host_lr:", self.host_lr)

        self.criteria = torch.nn.CrossEntropyLoss()

        guest_model_params = list(self.guest_lr.parameters()) + self.vftl_guest.get_model_parameters()
        host_model_params = list(self.host_lr.parameters()) + self.vftl_host.get_model_parameters()
        fed_model_params = list(
            self.fed_lr.parameters()) + self.vftl_host.get_model_parameters() + self.vftl_guest.get_model_parameters()

        # optim = "sgd"
        optim = "adam"
        if optim == "adam":
            self.guest_optimizer = torch.optim.Adam(params=guest_model_params,
                                                    lr=learning_rate,
                                                    weight_decay=weight_decay)
            self.host_optimizer = torch.optim.Adam(params=host_model_params,
                                                   lr=learning_rate,
                                                   weight_decay=weight_decay)
            self.fed_optimizer = torch.optim.Adam(params=fed_model_params,
                                                  lr=learning_rate,
                                                  weight_decay=weight_decay)
        elif optim == "sgd":
            momentum = 0.9
            self.scheduler_list = []
            self.guest_optimizer = torch.optim.SGD(params=guest_model_params,
                                                   lr=learning_rate,
                                                   momentum=momentum,
                                                   weight_decay=weight_decay)
            self.host_optimizer = torch.optim.SGD(params=host_model_params,
                                                  lr=learning_rate,
                                                  momentum=momentum,
                                                  weight_decay=weight_decay)
            self.fed_optimizer = torch.optim.SGD(params=fed_model_params,
                                                 lr=learning_rate,
                                                 momentum=momentum,
                                                 weight_decay=weight_decay)
            print("[INFO] self.guest_optimizer:", self.guest_optimizer)
            print("[INFO] self.host_optimizer:", self.host_optimizer)
            print("[INFO] self.fed_optimizer:", self.fed_optimizer)
        else:
            raise Exception("Does not support {} optimizer.".format(optim))

        self.semi_sl = SemiSupervisedLearningV2(fed_predictor=self.fed_lr,
                                                guest_predictor=self.guest_lr,
                                                host_predictor=self.host_lr,
                                                fed_training_param=self.fed_training_param,
                                                n_class=self.n_class,
                                                debug=self.debug)

    def adjust_learning_rate(self):
        if self.scheduler_list is None or len(self.scheduler_list) == 0:
            return

        for scheduler in self.scheduler_list:
            scheduler.step()

    def to_train_mode(self):
        self.fed_lr.train()
        self.host_lr.train()
        self.guest_lr.train()
        self.vftl_host.to_train_mode()
        self.vftl_guest.to_train_mode()

    def to_eval_mode(self):
        self.fed_lr.eval()
        self.host_lr.eval()
        self.guest_lr.eval()
        self.vftl_host.to_eval_mode()
        self.vftl_guest.to_eval_mode()

    def _train(self,
               guest_ll_x,
               host_ll_x,
               ll_y,
               guest_ul_x,
               host_ul_x,
               ul_y,
               guest_nl_x,
               guest_nl_y,
               host_nl_x,
               host_nl_y,
               guest_all_x,
               guest_all_y,
               host_all_x,
               host_all_y
               ):

        self.to_train_mode()

        self.guest_optimizer.zero_grad()
        self.host_optimizer.zero_grad()
        self.fed_optimizer.zero_grad()

        self.vftl_guest.prepare_local_data(guest_ll_x=guest_ll_x,
                                           ll_y=ll_y,
                                           guest_ul_x=guest_ul_x,
                                           ul_y=ul_y,
                                           guest_nl_x=guest_nl_x,
                                           guest_nl_y=guest_nl_y,
                                           guest_all_x=guest_all_x,
                                           guest_all_y=guest_all_y)
        self.vftl_host.prepare_local_data(host_ll_x=host_ll_x,
                                          ll_y=ll_y,
                                          host_ul_x=host_ul_x,
                                          ul_y=ul_y,
                                          host_nl_x=host_nl_x,
                                          host_nl_y=host_nl_y,
                                          host_all_x=host_all_x,
                                          host_all_y=host_all_y)

        # ===================================================================================================
        # weights for auxiliary losses, which include:
        # (1) loss for shared representations between host and guest
        # (2) (3) loss for orthogonal representation for host and guest respectively
        # (4) loss for distance between estimated host overlap labels and true overlap labels
        # (5) loss for distance between estimated guest overlap representation and true guest representation
        # (6) loss for distance between estimated host overlap representation and true host representation
        # (7) loss for distance between shared-repr-estimated host label and uniq-repr-estimated host label
        # ===================================================================================================

        Ug_all, Ug_non_overlap, Ug_ll_overlap, Ug_ul_overlap = self.vftl_guest.fetch_feat_reprs()
        Uh_all, Uh_non_overlap, Uh_ll_overlap, Uh_ul_overlap = self.vftl_host.fetch_feat_reprs()

        # print("guest train Ug_ll_overlap_uniq:", Ug_ll_overlap_[0], Ug_ll_overlap_[0].shape)
        # print("guest train Ug_ll_overlap_comm:", Ug_ll_overlap_[1], Ug_ll_overlap_[1].shape)

        Y_ll_overlap = self.vftl_guest.get_Y_ll_overlap()
        Y_ll_for_estimation = self.vftl_guest.get_Y_ll_overlap_for_est()
        Y_nl_guest = self.vftl_guest.get_Y_nl_overlap()
        Y_nl_for_estimation = self.vftl_guest.get_Y_nl_overlap_for_est()

        input_dict = dict()
        input_dict["Ug_all"] = Ug_all
        input_dict["Ug_non_overlap"] = Ug_non_overlap
        input_dict["Ug_ll_overlap"] = Ug_ll_overlap
        input_dict["Ug_ul_overlap"] = Ug_ul_overlap

        input_dict["Uh_all"] = Uh_all
        input_dict["Uh_non_overlap"] = Uh_non_overlap
        input_dict["Uh_ll_overlap"] = Uh_ll_overlap
        input_dict["Uh_ul_overlap"] = Uh_ul_overlap

        input_dict["Y_ll_overlap"] = Y_ll_overlap
        input_dict["Y_ll_for_estimation"] = Y_ll_for_estimation
        input_dict["Y_guest_non_overlap"] = Y_nl_guest
        input_dict["Y_nl_for_estimation"] = Y_nl_for_estimation

        mv_train_data_result, assistant_loss_dict = self.semi_sl.forward(input_dict)

        fed_reprs = mv_train_data_result["train_fed_reprs"]
        guest_reprs = mv_train_data_result["train_guest_reprs"]
        host_reprs = mv_train_data_result["train_host_reprs"]
        fed_y = mv_train_data_result["train_fed_y"]
        guest_y = mv_train_data_result["train_guest_y"]
        host_y = mv_train_data_result["train_host_y"]

        guest_logits = self.guest_lr.forward(guest_reprs)
        guest_loss = self.criteria(guest_logits, torch.argmax(guest_y, dim=1))

        host_logits = self.host_lr.forward(host_reprs)
        host_loss = self.criteria(host_logits, torch.argmax(host_y, dim=1))

        fed_logits = self.fed_lr.forward(fed_reprs)
        fed_objective_loss = self.criteria(fed_logits, torch.argmax(fed_y, dim=1))

        # print("fedcvt_train_with_ll_v3_guest_logits:", guest_logits[:5, :])
        # print("fedcvt_train_with_ll_v3_host_logits:", host_logits[:5, :])
        # print("fedcvt_train_with_ll_v3_fed_logits:", fed_logits[:5, :])

        loss_weight_dict = self.fed_training_param.loss_weight_dict
        if assistant_loss_dict is not None and loss_weight_dict is not None:
            if self.debug: print("[DEBUG] append loss factors:")
            for key, loss_fac in assistant_loss_dict.items():
                loss_fac_weight = loss_weight_dict[key]
                if self.debug: print("[DEBUG] append loss factor: {}, [{}], {}".format(key, loss_fac_weight, loss_fac))
                fed_objective_loss = fed_objective_loss + loss_fac_weight * loss_fac

        all_loss = guest_loss + host_loss + fed_objective_loss

        all_loss.backward()
        # guest_loss.backward()
        # host_loss.backward()
        # fed_objective_loss.backward()

        self.guest_optimizer.step()
        self.host_optimizer.step()
        self.fed_optimizer.step()

        loss_dict = {"fed_loss": fed_objective_loss, "guest_loss": guest_loss, "host_loss": host_loss}
        return loss_dict

    def _train_conventional(self,
                            guest_ll_x,
                            host_ll_x,
                            ll_y,
                            ):

        self.to_train_mode()

        ll_y = F.one_hot(ll_y, num_classes=self.n_class)

        guest_ll_x = guest_ll_x.to(self.device)
        host_ll_x = host_ll_x.to(self.device)
        ll_y = ll_y.to(self.device)

        self.fed_optimizer.zero_grad()

        Ug_overlap_uniq, Ug_overlap_comm = self.vftl_guest.local_predict(guest_ll_x)
        Uh_overlap_uniq, Uh_overlap_comm = self.vftl_host.local_predict(host_ll_x)

        using_uniq = self.fed_training_param.using_uniq
        using_comm = self.fed_training_param.using_comm
        Ug_overlap_reprs = concat_reprs(Ug_overlap_uniq, Ug_overlap_comm, using_uniq, using_comm)
        Uh_overlap_reprs = concat_reprs(Uh_overlap_uniq, Uh_overlap_comm, using_uniq, using_comm)

        fed_ol_reprs = aggregate(guest_reprs=Ug_overlap_reprs,
                                 host_reprs=Uh_overlap_reprs,
                                 aggregation_mode=self.aggregation_mode)

        fed_logits = self.fed_lr.forward(fed_ol_reprs)
        fed_objective_loss = self.criteria(fed_logits, torch.argmax(ll_y, dim=1))

        fed_objective_loss.backward()

        self.fed_optimizer.step()

        loss_dict = {"fed_loss": fed_objective_loss, "guest_loss": None, "host_loss": None}
        return loss_dict

    def train(self,
              ll_data_loader,
              ul_data_loader,
              nl_guest_data_loader,
              nl_host_data_loader,
              all_guest_data_loader,
              all_host_data_loader,
              val_data_loader,
              test_dataloader,
              num_iteration_per_epoch,
              only_use_ll):

        valid_iteration_interval = self.fed_training_param.valid_iteration_interval
        training_info_file_name = self.fed_training_param.training_info_file_name
        with open(training_info_file_name + '.json', 'w') as outfile:
            json.dump(self.fed_training_param.get_parameters(), outfile)

        early_stopping = EarlyStoppingCheckPoint(monitor=self.fed_training_param.monitor_metric,
                                                 epoch_patience=10,
                                                 file_path=training_info_file_name)
        early_stopping.set_model(self)
        early_stopping.on_train_begin()

        with open(training_info_file_name + ".csv", "a", newline='') as logfile:
            logger = csv.DictWriter(logfile, fieldnames=list(early_stopping.get_log_info().keys()))
            logger.writeheader()

        # # load validation data
        # guest_val_block_size = self.vftl_guest.load_val_block(0)
        # host_val_block_size = self.vftl_host.load_val_block(0)
        # print("[INFO] guest_val_block_size:", guest_val_block_size)
        # print("[INFO] host_val_block_size:", host_val_block_size)

        start_time = time.time()
        epoch = self.fed_training_param.epoch

        ll_data_iterator = ForeverDataIterator(ll_data_loader)
        ul_data_iterator = None if ul_data_loader is None else ForeverDataIterator(ul_data_loader)
        nl_guest_data_iterator = None if nl_guest_data_loader is None else ForeverDataIterator(nl_guest_data_loader)
        nl_host_data_iterator = None if nl_host_data_loader is None else ForeverDataIterator(nl_host_data_loader)
        all_guest_data_iterator = None if all_guest_data_loader is None else ForeverDataIterator(all_guest_data_loader)
        all_host_data_iterator = None if all_host_data_loader is None else ForeverDataIterator(all_host_data_loader)

        num_batches_per_epoch = num_iteration_per_epoch
        loss_list = list()
        test_ll_auc_list = list()
        test_ll_acc_list = list()
        test_ll_fscore_list = list()
        for i in range(epoch):
            print("[INFO] ===> start epoch:{0}".format(i))

            iteration = 0
            for iter_idx in range(num_batches_per_epoch):

                if only_use_ll:
                    print("Supervised Learning using labeled aligned samples")
                    ll_x, ll_y = next(ll_data_iterator)
                    guest_ll_x, host_ll_x = ll_x[0], ll_x[1]
                    loss_dict = self._train_conventional(guest_ll_x=guest_ll_x,
                                                         host_ll_x=host_ll_x,
                                                         ll_y=ll_y)
                else:
                    print("Semi-supervised learning")
                    ll_x, ll_y = next(ll_data_iterator)
                    guest_ll_x, host_ll_x = ll_x[0], ll_x[1]
                    # print("train guest_ll_x:", guest_ll_x[0], guest_ll_x[0].shape)

                    if ul_data_iterator is None:
                        guest_ul_x, host_ul_x, ul_y = None, None, None
                    else:
                        ul_x, ul_y = next(ul_data_iterator)
                        guest_ul_x, host_ul_x = ul_x[0], ul_x[1]

                    guest_nl_x_tmp, guest_nl_y = next(nl_guest_data_iterator)
                    guest_nl_x = guest_nl_x_tmp[0]

                    host_nl_x_tmp, host_nl_y = next(nl_host_data_iterator)
                    host_nl_x = host_nl_x_tmp[1]

                    guest_all_x_tmp, guest_all_y = next(all_guest_data_iterator)
                    guest_all_x = guest_all_x_tmp[0]

                    host_all_x_tmp, host_all_y = next(all_host_data_iterator)
                    host_all_x = host_all_x_tmp[1]

                    # print("guest_ll_x:", guest_ll_x, guest_ll_x.shape)
                    # print("guest_ul_x:", guest_ul_x, guest_ul_x.shape)
                    # print("guest_nl_x:", guest_nl_x, guest_nl_x.shape)
                    # print("guest_all_x:", guest_all_x, guest_all_x.shape)
                    # print("host_ll_x:", host_ll_x, host_ll_x.shape)
                    # print("host_ul_x:", host_ul_x, host_ul_x.shape)
                    # print("host_nl_x:", host_nl_x, host_nl_x.shape)
                    # print("host_all_x:", host_all_x, host_all_x.shape)

                    # ==========
                    #   train
                    # ==========
                    loss_dict = self._train(guest_ll_x=guest_ll_x,
                                            host_ll_x=host_ll_x,
                                            ll_y=ll_y,
                                            guest_ul_x=guest_ul_x,
                                            host_ul_x=host_ul_x,
                                            ul_y=ul_y,
                                            guest_nl_x=guest_nl_x,
                                            guest_nl_y=guest_nl_y,
                                            host_nl_x=host_nl_x,
                                            host_nl_y=host_nl_y,
                                            guest_all_x=guest_all_x,
                                            guest_all_y=guest_all_y,
                                            host_all_x=host_all_x,
                                            host_all_y=host_all_y)

                fed_loss = loss_dict['fed_loss'].item()
                loss_list.append(fed_loss)

                print("[INFO] ==> ep:{0}, iter:{1},  fed_loss:{2}".format(i, iteration, fed_loss))

                # print("[INFO] ==> ep:{0}, iter:{1}, ll_batch_idx:{2}, ul_batch_idx:{3}, "
                #       "nol_guest_batch_idx:{4}, nol_host_batch_idx:{5}, fed_loss:{6}".format(i,
                #                                                                              iter,
                #                                                                              ll_batch_idx,
                #                                                                              ul_batch_idx,
                #                                                                              nol_guest_batch_idx,
                #                                                                              nol_host_batch_idx,
                #                                                                              fed_loss))
                # print("[INFO] ll_block_idx:{0}, nol_guest_block_idx:{1}, nol_host_block_idx:{2}, "
                #       "ested_guest_block_idx:{3}, ested_host_block_idx:{4}".format(ll_block_idx,
                #                                                                    nol_guest_block_idx,
                #                                                                    nol_host_block_idx,
                #                                                                    ested_guest_block_idx,
                #                                                                    ested_host_block_idx))

                # ==========
                # validation
                # ==========
                if (iteration + 1) % valid_iteration_interval == 0:
                    self.to_eval_mode()

                    # ====================
                    # collaborative validation
                    # ====================
                    self.collaborative_predict(data_loader=val_data_loader,
                                               data_type="train")

                    test_ll_acc, test_ll_auc, test_ll_fscore = self.collaborative_predict(data_loader=test_dataloader,
                                                                                          data_type="test")
                    test_ll_acc_list.append(test_ll_acc)
                    test_ll_auc_list.append(test_ll_auc)
                    test_ll_fscore_list.append(test_ll_fscore)

                    log = {"fscore": test_ll_fscore, "acc": test_ll_acc, "auc": test_ll_auc}

                    early_stopping.on_iteration_end(epoch=i, batch=iter_idx, log=log)
                    if self.stop_training is True:
                        break

                iteration += 1

            if self.stop_training is True:
                break

            self.adjust_learning_rate()

        end_time = time.time()
        print("[INFO] training time (s):", end_time - start_time)
        print("[INFO] stopped epoch, batch:", early_stopping.stopped_epoch, early_stopping.stopped_batch)
        early_stopping.print_log_of_best_result()
        early_stopping.save_log_of_best_result()
        print("loss_list:", loss_list)
        # series_plot(losses=loss_list, fscores=test_ll_fscore_list, aucs=test_ll_acc_list)
        return early_stopping.get_log_info(), loss_list

    def collaborative_predict(self, data_loader, data_type="test"):
        # if debug:
        print("[INFO] ===> collaborative predict")

        using_uniq = self.fed_training_param.using_uniq
        using_comm = self.fed_training_param.using_comm

        y_hat_list = []
        y_prob_id_list_v0 = []
        y_prob_id_list_v1 = []
        y_true_1d_list = []
        y_true_1hot_list = []
        y_prob_list = []
        with torch.no_grad():
            for x, y_true in data_loader:
                guest_x = x[0].to(self.device)
                host_x = x[1].to(self.device)
                # y_true = y_true.to(self.device)
                y_true = y_true.cpu()

                y_true_1hot = F.one_hot(y_true, num_classes=self.n_class).detach().numpy()
                # print("y_true_1hot:", y_true_1hot)
                y_true_1hot_list = y_true_1hot_list + [y.tolist() for y in y_true_1hot]

                Ug_overlap_uniq, Ug_overlap_comm = self.vftl_guest.local_predict(guest_x)
                Uh_overlap_uniq, Uh_overlap_comm = self.vftl_host.local_predict(host_x)

                Ug_overlap_reprs = concat_reprs(Ug_overlap_uniq, Ug_overlap_comm, using_uniq, using_comm)
                Uh_overlap_reprs = concat_reprs(Uh_overlap_uniq, Uh_overlap_comm, using_uniq, using_comm)

                fed_ol_reprs = aggregate(guest_reprs=Ug_overlap_reprs,
                                         host_reprs=Uh_overlap_reprs,
                                         aggregation_mode=self.aggregation_mode)

                pred = self.fed_lr.forward(fed_ol_reprs)
                y_prob = F.softmax(pred, dim=1)

                y_prob = y_prob.cpu().detach().numpy()
                y_prob_list = y_prob_list + [y for y in y_prob]

                y_true_1d = y_true.detach().numpy()
                y_true_1d_list = y_true_1d_list + [y_t for y_t in y_true_1d]

                y_hat_1d = convert_to_1d_labels(y_prob)
                y_hat_list = y_hat_list + [y for y in y_hat_1d]

                y_prob_1d_v1 = y_prob[:, 1]
                y_prob_1d_v0 = y_prob[:, 0]

                y_prob_id_list_v0 = y_prob_id_list_v0 + [y for y in y_prob_1d_v0]
                y_prob_id_list_v1 = y_prob_id_list_v1 + [y for y in y_prob_1d_v1]

        acc = accuracy_score(y_true_1d_list, y_hat_list)
        if self.n_class > 2:
            auc = 0.0
            fscore_v0 = 0.0
            fscore_v1 = 0.0
            fscore_sciki = 0.0
        else:
            fscore_v0 = f_score_v2(y_true_1d_list, y_prob_id_list_v0)
            fscore_v1 = f_score_v2(y_true_1d_list, y_prob_id_list_v1)
            # print("y_true list:", y_true_1d_list)
            # print("y_hat  list:", y_hat_list)
            fscore_sciki = f1_score(y_true_1d_list, y_hat_list)
            auc = roc_auc_score(y_true_1hot_list, y_prob_list)

        print("[INFO] {} - fscore_v0:{}, fscore_v1:{}, fscore_sciki:{}, auc:{}, acc:{}".format(data_type,
                                                                                               fscore_v0,
                                                                                               fscore_v1,
                                                                                               fscore_sciki,
                                                                                               auc,
                                                                                               acc))

        return acc, auc, fscore_v1
