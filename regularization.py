import numpy as np
import pprint
from collections import OrderedDict

# We create a simple class, called EarlyStoppingCheckPoint, that combines simplified version of Early Stoping
# and ModelCheckPoint classes from Keras.
# References:
# https://keras.io/callbacks/
# https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L458
# https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L358


class EarlyStoppingCheckPoint(object):

    def __init__(self, monitor="acc", patience=5, iter_patience=500, file_path=None):
        self.model = None
        self.monitor = monitor
        self.epoch_patience = patience
        self.iter_patience = iter_patience
        self.file_path = file_path
        self.wait_iter = 0
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.stopped_batch = 0
        self.best = -np.Inf
        self.best_log = None
        self.best_list = []

    def set_model(self, model):
        self.model = model

    def on_train_begin(self):
        self.wait_iter = 0
        self.stopped_epoch = 0
        self.stopped_batch = 0
        self.best = -np.Inf

    def on_iteration_end(self, epoch, batch, log=None):

        current = log.get(self.monitor)
        if current is None:
            print('monitor does not available in logs')
            return

        if current > self.best:
            # current is so far the best and record everything necessary
            self.best = current
            self.best_list.append(current)
            self.best_log = log
            self.best_log["epoch"] = epoch
            self.best_log["batch"] = batch
            print("find best {0}: {1} at epoch {2}, batch {3}".format(self.monitor, self.best, epoch, batch))
            if self.file_path is not None:
                print("save model to {0}".format(self.file_path))
                self.model.save_model(self.file_path)
            self.wait_iter = 0
            self.best_epoch = epoch
        else:

            wait_epoch = epoch - self.best_epoch
            if wait_epoch >= self.epoch_patience:
                self.stopped_epoch = epoch
                self.stopped_batch = batch
                self.model.stop_training = True

            self.wait_iter += 1
            if self.wait_iter >= self.iter_patience:
                self.stopped_epoch = epoch
                self.stopped_batch = batch
                self.model.stop_training = True

            print("{0} is not the best {1}, which is {2}".format(current, self.monitor, self.best))
            print("current best info:", self.best_log)
            print("current wait iteration count is {0}".format(self.wait_iter))
            print("current wait epoch count is {0}".format(wait_epoch))

    def print_log_of_best_result(self):
        pp = pprint.PrettyPrinter(indent=4)
        print("log of best result:")
        pp.pprint("log: {0}".format(self.best_log))
        pp.pprint("best {0} list: {1} ".format(self.monitor, self.best_list))

    def get_log_info(self):
        log_info = OrderedDict()
        self._add_log_info("fscore", log_info, self.best_log)
        self._add_log_info("all_fscore", log_info, self.best_log)
        self._add_log_info("g_fscore", log_info, self.best_log)
        self._add_log_info("h_fscore", log_info, self.best_log)
        self._add_log_info("all_acc", log_info, self.best_log)
        self._add_log_info("g_acc", log_info, self.best_log)
        self._add_log_info("h_acc", log_info, self.best_log)
        self._add_log_info("all_auc", log_info, self.best_log)
        self._add_log_info("g_auc", log_info, self.best_log)
        self._add_log_info("h_auc", log_info, self.best_log)
        self._add_log_info("batch", log_info, self.best_log)
        self._add_log_info("epoch", log_info, self.best_log)
        return log_info

    @staticmethod
    def _add_log_info(info_name, log_info, original_info):
        value = original_info.get(info_name)
        log_info[info_name] = -1 if value is None else value
