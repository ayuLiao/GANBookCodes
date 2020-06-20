
import os
from core.data import DataLoader
from core.model import DomainTransferNet
from utils import log_utils, os_utils, time_utils


SVHN_DIR = "./data/svhn"
MNIST_DIR = "./data/mnist"
SAMPLE_DIR = "./sample"


params = {
    "log_dir": "./log",
    "model_dir": "./model",
    "summary_dir": "./summary",


    "learning_rate": 0.001,
    "max_batch": 2000,
    "max_batch_pretrain": 20000,
    "batch_size": 100,
    "eval_every_num_update_pretrain": 100,
    "eval_every_num_update": 10,


    "loss_const_weight": 15.,
    "loss_tid_weight": 15.,
    "loss_tv_weight": 0.,


    # balance the training of discriminator and generator
    "d_update_freq_source": 1,
    "g_update_freq_source": 6,
    "d_update_freq_target": 2,
    "g_update_freq_target": 4,


    "flip_gradient": False,
    "f_adaptation": False,

}


def main():

    # source domain
    print("load svhn")
    svhn_images_train, _ = DataLoader.load_svhn(SVHN_DIR, "train_32x32.mat")
    svhn_images_test, svhn_labels_test = DataLoader.load_svhn(SVHN_DIR, "test_32x32.mat")
    svhn_images_extra, svhn_labels_extra = DataLoader.load_svhn(SVHN_DIR, "extra_32x32.mat")

    auxiliary_data = {
        "X_train": svhn_images_extra,
        "y_train": svhn_labels_extra,
        "X_test": svhn_images_test,
        "y_test": svhn_labels_test,
    }

    # target domain
    print("load mnist")
    if not os.path.isfile(os.path.join(MNIST_DIR, "train.pkl")):
        DataLoader.prepare_mnist(MNIST_DIR, "train")
    mnist_images_train, _ = DataLoader.load_mnist(MNIST_DIR, "train")

    # dtn model
    print("init dtn")
    os_utils._makedirs(params["summary_dir"], force=True)
    os_utils._makedirs(params["log_dir"])
    logger = log_utils._get_logger(params["log_dir"], "tf-%s.log" % time_utils._timestamp())
    model = DomainTransferNet(params, logger)
    #拟合，即训练的主函数
    print("fit dtn")
    model.fit(auxiliary_data, Xs_train=svhn_images_train, Xt_train=mnist_images_train)

    print("evaluate dtn")
    model.evaluate(Xs=svhn_images_train, sample_batch=100, batch_size=100, sample_dir=SAMPLE_DIR)

        
if __name__ == '__main__':
    main()
