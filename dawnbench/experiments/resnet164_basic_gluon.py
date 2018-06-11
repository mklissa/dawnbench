import mxnet as mx
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arg_parsing import process_args
from logger import construct_run_id, configure_root_logger
from data_loaders.cifar10 import Cifar10
from models.resnet18_basic import resnet18Basic
from learners.gluon import GluonLearner
import random
import numpy as np
import pdb

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    random.seed(0)
    np.random.seed(0)
    mx.random.seed(777)
    

    hosts=['algo0']
    num_gpus= len(mx.test_utils.list_gpus())
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync'


    batch_size = 128
    train_data, valid_data = Cifar10(batch_size=batch_size,
                                     data_shape=(3, 32, 32),
                                     padding=4,
                                     padding_value=0,
                                     normalization_type="channel").return_dataloaders()

    mult = 1.
    lr_schedule = {0: 0.01*mult, 5: 0.1*mult, 95: 0.01*mult, 105: 0.001*mult}
    lr=0.1*mult
    
    dtype='float32'
    model = resnet18Basic(num_classes=10)

    learner = GluonLearner(model, hybridize=False, ctx=[mx.gpu(0)])
    learner.fit(train_data=train_data,
                valid_data=valid_data,
                epochs=30,
                lr=lr,
#                 lr_schedule=lr_schedule,
                initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2),
                optimizer=mx.optimizer.NAG(learning_rate=lr_schedule[0], rescale_grad=1.0/batch_size, momentum=0.9, wd=0.0005),
                early_stopping_criteria=lambda e: e >= 0.94,
                kvstore=kvstore,
                dtype=dtype,) 

    
#     _, test_data = Cifar10(batch_size=1, data_shape=(3, 32, 32),
#                            normalization_type="channel").return_dataloaders()
#     learner.predict(test_data=test_data, log_frequency=100)