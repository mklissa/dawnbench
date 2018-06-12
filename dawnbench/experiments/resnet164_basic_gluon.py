import mxnet as mx
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arg_parsing import process_args
from logger import construct_run_id, configure_root_logger
from data_loaders.cifar10 import Cifar10
from models.resnets_basic import resnet18Basic, wrn
from models.wrn_basic import wrn
from learners.gluon import GluonLearner
import random
import numpy as np
import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=140)
    parser.add_argument('--lr-schedule', type=str, default='step')
    parser.add_argument('--model', type=str, default='wrn')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=789)
    return parser.parse_args()

args = parse_args()

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    

    hosts=['algo0']
    num_gpus= len(mx.test_utils.list_gpus())
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync'



    train_data, valid_data = Cifar10(batch_size=args.batch_size,
                                     data_shape=(3, 32, 32),
                                     padding=4,
                                     padding_value=0,
                                     normalization_type="channel").return_dataloaders()

    dtype='float32'
    mult = 1.

    if args.lr_schedule=='step':
        args.lr_schedule = {0: 0.01*mult, 5: 0.1*mult, 95: 0.01*mult, 105: 0.001*mult}

    
    if args.model =='wrn':
        model = wrn(num_classes=10)
    elif args.model =='resnet18':
        model = resnet18Basic(num_classes=10)        
    else:
        logging.error("Model not currently supported.")
        sys.exit(0)
        
    learner = GluonLearner(model, hybridize=False, ctx=[mx.gpu(0)])
    learner.fit(train_data=train_data,
                valid_data=valid_data,
                epochs=args.epochs,
                lr_schedule=args.lr_schedule,
                initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2),
                optimizer=mx.optimizer.NAG(learning_rate=0.1, rescale_grad=1.0/batch_size, momentum=0.9, wd=0.0005),
                early_stopping_criteria=lambda e: e >= 0.94,
                kvstore=kvstore,
                dtype=dtype,) 

    
#     _, test_data = Cifar10(batch_size=1, data_shape=(3, 32, 32),
#                            normalization_type="channel").return_dataloaders()
#     learner.predict(test_data=test_data, log_frequency=100)