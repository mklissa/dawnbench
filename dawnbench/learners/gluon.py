import logging
import mxnet as mx
import time
import datetime
import os


class GluonLearner():
    def __init__(self, model, hybridize=False, tensorboard_logging=False, ctx=[mx.cpu()]):
        """

        Parameters
        ----------
        model: HybridBlock
        gpu_idxs: None or list of ints
            If None will set context to CPU.
            If list of ints, will set context to given GPUs.
        """
        logging.info("Using Gluon Learner.")
        self.model = model

        
        if hybridize:
            self.model.hybridize()
            logging.info("Hybridized model.")
            
#         self.context = get_context(gpu_idxs)
        self.context = ctx
    

    
    def fit(self, train_data, valid_data,
            epochs=300,
            lr=None, lr_schedule=None,
            initializer=mx.init.Xavier(),
            optimizer=None,
            kvstore='device',
            log_frequency=10000,
            early_stopping_criteria=None,
            dtype='float32'
        ):

        def linear_cycle(lr_initial=0.1, epochs=10, low_lr=0.005, extra=5, **kwargs):
            def f(progress):
                if progress < epochs / 2:
                    return 2 * lr_initial * (1 - float(epochs - progress) / epochs)
                elif progress <= epochs:
                    return low_lr + 2 * lr_initial * float(epochs - progress) / epochs
                elif progress <= epochs + extra:
                    return low_lr * float(extra - (progress - epochs)) / extra
                else:
                    return low_lr / 10

            return f        

        lr_scheduler = linear_cycle(epochs=epochs, low_lr=0.005, extra=7)
        
        
        
        if lr_schedule is None:
            assert lr is not None, "lr must be defined if not using lr_schedule"
            lr_schedule = {0: lr}
        else:
            assert lr is None, "lr should not be defined if using lr_schedule"
            assert 0 in lr_schedule.keys(), "lr for epoch 0 must be defined in lr_schedule"

        self.model.initialize(initializer, ctx=self.context)

        trainer = mx.gluon.Trainer(params=self.model.collect_params(), optimizer=optimizer, kvstore=kvstore)
        train_metric = mx.metric.Accuracy()
        criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        max_val_acc = {'val_acc': 0, 'trn_acc': 0, 'epoch': 0}

        for epoch in range(epochs+5+5):
            epoch_tick = time.time()


            logging.info('Epoch {}, Learning rate={}'.format(epoch, trainer.learning_rate))


            train_metric.reset()
            samples_processed = 0
            for batch_idx, (data, label) in enumerate(train_data):
                batch_tick = time.time()
                batch_size = data.shape[0]
                
                new_lr = lr_scheduler(epoch + batch_idx / int(50000/batch_size))
                trainer.set_learning_rate(new_lr)
                
                # partition data across all devices in context
                data = mx.gluon.utils.split_and_load(data.astype(dtype), ctx_list=self.context, batch_axis=0)
                label = mx.gluon.utils.split_and_load(label, ctx_list=self.context, batch_axis=0)

                y_pred = []
                losses = []
                with mx.autograd.record():
                    # calculate loss on each partition of data
                    for x_part, y_true_part in zip(data, label):
#                         print(len(x_part))
                        y_pred_part = self.model(x_part)
                        loss = criterion(y_pred_part, y_true_part)
                        if epoch ==0 and batch_idx ==0:
                            experiment_start=time.time()
                        # store the losses and do backward after we have done forward on all GPUs.
                        # for better performance on multiple GPUs.
                        losses.append(loss)
                        y_pred.append(y_pred_part)
                    for loss in losses:
                        loss.backward()
                trainer.step(batch_size)
                train_metric.update(label, y_pred)



                # log batch speed (if a multiple of log_frequency is contained in the last batch)
#                 log_batch = (samples_processed // log_frequency) != ((samples_processed + batch_size) // log_frequency)
#                 if ((batch_idx >= 1) and log_batch):
#                     # batch estimate, not averaged over multiple batches
#                     speed = batch_size / (time.time() - batch_tick)
#                     logging.info('Epoch {}, Batch {}, Speed={:.2f} images/second'.format(epoch, batch_idx, speed))
                samples_processed += batch_size

            # log training accuracy
            _, trn_acc = train_metric.get()
#             logging.info('Epoch {}, Training accuracy={}'.format(epoch, trn_acc))


            # log validation accuracy
            val_acc = evaluate_accuracy(valid_data, self.model, ctx=self.context)
            logging.info('Epoch {}, Training accuracy={}, Validation accuracy={}'.format(epoch, trn_acc, val_acc))

            # log maximum validation accuracy
            if val_acc > max_val_acc['val_acc']:
                max_val_acc = {'val_acc': val_acc, 'trn_acc': trn_acc, 'epoch': epoch}
            logging.info(("(Max val={}, Max train={}, Time= {})").format(max_val_acc['val_acc'],
                                                                        max_val_acc['trn_acc'],
                                                                        time.time()-experiment_start))

            

#             logging.info('Epoch {}, Duration={}'.format(epoch, time.time() - epoch_tick))

            
            if early_stopping_criteria:
                if early_stopping_criteria(val_acc):
                    logging.info("Epoch {}, Reached early stopping target, stopping training.".format(epoch))
                    break



class WaitOnReadAccuracy():
    def __init__(self, ctx):
        if isinstance(ctx, list):
            self.ctx = ctx[0]
        else:
            self.ctx = ctx
        self.metric = mx.nd.zeros(1, self.ctx)
        self.num_instance = mx.nd.zeros(1, self.ctx)

    def reset(self):
        self.metric = mx.nd.zeros(1, self.ctx)
        self.num_instance = mx.nd.zeros(1, self.ctx)

    def get(self):
        return float(self.metric.asscalar()) / float(self.num_instance.asscalar())

    def update(self, label, pred):
        # for single context
        if isinstance(label, mx.nd.NDArray) and isinstance(pred, mx.nd.NDArray):
            pred = mx.nd.argmax(pred, axis=1)
            self.metric += (pred == label).sum()
            self.num_instance += label.shape[0]
        # for multi-context where data is partitioned
        elif isinstance(label, list) and isinstance(pred, list):
            for label_part, pred_part in zip(label, pred):
                pred_part = mx.nd.argmax(pred_part, axis=1)
                self.metric += (pred_part == label_part).sum()
                self.num_instance += label_part.shape[0]
        else:
            raise TypeError


def evaluate_accuracy(valid_data, model, ctx):
#     print("eval context is :{}".format(ctx))
    if isinstance(ctx, list):
        ctx = ctx[0]
    accuracy = WaitOnReadAccuracy(ctx)
    for batch_idx, (data, label) in enumerate(valid_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = model(data)
        accuracy.update(label, output)
    return accuracy.get()


