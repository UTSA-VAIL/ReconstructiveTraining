import inspect
from util import loss_functions
import tensorflow as tf


class CombinedLoss:
    def __init__(self, loss_classes, loss_weights, kwargs):
        if loss_weights is None:
            self.loss_weights = [1.0] * len(loss_classes)
        else:
            self.loss_weights = loss_weights

        self.loss_functions = []
        for loss in loss_classes:
            loss_func_partial = loss(kwargs)
            self.loss_functions.append(loss_func_partial)

    def __call__(self, y_pred, y_true):
        loss_outputs = []
        for (loss, weight) in zip(self.loss_functions, self.loss_weights):
            partial_output = tf.math.multiply(loss(y_pred, y_true), weight)
            loss_outputs.append(partial_output)

        output_stack = tf.stack(loss_outputs)

        total_loss = tf.math.reduce_sum(output_stack, axis=0)

        return total_loss


class Loss:
    def __get_loss_function(self, loss_names, loss_weights, kwargs):

        self.__name__ = "_".join(loss_names)

        # Get the list of all available loss functions
        valid_loss_functions = dict(inspect.getmembers(loss_functions, inspect.isclass))

        # Find the loss functions we asked for
        loss_classes = []
        for loss_function_name in loss_names:
            try:
                loss_classes.append(valid_loss_functions[loss_function_name])
            except Exception as e:
                print(e)
                raise ValueError("Loss functions must be one of " + str(list(valid_loss_functions.keys())) + " found '" + loss_function_name + "'")

        # Check if we need to do a combined loss
        if len(loss_classes) > 1:
            loss_function = CombinedLoss(loss_classes, loss_weights, kwargs)
        else:
            # Just get the loss function if there is only one
            loss_function = loss_classes[0](kwargs)

        return loss_function

    def __init__(self, loss_names, loss_weights=None, **kwargs):
        self.loss_function = self.__get_loss_function(loss_names=loss_names, loss_weights=loss_weights, kwargs=kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return self.loss_function(y_true, y_pred)

    def asfunction(self):
        return self.loss_function
