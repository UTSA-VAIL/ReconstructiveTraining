import tensorflow as tf


class psnr:
    def __init__(self, kwargs):
        if 'psnr_max_value' in kwargs:
            self.max_val = kwargs['psnr_max_value']
        else:
            self.max_val = 1.0

    def __call__(self, y_true, y_pred):
        psnr = tf.image.psnr(y_true, y_pred, max_val=self.max_val)  # Returns: shape [batch_size, 1].
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        psnr_loss = tf.reduce_sum(psnr) * (1.0/batch_size)
        psnr_loss = tf.math.subtract(100.0, psnr_loss)

        return psnr_loss


class mse:
    def __init__(self, kwargs):
        if 'mse_reduction' in kwargs:
            self.reduction = kwargs['mse_reduction']
        else:
            self.reduction = tf.keras.losses.Reduction.NONE

    def __call__(self, y_true, y_pred):
        keras_mse = tf.keras.losses.MeanSquaredError(reduction=self.reduction)
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        mse_loss = tf.reduce_sum(mse_loss) * (1.0 / batch_size)
        return mse_loss


class rmse:
    def __init__(self, kwargs):
        if 'rmse_reduction' in kwargs:
            self.reduction = kwargs['rmse_reduction']
        else:
            self.reduction = tf.keras.losses.Reduction.NONE

    def __call__(self, y_true, y_pred):
        keras_mse = tf.keras.losses.MeanSquaredError(reduction=self.reduction)
        mse_loss = keras_mse(y_true, y_pred)
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        rmse_loss = tf.math.sqrt(mse_loss)
        rmse_loss = tf.reduce_sum(rmse_loss) * (1.0 / batch_size)
        return rmse_loss


class mae:
    def __init__(self):
        pass  # no kwargs implemented

    def __call__(self, y_true, y_pred):
        keras_mae = tf.keras.losses.MeanAbsoluteError()
        mae_loss = keras_mae(y_true, y_pred)
        return mae_loss


class kld:
    def __init__(self):
        pass  # no kwargs implemented

    def __call__(self, y_true, y_pred):
        y_true_softmax = tf.keras.activations.softmax(y_true)
        y_pred_softmax = tf.keras.activations.softmax(y_pred)
        keras_kld = tf.keras.losses.KLDivergence()
        kld_loss = keras_kld(y_true_softmax, y_pred_softmax)
        return kld_loss

class cc:
    def __init__(self, kwargs):
        if 'cc_reduction' in kwargs:
            self.reduction = kwargs['cc_reduction']
        else:
            self.reduction = tf.keras.losses.Reduction.NONE

    def __call__(self, y_true, y_pred):
        keras_cc = tf.keras.losses.CategoricalCrossentropy(reduction=self.reduction)
        cc_loss = keras_cc(y_true, y_pred)
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        cc_loss = tf.reduce_sum(cc_loss) * (1.0 / batch_size)
        return cc_loss

class l2dist:
    def __init__(self, kwargs):
        if 'l2dist_reduction' in kwargs:
            self.reduction = kwargs['l2dist_reduction']
        else:
            self.reduction = tf.keras.losses.Reduction.NONE

    def __call__(self, y_true, y_pred):
        l2dist_loss = tf.norm(y_true - y_pred, ord="euclidean")
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        l2dist_loss = tf.reduce_sum(l2dist_loss) * (1.0 / batch_size)
        return l2dist_loss