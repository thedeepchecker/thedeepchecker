import tensorflow as tf
import tfun.global_config as global_cfg

class loss(object):
    """
    This class work by adding loss values (computed from the net + target values) into the loss collection, named [lkey]
    Before running actual computation, it will collect all added losses from the collection and add them together
    """
    def __init__(self): 
        self.use_tboard = global_cfg.use_tboard
        self.lkey = global_cfg.lkey 
        self.reg_key = 'reg'

    def softmax_log_loss(self, X, target, target_weight=None, lm=1):
        xdev = X - tf.reduce_max(X, keep_dims=True, reduction_indices=[-1])
        lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), keep_dims=True, reduction_indices=[-1]))
        #lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), keep_dims=True, reduction_indices=[-1]))
        if (target_weight == None):
            target_weight=1
        l = -tf.reduce_mean(target_weight*target*lsm, name='softmax_log_loss')
        #l = -tf.reduce_sum(target_weight*target*lsm, name='softmax_log_loss')/tf.cast(tf.shape(X)[0], dtype=global_cfg.dtype)
        tf.add_to_collection(self.lkey, l)
        if (self.use_tboard):
            tf.summary.scalar('softmax_log_loss', l)
        return l

    def l2_loss(self, wkey, lm):
        all_var = tf.trainable_variables()
        for var in all_var:
            if (wkey in var.op.name):
                l = tf.multiply(tf.nn.l2_loss(var), lm, name='weight_loss')
                tf.add_to_collection(self.lkey, l)
                tf.add_to_collection(self.reg_key, l)
                if self.use_tboard:
                    tf.summary.scalar(var.op.name + '/weight_loss', l)
        return tf.add_n(tf.get_collection(self.reg_key), name='reg_loss')

    def total_loss(self):
        l = tf.add_n(tf.get_collection(self.lkey), name='total_loss')
        if self.use_tboard:
            tf.summary.scalar('total_loss', l)
        return l
