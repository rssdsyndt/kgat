import tensorflow as _tf

tf = _tf.compat.v1
tf.disable_v2_behavior()


def xavier_initializer(uniform=True):
    if uniform:
        return _tf.keras.initializers.GlorotUniform()
    return _tf.keras.initializers.GlorotNormal()
