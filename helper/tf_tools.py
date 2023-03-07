import numpy as np
import tensorflow as tf

def LCM_CAM_tf2(last_conv_layer_model, classifier_model, dense_weight, img, threshold = 0.5):
    """
    Perform Land Cover Mapping from Class Activation Map.

    Parameters
    ----------
    last_conv_layer_model: tensorflow-keras Model
        Model [Input: Input Layer -> Output: Last conv layer].
    classifier_model: tensorflow-keras Model
        Model [Input: Last conv layer -> Output: classifier layer(last)].
    dense_weight: tensor variable
        Weights of the classifier layer (last layer)
    img: numpy array
        Input image for Land Cover Mapping
    threshold: float
        Classification's threshold

    Examples
    --------
    >>> LCM_CAM_tf2(last_conv_layer_model, classifier_model, dense_weight, img=x_test[i])

    Returns
    -------
    Output image from Land Cover Mapping process
    """

    last_conv_layer_output = last_conv_layer_model(np.expand_dims(img, axis=0))

    preds = classifier_model(last_conv_layer_output)[0]

    preds_thresh = tf.cond(
        tf.reduce_any(tf.math.greater_equal(preds, threshold)),
        lambda: tf.math.greater_equal(preds, threshold),
        lambda: tf.one_hot(tf.argmax(preds), len(preds), on_value=True, off_value=False, dtype=tf.bool)
    )

    preds_class_index = tf.reshape(tf.where(preds_thresh), shape=[-1]).numpy()
    preds_class_index = {idx: item for idx, item in enumerate(preds_class_index)}

    dense_weight_preds = tf.boolean_mask(dense_weight, preds_thresh, axis=1)

    last_conv_layer_output = tf.reshape(last_conv_layer_output, shape=[-1, last_conv_layer_output.shape[-1]])

    CAM = tf.matmul(last_conv_layer_output, dense_weight_preds)
    LCM = tf.argmax(CAM, axis=-1)

    LCM = LCM.numpy()
    LCM = np.vectorize(lambda val: preds_class_index[val])(LCM).reshape(img.shape[:2])

    return LCM