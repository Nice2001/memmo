import tensorflow as tf
import keras

# Ta loss custom (copie-la depuis ton app)
@keras.utils.register_keras_serializable(package="Custom")
def ordinal_loss(y_true, y_pred):
    weights = tf.constant([[0.0,1.0,2.0],[1.0,0.0,1.0],[2.0,1.0,0.0]], dtype=tf.float32)
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.1)
    true_class = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
    pred_class = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
    indices = tf.stack([true_class, pred_class], axis=1)
    penalty = tf.gather_nd(weights, indices)
    return ce + 0.5 * tf.cast(penalty, tf.float32)

# Charger en mode tolérant
model = tf.keras.models.load_model(
    "breast_density_cnn_final.keras",
    custom_objects={"ordinal_loss": ordinal_loss},
    compile=False,
    safe_mode=False
)

# Réenregistrer en .h5 (format plus universel)
model.save("breast_density_cnn.h5")
print("OK : modèle converti en breast_density_cnn.h5")