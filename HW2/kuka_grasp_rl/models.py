import tensorflow as tf

class ActorCriticPolicy(tf.keras.layers.Layer):
    def __init__(self, input_shape, action_dim):
        super(ActorCriticPolicy, self).__init__()
        self.transform = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten()
            ]
        )

        self.action_head = tf.keras.layers.Dense(action_dim)
        self.value_head = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.transform(x)
        action = self.action_head(x)
        value_states = self.value_head(x)

        return action, value_states
