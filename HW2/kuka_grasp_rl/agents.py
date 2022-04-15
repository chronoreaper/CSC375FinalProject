import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class ActorCriticAgent():

    def __init__(self, model):
        super(ActorCriticAgent, self).__init__()
        self.model = model
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    def act(self, state):
        action_logits_t, value = self.model(np.array([state]))
        action_dist = tfp.distributions.Categorical(logits=action_logits_t)
        action_t = action_dist.sample()
        action_probs_t = tf.nn.softmax(action_logits_t)
        action_log_probs_t = tf.math.log(action_probs_t)
        return int(action_t[0]), action_log_probs_t, value

    def compute_loss(self, log_probs, returns, values):
        actor_loss = -tf.math.reduce_sum(log_probs * (np.array(returns) - np.array(values)))
        critic_loss = self.huber_loss(values, returns)
        return actor_loss + critic_loss

    def compute_expected_return(self, rewards, gamma):
        eps = np.finfo(np.float32).eps.item()

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape

        for i in tf.range(n):
          reward = rewards[i]
          discounted_sum = reward + gamma * discounted_sum
          discounted_sum.set_shape(discounted_sum_shape)
          returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + eps))

        return returns
