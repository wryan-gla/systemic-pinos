import tensorflow as tf


class AdaptiveLossWeights:
    """Gradient-norm balancing of loss terms.

    Every so often, each weight is relaxed towards
    (sum of all terms' gradient norms) / (this term's gradient norm),
    so terms with small gradients are up-weighted.
    """

    def __init__(self, keys, decay=0.9):
        self.weights = {k: 1. for k in keys}
        self.decay = decay

    def update(self, grad_norms):
        total = sum(grad_norms.values())
        for k, norm in grad_norms.items():
            if norm > 0:
                self.weights[k] = self.decay * self.weights[k] + (1 - self.decay) * total / norm

    def as_tensors(self):
        return {k: tf.constant(w, tf.float32) for k, w in self.weights.items()}
