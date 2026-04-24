import numpy as np

class GAN:
    def __init__(self, G_W, D_W):
        """
        Initialize GAN with concrete weights.
        """
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)
    
    def generate(self, z):
        """
        Generate fake samples from noise z using tanh(z @ G_W).
        Returns list of lists, rounded to 4 decimals.
        """
        out = np.tanh(np.dot(z, self.G_W))
        return np.round(out, 4).tolist()
    
    def discriminate(self, x):
        """
        Classify samples using sigmoid(x @ D_W).
        Returns list of lists, rounded to 4 decimals.
        """
        logits = np.dot(x, self.D_W)
        out = 1 / (1 + np.exp(-logits))
        return np.round(out, 4).tolist()
    
    def train_step(self, real_data, z):
        """
        Compute d_loss and g_loss for one training step.
        Returns dict with "d_loss" and "g_loss", rounded to 4 decimals.
        """
        eps = 1e-8

        fake_data = np.tanh(np.dot(z, self.G_W))

        real_probs = 1 / (1 + np.exp(-np.dot(real_data, self.D_W)))
        fake_probs = 1 / (1 + np.exp(-np.dot(fake_data, self.D_W)))

        real_probs = np.clip(real_probs, eps, 1 - eps)
        fake_probs = np.clip(fake_probs, eps, 1 - eps)

        d_loss = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))
        g_loss = -np.mean(np.log(fake_probs))

        return {
            "d_loss": round(float(d_loss), 4),
            "g_loss": round(float(g_loss), 4),
        }