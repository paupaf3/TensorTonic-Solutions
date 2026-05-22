import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    # Calculate the expected values (means) of the scores
    expected_real = np.mean(real_scores)
    expected_fake = np.mean(fake_scores)
    
    # Return the negative Wasserstein distance
    return expected_fake - expected_real