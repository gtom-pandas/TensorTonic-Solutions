import math

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    if len(prob_distributions) == 0:
        return 0.0
    
    log_prob_sum = 0.0
    for dist, token_idx in zip(prob_distributions, actual_tokens):
        prob = dist[token_idx]
        log_prob_sum += math.log(prob)
    
    cross_entropy = -log_prob_sum / len(prob_distributions)
    perplexity_score = math.exp(cross_entropy)
    
    return perplexity_score