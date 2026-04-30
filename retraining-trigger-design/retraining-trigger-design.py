def retraining_policy(daily_stats, config):
    """
    Decide which days to trigger model retraining.
    """

    days_since_retrain = config['cooldown'] - 1
    retrain_days = []
    # Iterate each day stats
    for day in daily_stats:
        days_since_retrain += 1
        
        drift_trigger =  day['drift_score'] > config['drift_threshold']
        performance_trigger = day['performance'] < config['performance_threshold']
        staleness_trigger = days_since_retrain >= config['max_staleness']
        enough_cooldown = days_since_retrain >= config['cooldown']
        enough_budget = config['retrain_cost'] <= config['budget']

        if (drift_trigger or performance_trigger or staleness_trigger) and enough_cooldown and enough_budget:
            days_since_retrain = 0
            config['budget'] -= config['retrain_cost']
            retrain_days.append(day['day'])            

    return retrain_days