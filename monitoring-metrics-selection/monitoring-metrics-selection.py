import numpy as np

def compute_monitoring_metrics_classification(y_true, y_pred):
    """
    Computes Accuracy, Precision, Recall, and F1-score for binary classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate accuracy
    n = len(y_true)
    accuracy = (tp + tn) / n if n > 0 else 0
    
    # Calculate precision and recall with zero-division protection
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1-score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return [("accuracy", accuracy), ("f1", f1), ("precision", precision), ("recall", recall)]

def compute_monitoring_metrics_regression(y_true, y_pred):
    """
    Computes Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    return [("mae", mae), ("rmse", rmse)]

def compute_monitoring_metrics_ranking(y_true, y_pred, k=3):
    """
    Computes Precision@k and Recall@k.
    - y_true: Binary relevance labels (e.g., [0, 1, 0, 1])
    - y_pred: Model scores or probabilities (e.g., [0.1, 0.8, 0.2, 0.9])
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 1. Get indices that would sort y_pred in descending order (highest score first)
    # np.argsort returns indices for ascending order, so we flip it with [::-1]
    sorted_indices = np.argsort(y_pred)[::-1]
    
    # 2. Select the top k indices
    top_k_indices = sorted_indices[:k]
    
    # 3. Use those integer indices to see which of the top k were actually relevant
    # This is where the "integer type" error is resolved
    relevant_retrieved = np.sum(y_true[top_k_indices])
    
    # Precision@k: (Relevant items in top k) / k
    precision_at_k = relevant_retrieved / k
    
    # Recall@k: (Relevant items in top k) / (Total relevant items in the whole set)
    total_relevant = np.sum(y_true)
    recall_at_k = relevant_retrieved / total_relevant if total_relevant > 0 else 0
    
    return [(f"precision_at_{k}", precision_at_k), (f"recall_at_{k}", recall_at_k)]
    
def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    match system_type:
        case "classification":
            return compute_monitoring_metrics_classification(y_true, y_pred)
        case "regression":
            return compute_monitoring_metrics_regression(y_true, y_pred)        
        case "ranking":
            return compute_monitoring_metrics_ranking(y_true, y_pred)
        case _:
            raise ValueError(f"Unknown system type: {system_type}")