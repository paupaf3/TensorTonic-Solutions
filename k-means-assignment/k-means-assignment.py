def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    def squared_distance(p1, p2):
        return sum((a - b) ** 2 for a, b in zip(p1, p2))

    assignments = []
    for point in points:
        # Find the index of the centroid with the minimum squared distance
        nearest_idx = min(
            range(len(centroids)), 
            key=lambda i: squared_distance(point, centroids[i])
        )
        assignments.append(nearest_idx)

    return assignments