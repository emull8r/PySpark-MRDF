"""Pyspark implementation of KNN"""
from utilities import measure_euclidean

def brute_force_knn(sparkcontext, rdd, k):
    """Implement brute-force KNN. WARNING expensive, only do on cluster."""
    # turn into broadcasted read-only dictionary of (id, vector) to make comparison easier
    v_dict = sparkcontext.broadcast(rdd.collectAsMap())
    # initialize G to be of the form (vector id, empty list)
    knn = rdd.map(lambda v: knn_map(v, v_dict, k)).sortByKey(ascending=True)
    return knn
    
def knn_map(v, v_dict, k):
    """Do KNN. WARNING: Expensive, should only do on cluster."""
    nns = []
    for node_id, vector in v_dict.value.items():
        if node_id != v[0]: # don't add ourselves
            distance = measure_euclidean(v[1], vector)
            neighbor = (distance, node_id)
            nns.append(neighbor)
            nns.sort()
            if len(nns) > k:
                nns = nns[:-1]
    neighbor_ids = []
    for distance, node_id in nns:
        neighbor_ids.append(node_id)
    return (v[0], neighbor_ids)