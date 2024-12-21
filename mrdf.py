"""Multiway Random Division Forest, implemented by Evan Mulloy. Algorithm introduced in the following paper:
Kim, S. H., & Park, H. M. (2023). Efficient Distributed Approximate k-Nearest Neighbor Graph Construction by
Multiway Random Division Forest. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery
and Data Mining, 1097–1106. https://doi.org/10.1145/3580305.3599327"""
import random
import math
import time
from collections import defaultdict
#from pyspark import SparkContext, StorageLevel
from utilities import reservoir_sampling, measure_euclidean, list_sum
from nndescent import nn_descent_full

def mrdf(sparkcontext, rdd, k, rho, alpha, tau, random_seed=42, max_iter=0):
    """Given a spark context, Spark RDD rdd that contains vectors v, 
    #neighbors k, dividing factor rho, subset size limit alpha,
    threshold limit tau, random seed (optional) and max iterations (optional),
    perform k-NN and produce graph g of v """
    # Persist original RDD for later
    #rdd.persist(StorageLevel.DISK_ONLY)
    # Add treepaths
    treepath_rdd = rdd.map(lambda x: ('', x))
    #MRDF ALGORITHM STARTS HERE
    # initialize g, remember, rdd is an fvecs that that contains a list of tuples of (string, numpy array),
    # where the string is the tree path. This way, they are already blocks.
    print('Initialize G ...')
    last_time = time.time()
    g = initialize_g(sparkcontext, rdd)
    print(f'Initializing G took {time.time() - last_time} s')
    # Random seed
    random.seed(random_seed)
    # Count iterations
    iteration = 1
    # set up outer while loop
    termination = False
    while termination is False:
        print(f'-------------Iteration: {iteration}-------------')
        # initialize all tree paths to be empty. We will use strings to be tree paths.
        print('Setting empty tree paths ...')
        last_time = time.time()
        treepath_rdd = treepath_rdd.map(lambda x: ('', x[1]))
        tree_paths = ['']
        print(f'Setting tree paths took {time.time() - last_time} s')
        # partitioning phase.
        print('Partitioning phase ...')
        print('Checking subset sizes ...')
        last_time = time.time()
        while check_subset_sizes(rdd, alpha): # Check to see if there are subsets less than alpha in size
            print(f'Checking subset sizes took {time.time() - last_time} s')
            print('Centroid sampling ...')
            last_time = time.time()
            # broadcast C_curr centroids to all machines
            centroids = sparkcontext.broadcast(centroid_sampling_2(treepath_rdd, tree_paths, rho, alpha))
            print(f'Centroid sampling took {time.time() - last_time} s')
            print('Tree path extension ...')
            last_time = time.time()
            treepath_rdd, tree_paths = tree_path_extension(treepath_rdd, centroids)
            print(f'Tree path extension took {time.time() - last_time} s')
        print('Local graph construction ...')
        last_time = time.time()
        g_prime = local_graph_construction(sparkcontext, treepath_rdd, k)
        print(f'Local graph construction took {time.time() - last_time} s')
        print('Graph update ...')
        last_time = time.time()
        g, ratio = graph_update(sparkcontext, g, g_prime, k)
        print(f'Graph update took {time.time() - last_time} s')
        termination = (ratio <= tau) # terminate when this is true
        # the below is an addition to the MRDF algorithm, max iterations
        iteration += 1
        if max_iter > 0 and iteration >= max_iter:
            termination = True
        write_out_mrdf_details(sparkcontext, rdd, k, rho, alpha, tau, random_seed, max_iter, iteration)
    return format_g(g) # should be of form (node id, [node id 1, node id 2, ...])


def centroid_sampling_2(rdd, tree_paths, rho, alpha):
    """2nd implementation of the Sample Centroids step where centroids are sampled on each cluster machine."""
    # This returns a list where each row is two dictionaries,
    # one for the counts of each key and one for sampled centroids for each key
    dictionary_list = rdd.mapPartitions(lambda iterator : centroid_sampling_partition_2(iterator, tree_paths, rho))
    #print(f'Dictionary list: {dictionary_list.collect()}')
    # Shuffle step the input is rho * #machines centroids
    # Shuffle step occurs automatically
    counts_and_centroids = dictionary_list.reduceByKey(lambda a,b: centroid_sampling_reduce_2(a, b, rho))
    #print(f'Samples: {counts_and_centroids.collect()}')
    # get only centroids where the count is >= alpha
    centroids = counts_and_centroids.map(lambda sample: centroid_sampling_map_2(sample, alpha)).collectAsMap()
    return centroids

def centroid_sampling_reduce_2(tuple1, tuple2, rho):
    """Combine counts with counts, sampled centroids with sampled centroids"""
    new_tuple = (tuple1[0]+tuple2[0], reservoir_sampling(list_sum(tuple1[1],tuple2[1]), rho))
    return new_tuple

def centroid_sampling_map_2(count_and_sample, alpha):
    """Check if the centroids (sampled for a specific key) have a length greater than or equal to alpha.
    If so, return just the key and its centroids. Otherwise, return just the key and the empty set."""
    if count_and_sample[1][0] >= alpha:
        return (count_and_sample[0], count_and_sample[1][1])
    return (count_and_sample[0], []) # return empty list

def centroid_sampling_partition_2(iterator, tree_paths, rho):
    """Sample centroids for each partition.
    Although the reservoir sampling function isn't called, reservoir sampling is performed here."""
    centroids = defaultdict(list)
    counts = defaultdict(int)
    # Initialize list of centroids and counts
    for path in tree_paths:
        centroids[path] = []
        counts[path] = 0
    for key, value in iterator:
        counts[key] = counts[key] + 1
        if counts[key] <= rho:
            centroids[key].append(value)
        else:
            key_count = counts[key]
            j = int(random.randint(1, key_count) * (key_count + 1))
            if j < rho:
                centroids[key][j] = value
    for key, value in centroids.items():
        counts_and_centroids = (counts[key], value)
        yield (key, counts_and_centroids)

def tree_path_extension(rdd, centroids):
    """Tree path extension step that updates the logical tree paths of each vector.
    Returns an updated RDD and updated tree paths list."""
    updated_rdd = rdd.map(lambda pair : tree_path_extension_map(pair, centroids.value))
    tree_paths = updated_rdd.countByKey().items()
    return updated_rdd, tree_paths

def tree_path_extension_map(pair, centroids):
    """Tree path extension that happens on each partition in the cluster"""
    key, value = pair
    if len(centroids[key]) > 0: # centroids is used as a dictionary
        i = 0
        min_i = 0
        shortest_distance = math.inf
        for centroid in centroids[key]:
            # each centroid should be a tuple (int, vector), where int is the id, so we measure by [1]
            centroid_distance = measure_euclidean(centroid[1], value[1])
            if centroid_distance < shortest_distance:
                shortest_distance = centroid_distance
                min_i = i
            i = i + 1
        updated_key = key + ',' + str(min_i)
        return updated_key, value
    return pair
            
def local_graph_construction(sparkcontext, rdd, k):
    """Build a graph from the vectors belonging to each tree path, and combine the graphs"""
    g_prime = sparkcontext.emptyRDD()
    for key, list_of_results in rdd.groupByKey().collect():
        g_prime = g_prime.union(nn_descent_full(sparkcontext=sparkcontext, V=list_of_results, k=k))
    return g_prime

def graph_update(sparkcontext, g, g_prime, k):
    """Update G with G'"""
    # Update G
    updated_g = g.union(g_prime).reduceByKey(lambda a,b: graph_update_map(a,b,k))
    updated_g = sparkcontext.parallelize(updated_g.collect())
    # Calculate ratio
    num_edges = updated_g.count() * k
    num_updated_edges = g.join(updated_g).map(lambda x: graph_update_count_updates(x, k)).sum()
    ratio = num_updated_edges / num_edges
    return updated_g, ratio

def graph_update_map(a, b, k):
    """Use with reduce by key to merge lists of neighbors"""
    updated_list = list(set(a+b))
    updated_list.sort()
    return updated_list[:k]

def graph_update_count_updates(x, k):
    """Count # of updated edges"""
    old_neighbors = set(x[1][0])
    new_neighbors = set(x[1][1])
    if(len(old_neighbors) <= 0):
        return k # If we are just starting out, neighbors will be empty, so return k
    # else return # of different neighbors (AKA edges), should be 0 if both are same
    return len(old_neighbors.difference(new_neighbors))

def check_subset_sizes(rdd, alpha):
    """Given an RDD and min subset size alpha, check to see if there are any subsets greater than alpha in size.
    Return true if this is the case, false otherwise."""
    tree_path_counts = rdd.countByKey()
    for key, count in tree_path_counts.items():
        if count >= alpha:
            return True
    return False

def initialize_g(sparkcontext, rdd):
    """Initialize G to be (V, empty set)"""
    g = sparkcontext.parallelize(c=rdd.map(initialize_g_map).collect())
    return g

def initialize_g_map(v):
    """Return the vector id and an empty list for each vector"""
    return (v[1][0], []) # Remember, v[0] should be tree path, v[1][0] should be the node id

def format_g(g):
    """OPTIONAL: After nn_descent_full, g is of the form [(vector id, [(weight, neighbor vector id, flag), ...]), ...].
    Get rid of the weights and the flag from G so that it is simply [(vector id, [neighbor id 1, neighbor id 2, ...]), ...]."""
    g = g.map(remove_weight_and_flag_from_neighbors).sortByKey(ascending=True)
    return g

def format_g_with_vectors(g, rdd):
    """OPTIONAL: Given formatted G and the original RDD, replace node ids with the vectors to compare with query file"""
    combined = g.join(rdd).map(lambda x: (x[1][1].tolist(), x[1][0])) # should now be have form (vector, [node id 1, node id 2, ...])
    return combined

def remove_weight_and_flag_from_neighbors(node):
    """Self explanatory, see format_g"""
    just_neighbor_ids = []
    for neighbor in node[1]:
        just_neighbor_ids.append(neighbor[1])
    return (node[0], just_neighbor_ids)

def write_out_mrdf_details(sparkcontext, rdd, k, rho, alpha, tau, random_seed, max_iter, iteration):
    """Write memory used and other parameters."""
    total_memory = sparkcontext._jvm.Runtime.getRuntime().totalMemory()
    free_memory = sparkcontext._jvm.Runtime.getRuntime().freeMemory()
    used_memory = total_memory - free_memory
    count = rdd.count()
    jobname = f'output/mrdfk{k}rho{rho}alpha{alpha}count{count}.txt'
    with open(file=jobname, mode='a') as f:
        f.write(f'Iteration: {iteration}, memory={used_memory}, k={k}, rho={rho}, tau={tau}, seed={random_seed}, max iterations={max_iter}\n')
        f.close()

# NOTE: This is the first implementation, current one is centroid_sampling_2
#def centroid_sampling(rdd, tree_paths, rho, alpha):
#    """Sample centroids on each cluster machine"""
#    # This returns a list where each row is two dictionaries,
#    # one for the counts of each key and one for sampled centroids for each key
#    dictionary_list = rdd.mapPartitions(lambda iterator : centroid_sampling_partition(iterator, tree_paths, rho))
#    # Shuffle step the input is rho * #machines centroids
#    # Shuffle step occurs automatically
#    #NOTE: May want to look into reduceByKey or foldByKey for this
#    combined_centroids = defaultdict(list)
#    combined_counts = defaultdict(int)
#    # Combine all of the counts and all the picked centroids
#    i = 1
#    for row in dictionary_list:
#        if i % 2 != 0:
#            #If counter is odd, process as count dict
#            for key, count in row.items():
#                combined_counts[key] = combined_counts[key] + count
#        else:
#            #Process as centroid dict
#            for key, sample in row.items():
#                for vector in sample:
#                    combined_centroids[key].append(vector)
#        i = i + 1
#    # Set sampled centroids only if the count of the tree path is greater than alpha
#    for key, sample in combined_centroids.items():
#        if combined_counts[key] >= alpha:
#            combined_centroids[key] = reservoir_sampling(sample, rho)
#        else:
#            combined_centroids[key] = []
#    return combined_centroids

#def centroid_sampling_partition(iterator, tree_paths, rho):
#    """Sample centroids for each partition.
#    Although the reservoir sampling function isn't called, reservoir sampling is performed here."""
#    centroids = defaultdict(list)
#    counts = defaultdict(int)
#    # Initialize list of centroids and counts
#    for path in tree_paths:
#        centroids[path] = []
#        counts[path] = 0
#    for key, value in iterator:
#        counts[key] = counts[key] + 1
#        if counts[key] <= rho:
#            centroids[key].append(value)
#        else:
#            key_count = counts[key]
#            j = int(random.randint(1, key_count) * (key_count + 1))
#            if j < rho:
#                centroids[key][j] = value
#    dictionary_tuple = (counts, centroids)
#    return dictionary_tuple
