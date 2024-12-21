"""Contains functions used to implement NN-Descent algorithm. Implementation by Evan Mulloy.
The NN-Descent algorithm is introduced here:
Dong, W., Moses, C., & Li, K. (2011). Efficient k-nearest neighbor graph construction for generic
similarity measures. Proceedings of the 20th international conference on World wide web, 577-586.
https://doi.org/10.1145/1963405.1963487"""
import math
from utilities import update_reservoir, measure_euclidean, list_sum

def nn_descent_full(sparkcontext, V, k, sample_rate=1, precision=0.001):
    """Version of NN-Descent that uses parallelization
    V: A dataframe
    K: Number of neighbors
    Sample_rate: The sample rate
    Precision: Precision"""
    #Turn V into a dictionary so we can refer to the original vectors by ID, instead of storing vectors as neighbors
    v_rdd = sparkcontext.parallelize(c=V)
    v_dict = sparkcontext.broadcast(v_rdd.collectAsMap())
    # First step: B[v] ←− Sample(V, K) × {∞, true} ∀v ∈ V
    b = sparkcontext.parallelize(c=sample_infinity(V, k))
    n = v_rdd.count()
    # loop
    terminate = False
    while terminate is False:
        #old[v] ←− all items in B[v] with a false flag
        old_item_rdd = b.map(get_old)
        #new[v] ←− ρK items in B[v] with a true flag
        #Mark sampled items in B[v] as false;
        new_item_rdd = b.map(lambda v: get_new(v, sample_rate, k))
        new_dict = sparkcontext.broadcast(new_item_rdd.collectAsMap())
        # Mark sampled items in B[v] as false;
        b = b.map(lambda v: mark_sampled_as_false(v, new_dict))
        #old′ ← Reverse(old),
        old_reverses = sparkcontext.parallelize(c=generate_reverses(old_item_rdd.collect()))
        old_reverses = old_reverses.reduceByKey(list_sum) # a+b should join two lists
        #new′ ← Reverse(new)
        new_reverses = sparkcontext.parallelize(c=generate_reverses(new_item_rdd.collect()))
        new_reverses = new_reverses.reduceByKey(list_sum)
        # update counter
        c = 0
        #old[v] ←− old[v] ∪ Sample(old′[v], ρK)
        old_reverse_samples = old_reverses.map(lambda v: sample_reverses(v, sample_rate*k))
        old_item_rdd = old_item_rdd.union(old_reverse_samples).reduceByKey(list_sum)
        #new[v] ←−new[v] ∪ Sample(new′[v], ρK)
        new_reverse_samples = new_reverses.map(lambda v: sample_reverses(v, sample_rate*k))
        new_item_rdd = new_item_rdd.union(new_reverse_samples).reduceByKey(list_sum)
        # Old dict to broadcast
        old_dict = sparkcontext.broadcast(old_item_rdd.collectAsMap())
        # for u1, u2 ∈ new[v], u1 < u2
        # or u1 ∈ new[v], u2 ∈ old[v] do
        # l ←− σ(u1, u2)
        nn_updates = new_item_rdd.map(lambda v: get_nn_updates_to_make(v, old_dict, v_dict))
        # Turn list of lists into one list
        # Reduce to merge all the lists into one list of tuples of structure (id, [(weight, neighbor id, flag)])
        # Then, reduce all the tuples BY KEY, so we have (id, [(weight, neighbor id, flag), ...])
        nn_updates = nn_updates.flatMap(lambda x: x).reduceByKey(list_sum)
        # Turn that into a dictionary and broadcast
        nn_updates_dict = sparkcontext.broadcast(nn_updates.collectAsMap())
        # // c and B[.] are synchronized.
        # c ←− c + UpdateNN(B[u1], u2, l, true)
        # c ←− c + UpdateNN(B[u2], u1, l, true)
        updated_b = b.map(lambda v: nn_update_heap(v, nn_updates_dict, k))
        c = b.join(updated_b).map(lambda x: 0 if x[1][0] == x[1][1] else 1).sum()
        # update b
        b = sparkcontext.parallelize(c=updated_b.collect())
        # return B if c < δNK
        if c < (precision * n * k):
            terminate = True
    return b

def sample_infinity(V, K):
    """For v in V, sample(V, K) x <infinity, true>"""
    for v in V:
        v_id = v[0]
        yield v_id, initial_sample(V, K, v_id)

def get_old(v):
    """Return old[v], containing the items with the Flag set to False"""
    old_list = []
    for neighbor in v[1]:
        if neighbor[2] is False:
            old_list.append(neighbor)
    return (v[0], old_list)

def sample_n_nodes(S, n):
    """Sample N nodes from dataset S, including the original weight, node id, and flag."""
    reservoir = []
    for weight, node_id, flag in S:
        item = (weight, node_id, flag)
        reservoir = update_reservoir(reservoir, item, n)
    return reservoir

def get_new(v, sample_rate, k):
    """Return new[v], containing the items with the Flag set to True, sampled by sample rate * K,
    and then with Flags set to False"""
    new_list = []
    for neighbor in v[1]:
        if neighbor[2] is True:
            new_list.append((neighbor[0], neighbor[1], False))
    new_list = sample_n_nodes(new_list, sample_rate * k)
    return (v[0], new_list)

def generate_reverses(V):
    """Build a reverse KNN list. Given a KNN dictionary of the form <id of u, list of neighbor node ids v, w, x, ...>,
    add u as a neighbor in the lists for entries/keys v, w, and x"""
    for u in V:
        u_id = u[0]
        u_neighbors = u[1]
        for neighbor in u_neighbors:
            reverse_item = (neighbor[0], u_id, neighbor[2])
            reverse_list = []
            reverse_list.append(reverse_item)
            reverse_tuple = (neighbor[1], reverse_list)
            yield reverse_tuple

def sample_reverses(v, n):
    """Sample N items from reverse dataset v[1], return (v0, samples)"""
    return (v[0], sample_n_nodes(v[1], n))

def mark_sampled_as_false(v, new_dict):
    """Mark items in B[v] sampled for new[v] as False"""
    # Check if v is in the dictionary
    if (v[0] in new_dict.value) is False:
        return v
    # Get the sampled neighbors of v from the dictionary
    sampled_of_v = new_dict.value[v[0]]
    # If we sampled nothing for B[v], return v
    if len(sampled_of_v) <= 0:
        return v
    current_of_v = v[1]
    new_v_neighbors = []
    for weight1, id1, flag1 in current_of_v:
        for weight2, id2, flag2 in sampled_of_v:
            if id1 == id2:
                flag1 = False
        new_v_neighbors.append((weight1, id1, flag1))
    return (v[0], new_v_neighbors)

# Call this on new_item_rdd
def get_nn_updates_to_make(v, old_dict, v_dict):
    """Generate a list of NN updates to make from the neighbors of V"""
    updates = []
    neighbors_of_v = v[1]
    v_id = v[0]
    neighbors_length = len(neighbors_of_v)
    for i in range(0, neighbors_length):
        for j in range(0, neighbors_length):
            if i is not j:
                u1 = neighbors_of_v[i]
                u2 = neighbors_of_v[j]
                u1_w = u1[0] # w stands for weight
                u2_w = u2[0]
                u1_id = u1[1] # weight comes before id because heaps sort based on first element
                u2_id = u2[1]
                condition_1 = (u1_w < u2_w)
                condition_2 = (u1_w == math.inf and u2_w == math.inf)
                condition_3 = (u2 in old_dict.value[v_id])
                # Make sure we don't add ourselves as a neighbor
                condition_4 = (u1_id is not v_id) and (u2_id is not v_id)
                # for u1, u2 ∈ new[v], u1 < u2
                # or u1 ∈ new[v], u2 ∈ old[v] do
                if (condition_1 or condition_2 or condition_3) and condition_4:
                    u1_vector = v_dict.value[u1_id]
                    u2_vector = v_dict.value[u2_id]
                    # l ←− σ(u1, u2)
                    l = measure_euclidean(u1_vector, u2_vector) # items are tuples of (int, vector)
                    updated_u2 = (l, u2_id, True) # Use this to update B[id of u1]
                    updated_u1 = (l, u1_id, True) # Use this to update B[id of u2]
                    u1_update_tuple = (u1_id, [updated_u2]) # important to put them into lists to do reduce by key
                    u2_update_tuple = (u2_id, [updated_u1])
                    updates.append(u1_update_tuple)
                    updates.append(u2_update_tuple)
    return updates

def nn_update_heap(v, update_dict, k):
    """Update the heap of B[v] using a dictionary full of updates to make."""
    v_id = v[0]
    #print(f'Original heap of {v_id}: {v[1]}')
    if v_id in update_dict.value:
        updates_to_make = []
        updates = update_dict.value[v_id]
        for weight, node_id, flag in updates:
            if node_id != v_id:
                updates_to_make.append((weight, node_id, False))
        updated_heap = list(set(v[1] + updates_to_make))
        # Sort the heap. Because the weight is the first entry, should be sorted by weight.
        #heapq.heapify(updated_heap)
        updated_heap.sort()
        # Get only the first k entries
        updated_heap = updated_heap[:k]
        #print(f'Updated heap of {v_id}: {updated_heap}')
        return (v_id, updated_heap)
    return v # no updates

def initial_sample(S, n, parent_id):
    """Sample N items from dataset S that do not have id parent_id,
    append INFINITY as the distance for each node and setting flag to true, return as heap"""
    reservoir = []
    for node in S:
        node_id = node[0]
        item = (math.inf, node_id, True)
        if node_id != parent_id:
            reservoir = update_reservoir(reservoir, item, n)
    return reservoir
