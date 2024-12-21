"""Contains utilities for reading, writing, etc."""
import os
import struct
import random
import numpy as np

def list_sum(a, b):
    """Concatenate lists"""
    return a+b

def measure_euclidean(u1, u2):
    """Measure the Euclidean distance between u1 and u2, two vectors"""
    return np.linalg.norm(u1-u2)

def reservoir_sampling(subset, k):
    """Select k items from the subset"""
    reservoir = []
    for item in subset:
        reservoir = update_reservoir(reservoir, item, k)
    return reservoir

def update_reservoir(reservoir, item, n):
    """Update a reservoir for reservoir sampling. Item is an item to add, N is the size the reservoir should be."""
    if len(reservoir) < n:
        reservoir.append(item)
    else:
        j = random.randint(0, len(reservoir))
        if j < len(reservoir):
            reservoir[j] = item
    return reservoir

def write_fvecs(filename, vectors):
    """Write an fvecs file"""
    with open(filename, "wb") as f:
        for vector in vectors:
            # Write the dimension of the vector
            f.write(struct.pack("<i", len(vector)))
            # Write the vector elements as floats
            for val in vector:
                f.write(struct.pack("<f", val))

def check_path(file_path):
    """Check if a file exists"""
    return os.path.exists(file_path)

def parse_fvecs_stream(file_path, input_max=0):
    """Parse a fvecs file as a stream. Set input max to 0 (default) to parse everything."""
    root_tree_path = ',' # when starting out MRDF, we are at the root level
    with open(file_path, 'rb') as f:
        node_id = -1
        parsed_vectors = 0
        while (input_max == 0) or (parsed_vectors < input_max) :
            # Read the dimension of the vector
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # End of file
            dim = int.from_bytes(dim_bytes, byteorder='little')
            # Read the vector data
            vector_bytes = f.read(dim * 4)
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            node_id = node_id + 1
            # return tuple of type string, tuple(int, vector), where int represents the id to be used for edges later
            tree_path_and_vector = root_tree_path, (node_id, vector)
            parsed_vectors = parsed_vectors + 1
            yield tree_path_and_vector

def parse_fvecs_stream_no_treepath(file_path, input_max=0):
    """Parse a fvecs file as a stream. Set input max to 0 (default) to parse everything.
    Do not include the empty treepath for MRDF. """
    with open(file_path, 'rb') as f:
        node_id = -1
        parsed_vectors = 0
        while (input_max == 0) or (parsed_vectors < input_max) :
            # Read the dimension of the vector
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # End of file
            dim = int.from_bytes(dim_bytes, byteorder='little')
            # Read the vector data
            vector_bytes = f.read(dim * 4)
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            node_id = node_id + 1
            # return tuple of type string, tuple(int, vector), where int represents the id to be used for edges later
            vector_tuple = (node_id, vector)
            parsed_vectors = parsed_vectors + 1
            yield vector_tuple

def parse_fvecs_stream_just_vectors(file_path, input_max=0):
    """Parse a fvecs file as a stream. Set input max to 0 (default) to parse everything.
    Only return vectors, no other keys."""
    with open(file_path, 'rb') as f:
        parsed_vectors = 0
        while (input_max == 0) or (parsed_vectors < input_max) :
            # Read the dimension of the vector
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # End of file
            dim = int.from_bytes(dim_bytes, byteorder='little')
            # Read the vector data
            vector_bytes = f.read(dim * 4)
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            # return tuple of type string, tuple(int, vector), where int represents the id to be used for edges later
            parsed_vectors = parsed_vectors + 1
            yield vector

def parse_ivecs_stream(file_path, input_max=0, k=0):
    """Parse a ivecs file as a stream. Specify an input max to stop parsing after the max is reached.
    Specify a k value to return the first k values of the vector.
    Set input max to 0 (default) to parse everything, and set k to 0 (default) to get the full vector.
    """
    with open(file_path, 'rb') as f:
        node_id = -1
        parsed_vectors = 0
        while (input_max == 0) or (parsed_vectors < input_max) :
            # Read the dimension of the vector
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # End of file
            dim = int.from_bytes(dim_bytes, byteorder='little')
            # Read the vector data, use k if not 0
            if k < dim:
                dim = k
            vector_bytes = f.read(dim * 4)
            vector = np.frombuffer(vector_bytes, dtype=np.int32)
            node_id = node_id + 1
            # return tuple of type string, tuple(int, vector), where int represents the id to be used for edges later
            node_and_vector = (node_id, vector)
            parsed_vectors = parsed_vectors + 1
            yield node_and_vector

def parse_fvecs_array(file_path):
    """Reads an fvecs file and returns a numpy array if it exists"""
    data = []
    if check_path(file_path):
        with open(file_path, 'rb') as f:
            # Read the dimension from the first 4 bytes
            dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
            # Read the rest of the file as a byte array
            data = f.read()
            # Reshape the data into vectors
            data = np.frombuffer(data, dtype=np.float32)
            data = data.reshape(-1, dim)
    return data
