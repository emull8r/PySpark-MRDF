#!/usr/bin/python3
"""Unfinished script to calculate recall given both a graph generated from MRDF (text file)
and a ground truth file (IVECS)"""
import argparse
import ast
import numpy as np
from pyspark import SparkContext
import utilities

def convert_lists_to_np(s):
    """Convert the tuple of Python lists into a tuple of NumPy arrays."""
    tuple_of_lists = ast.literal_eval(s)
    return (str(np.array(tuple_of_lists[0], dtype=np.float32)),
            str(np.array(tuple_of_lists[1], dtype=np.int32)))

if __name__ == '__main__':
    # Take in command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='The graph file output from MRDF')
    parser.add_argument('--query', type=str, help='The query file in FVECS format')
    parser.add_argument('--groundtruth', type=str, help='The ground truth file in IVECS format')
    parser.add_argument('--k', type=int, help='Number of neighbors')
    # context
    sc = SparkContext('local', 'MRDF get recall')
    # parse args
    aargs = parser.parse_args()
    # Load graph file from text file
    # each line should be in form ([1.0, 3.4], [2, 3])
    graph = sc.textFile(aargs.graph).map(convert_lists_to_np)
    print(f'Graph: {graph.collect()[0]}')
    # Load query
    query = sc.parallelize(c=utilities.parse_fvecs_stream_no_treepath(aargs.query, 0))
    #print(f'Query: {query.collect()[0]}')
    # Load groundtruth
    groundtruth = sc.parallelize(c=utilities.parse_ivecs_stream(aargs.groundtruth, 0, aargs.k))
    #print(f'Groundtruth: {groundtruth.collect()[0]}')
    # Merge query with groundtruth
    query_groundtruth = query.join(groundtruth).map(lambda x: (str(x[1][0]), str(x[1][1])))
    #print(f'Combined: {query_groundtruth.collect()[0]}')
    # Merge with graph, reduce by key
    #query_groundtruth_graph = graph.join(query_groundtruth)
    #print(f'Combined: {query_groundtruth_graph.collect()[0]}')