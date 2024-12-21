#!/usr/bin/python3
"""Calculate recall from two output text files"""
import argparse
import ast
from pyspark import SparkContext

def count_matching(x, k):
    """Count # of updated edges"""
    knn_neighbors = set(x[1][0])
    mrdf_neighbors = set(x[1][1])
    count = len(knn_neighbors.intersection(mrdf_neighbors))
    return count

if __name__ == '__main__':
    # Take in command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--knn', type=str, help='KNN file to compare')
    parser.add_argument('--mrdf', type=str, help='MRDF file to compare')
    # context
    sc = SparkContext('local', 'MRDFGetRecall')
    # parse args
    aargs = parser.parse_args()
    # Load two RDDs from text files
    # each line should be in form (1, [2, 3])
    knn_rdd = sc.textFile(aargs.knn).map(ast.literal_eval)
    mrdf_rdd = sc.textFile(aargs.mrdf).map(ast.literal_eval)
    # get k
    k = len(knn_rdd.collect()[0][1])
    # calculate total number of edges
    total_edges = k * knn_rdd.count()
    # get number of non-matching edges
    num_matching = knn_rdd.join(mrdf_rdd).map(lambda x: count_matching(x, k)).sum()
    # calculate recall
    recall = num_matching / total_edges
    print(f'Recall: {recall}')