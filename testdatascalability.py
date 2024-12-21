#!/usr/bin/python3
"""Test scalability of MRDF. Pass in parameters for MRDF, input/output, and a list of vector counts
(e.g., 100 1000 10000) in order to test to see if MRDF scales linearly."""
import argparse
import time
from pyspark import SparkContext
from mrdf import mrdf
import utilities

if __name__ == '__main__':
    # Take in command line args
    parser = argparse.ArgumentParser()
    # Note: Siftsmall has 10,000 vectors, use that
    parser.add_argument('--rho', type=int, default=15, help='The dividing factor. 15 by default.') # 15 default
    parser.add_argument('--alpha', type=int, default=150000, help='The maximum subset size. 150000 by default.') #150000 default
    parser.add_argument('--k', type=int, default=30, help='Number of neighbors. 30 by default.') # 30 default
    parser.add_argument('--tau', type=float, default=0.01, help='Early termination threshold. 0.01 by default.') # 0.01 default
    parser.add_argument('--randomseed', type=int, default=42, help='Random seed used for MRDF. 42 by default.')
    parser.add_argument('--maxiterations', type=int, default=0, help='Maximum number of iterations (default 0 for no max)')
    parser.add_argument('--inputfile', type=str, help='FVECS input file that contains the vectors')
    parser.add_argument('--vectorcounts', nargs='+', type=int, help='List of vector counts to use, in order')
    # context
    sc = SparkContext('local', 'MRDF - test data scalability')
    # parse args
    aargs = parser.parse_args()
    # plot running time to number of vectors
    plots = []
    experiment_start_time = time.time()
    for vector_count in aargs.vectorcounts:
        fvecs_rdd = SparkContext.parallelize(self=sc,
                                            c=utilities.parse_fvecs_stream(aargs.inputfile, vector_count))
        # Record execution time before running MRDF
        start_time = time.time()
        mrdf(sc, fvecs_rdd, aargs.k, aargs.rho, aargs.alpha, aargs.tau, aargs.randomseed, aargs.maxiterations)
        end_time = time.time()
        running_time = end_time - start_time
        time_tuple = (vector_count, running_time)
        plots.append(time_tuple)
    plots_rdd = SparkContext.parallelize(self=sc, c=plots)
    plots_rdd.saveAsTextFile(aargs.outputfile)
    final_end_time = time.time()
    print(f'Total time taken: {final_end_time - experiment_start_time}')
    print(f'Plots: {plots}')
    


    