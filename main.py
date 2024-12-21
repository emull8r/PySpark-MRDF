#!/usr/bin/python3
"""Main runner program"""
import time
import argparse
from pyspark import SparkContext
import utilities
from mrdf import mrdf
from knn import brute_force_knn

# At first,
# CentroidSampling initializes an empty set 𝐶𝑝 to store sampled
# elements (line 3), reads the input stream one by one, and stores the
# first 𝜌 elements in the set (lines 5-7).
if __name__ == '__main__':
    # Take in command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', type=int, default=15, help='The dividing factor. 15 by default.') # 15 default
    parser.add_argument('--alpha', type=int, default=150000, help='The maximum subset size. 150000 by default.') #150000 default
    parser.add_argument('--k', type=int, default=30, help='Number of neighbors. 30 by default.') # 30 default
    parser.add_argument('--tau', type=float, default=0.01, help='Early termination threshold. 0.01 by default.') # 0.01 default
    parser.add_argument('--randomseed', type=int, default=42, help='Random seed used for MRDF. 42 by default.')
    parser.add_argument('--maxiterations', type=int, default=0, help='Maximum number of iterations (default 0 for no max)')
    parser.add_argument('--inputmax', type=int, default=0, help='Maximum number of input vectors to take. If set to 0 (default), will take ALL vectors.')
    parser.add_argument('--inputfile', type=str, help='FVECS input file that contains the vectors')
    parser.add_argument('--outputfile', type=str, default='output', help='Name of output folder to save results in. Default is \'output\'.')
    parser.add_argument('--bruteforce', type=bool, default=False, help='Run brute-force KNN instead of MRDF (default false).')
    sc = SparkContext('local', 'MRDF')
    # parse args
    aargs = parser.parse_args()
    # get rdd
    fvecs_rdd = sc.parallelize(c=utilities.parse_fvecs_stream_no_treepath(aargs.inputfile, aargs.inputmax))
    # Record execution time before running MRDF
    start_time = time.time()
    if aargs.bruteforce is False:
        # Run MRDF
        print('Running MRDF ...')
        g = mrdf(sc, fvecs_rdd, aargs.k, aargs.rho, aargs.alpha, aargs.tau, aargs.randomseed, aargs.maxiterations)
    else:
        # Brute force KNN for getting recall
        print('Running brute force KNN ...')
        g = brute_force_knn(sc, fvecs_rdd, aargs.k)
    # Get time taken after running MRDF
    end_time = time.time()
    execution_time = end_time - start_time
    # Save to output
    g.saveAsTextFile(aargs.outputfile)
    # Print execution time
    print(f'Total time taken to run algorithm: {execution_time} s')


    


