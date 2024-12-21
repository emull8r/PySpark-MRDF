# PySpark MRDF
A PySpark implementation of Multiway Random Division Forest (MRDF), an algorithm for building approximate K-Nearest Neighbors (KNN) graphs on distributed systems. MRDF is introduced in the following paper by Kim and Park:

Kim, S. H., & Park, H. M. (2023). Efficient Distributed Approximate k-Nearest Neighbor Graph Construction by Multiway Random Division Forest. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1097–1106. https://doi.org/10.1145/3580305.3599327

An approximate KNN graph is a k-neighbor graph such that the proportion of edges in the approximate graph that are also in the *true* KNN graph is really high. This proportion is known as *recall*.

MRDF uses NN-Descent, introduced by Dong, Moses, and Li:

Dong, W., Moses, C., & Li, K. (2011). Efficient k-nearest neighbor graph construction for generic similarity measures. Proceedings of the 20th international conference on World wide web, 577-586.
https://doi.org/10.1145/1963405.1963487

## Prerequisites
The project uses Pyspark. Python 3.9 was used during development.

## How to run
The project contains two runner programs: main.py and getrecall.py. The first can be used to run MRDF as well as brute-force KNN. getrecall.py can be used to compare output from MRDF and output from KNN to compute the recall from running MRDF. Input is assumed to be in FVECS format.

### main.py
main.py accepts .fvecs files as input and is run with the following parameters:

--rho: The dividing factor. 15 by default.

--alpha: The maximum subset size. 150000 by default.

--k: Number of neighbors. 30 by default.

--tau: Early termination threshold. 0.01 by default.

--randomseed: Random seed used for MRDF. 42 by default.

--maxiterations: Maximum number of iterations for MRDF. Use if you want to terminate before the threshold is reached.

--inputmax: Maximum number of input vectors to take. If set to 0 (default), will take ALL vectors.

--inputfile: FVECS input file that contains the vectors. Required.

--outputfile: Name of output folder to save results in (default is "output")

--bruteforce: Run brute-force KNN instead of MRDF (default false to run MRDF).

Running MRDF on main.py will output the execution time in seconds, and save the resulting approximate k-NN graph in
the specified output location. The k-NN graph itself is a text file, where each line follows the format:

(node id, [neighbor id, neighbor id, neighbor id, ...])

The node ID corresponds to the index--starting at 0--of the corresponding vector in the .fvecs input file.
Neighbor IDs correspond to indexes of edge nodes / neighbor vectors. For example, K=1 for two vectors would look like:
(0, [1])
(1, [0])

### getrecall.py
To get the recall, run main.py with and without bruteforce set to true, using different output locations. Output should be saved as text files of tuples that follow the above format. getrecall.py can compare the files to get the recall. It uses the following parameters:

--knn: KNN file to compare (required)
--mrdf: MRDF file to compare (required)

It will infer K from the first file, which should not be empty. The order of files used does not matter. The recall is printed to console.

### testdatascalability.py
Runs MRDF multiple times. --outputfile is replaced with this argument:

--vectorcounts: List of vector counts to use, in order

The script will execute MRDF for each of the maximum vector counts and print the running time.

### Submitting spark job

On a cluster, it can be submitted like this:

spark-submit --files input.fvecs main.py -- --inputfile input.fvecs --alpha=100000 --rho=15 --tau=0.01 --outputfile mrdfsift

On a local system, you can omit the extra "--" between main.py and other parameters.

## Disclaimer

This project was originally a student project created for Old Dominion University CS 722, and as such is imperfect. It tends to "hang" when submitted to a cluster, but will finish within a few minutes on a 1.1 GHz machine when given <= 1000 vectors to process.



