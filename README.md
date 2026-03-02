# Distributed_KMeans_parallel
This project was carried out with Beccarelli Cesare and Schiavinato Alberto. It consists in creating a cluster of resources which communicate using Docker and then implementing a parallel initialization (Scalable K-Means++, Bahman Bahmani, 2012) and the algorithm for KMeans. These functions ran on the distributed cluster using Dask. 

In particular, we ran the algorithm on the KDDCup1999 dataset and implemented the parallel initialization proposed in the paper by Bahmani. To help keeping the memory on the workers and speeding up the process, we developed a minibatch distributed version of KMeans, where the dataset was divided into blocks which are stored contiguously in memory.

Finally, the notebook we used for the analysis is reported as well.
