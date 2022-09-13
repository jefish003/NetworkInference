# NetworkInference
A package which includes various versions of the optimal causation entropy (oCSE).
NetworkInference is a python package which contains different methods for estimating network/graph structure using oCSE. Eventually other methods will be included as well, such as Graphical Lasso, traditional Lasso, Entropic Regression and several more, however those are not currently available.

Currently available methods are the Standard Gaussian version of oCSE (assuming Gaussian random variables), K-nearest neighbor (KNN) oCSE (a nonparametric estimator which does not assume a distribution, but needs significantly more data to converge to the correct network structure), and Poisson oCSE which assumes Poisson random variables.
