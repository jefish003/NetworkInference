# NetworkInference
A package which includes various versions of the optimal causation entropy (oCSE).
NetworkInference is a python package which contains different methods for estimating network/graph structure using oCSE. Eventually other methods will be included as well, such as Graphical Lasso, traditional Lasso, Entropic Regression and several more, however those are not currently available.

Currently available methods are the Standard Gaussian version of oCSE (assuming Gaussian random variables), K-nearest neighbor (KNN) oCSE (a nonparametric estimator which does not assume a distribution, but needs significantly more data to converge to the correct network structure), and Poisson oCSE which assumes Poisson random variables.


Installation:
You will need a python environment which has the following packages installed: 
sklearn,
scipy,
numpy,
itertools,
copy,
datetime,
glob,
matplotlib.

This code has been tested on python 3.8+, so earlier versions of python may not work (as some functionalities in the libraries may not have existed in prior versions), so you may try python 3.7 or earlier at your own risk.

Once you have the above packages you will need to download NetworkInference.py to the directory you wish to import the library from.
