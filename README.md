# NetworkInference
A package which includes various versions of the optimal causation entropy (oCSE).

ESSENTIAL READING ON OCSE: Please visit the paper "Causal Network Inference by Optimal Causation Entropy" By Jie Sun, Dane Taylor and Erik Bollt

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

Documentation:
(Note: there are numerous planned functionalities which are not active yet, but I have begun working on them, please ignore these).

Generating synthetic data on a network and then estimating its network structure
Currently you must supply the (dense) adjacency matrix of a network to generate synthetic data. There are several types of data currently available:
-Gaussian stochastic process-
Example code:

```
from NetworkInference import NetworkInference
import networkx as nx
from matplotlib import pyplot as plt

#GENERATE NETWORK FOR SYNTHETIC DATA
n = 20
p = 0.1
G = nx.erdos_renyi_graph(n,p)
A = nx.adjacency_matrix(G)
A = A.todense()

#SET NETWORK STRUCTURE FOR DATA GENERATION
NI = NetworkInference()
NI.set_NetworkAdjacency(A)

#SET SYNTHETIC DATA GENERATION PARAMETERS AND GENERATE DATA
#This sets the number of time steps to simulate. Current default is 100. 
NI.set_T(250)
#This is 0< rho < 1, which is closely related to the signal to noise ratio. The close rho is to 1 the higher the signal to noise...
#for the actual details on the parameter rho please visit the paper: "Causal Network Inference by Optimal Causation Entropy"
NI.set_Rho(0.95)
#Epsilon in the below expression is the variance of the variables in the stochastic process
#This will generate synthetic data which is stored inside
NI.Gen_Stochastic_Gaussian(Epsilon=1)

#Now that the data has been generated, it is stored internally and if we want we can immediately estimate the network structure,
#or we can retrieve the data if we wish.

#Retrieve the data
Data = NI.return_XY()
plt.plot(Data)

#Now lets estimate the network structure. 
#SET ALL NECESSARY PARAMETERS
#Set the inference method to Gaussian
NI.set_InferenceMethod_oCSE('Gaussian')

#Set the number of shuffles (see the Sun, Taylor, Bollt paper for more details)
NI.set_Num_Shuffles_oCSE(1000)

#Set the alpha value (in the Sun, Taylor, Bollt paper alpha = 1- theta that they used). 
#Essentially this is like a p-value and it is your level of confidence in an edge. 
#The lower alpha the more confident you are in the edges it finds.
NI.set_Forward_oCSE_alpha(0.001)
#There is a forward and a backward stage to the oCSE algorithm...
NI.set_Backward_oCSE_alpha(0.001)


#Now actually estimate the network using the Gaussian oCSE method. This may take some time, but it will print
#progress in terms of which node number it is working on (starting from node 0) to estimate incoming edges.
#note that the run time is mainly dependent on the number of EDGES not the number of nodes...
B = NI.Estimate_Network()

#Now lets see how we did. We will calculate the true positive and false positive rates.
TPR,FPR = NI.Compute_TPR_FPR()
print("This is the TPR and FPR: ",TPR,FPR)

#Finally if we wish we can save the state of all of the parameters we just used, that way we could 
#come back later and use the same data and number of shuffles and alpha and so on... State will be 
#saved in the local directory. To load a saved state use .load_state(). You may have to set a date
#in load_state() and a saved number though as the default is to load the previously saved state from today. 
NI.save_state()
```
We can also imagine a scenario in which the data is discrete and therefore looks a bit more like a Poisson distribution than a Gaussian one. Note that in this scenario it has been shown that using Gaussian oCSE creates a significantly higher false positive rate than using the Poisson version of oCSE. In other words the distribution really does matter! See "Interaction Networks from Discrete Event Data by Poisson Multivariate Mutual Information Estimation and Information Flow with Applications from Gene Expression Data" By Jeremie Fish, Jie Sun and Erik Bollt for more details.

Example code:
-Poisson Count Process -

```
from NetworkInference import NetworkInference
import networkx as nx

n = 20
p = 0.1
G = nx.erdos_renyi_graph(n,p)
A = nx.adjacency_matrix(G)
A = A.todense()
NI = NetworkInference()
NI.set_NetworkAdjacency(A)

NI.set_T(1000)
#The higher Epsilon is with respect to noiseLevel, the higher the signal to noise ratio...
Data = NI.Gen_Poisson_Data(Epsilon=1,noiseLevel=0.5)
NI.set_InferenceMethod_oCSE('Poisson')
NI.set_Num_Shuffles_oCSE(500)
NI.set_Forward_oCSE_alpha(0.01)
NI.set_Backward_oCSE_alpha(0.01)
#The time shift parameter is Tau. In the above paper Tau of 0 was used. 
#In Gaussian oCSE paper Tau = 1 was used. The default in the code is Tau=1.
NI.set_Tau(0)

#Set the data...
NI.set_XY(Data)

#Note the above parameters are not as strict as the paper, see the details above for better results...
B = NI.Estimate_Network()
TPR,FPR = NI.Compute_TPR_FPR()
print("This is the TPR and FPR: ",TPR,FPR)
```

There is also a built in way to find the area under curve (AUC) for TPR vs FPR and to plot the reciever operator curve (ROC). 
Example code:

```
 from NetworkInference import NetworkInference
 import numpy as np
 import networkx as nx
 
 NI = NetworkInference()
 n = 20
 p = 0.1
 G = nx.erdos_renyi_graph(n,p)
 A = nx.adjacency_matrix(G)
 A = A.todense()

 NI.set_NetworkAdjacency(A)

 NI.set_T(250)
 NI.set_Rho(0.95)
 NI.Gen_Stochastic_Gaussian(Epsilon=1e-3)
 NI.set_InferenceMethod_oCSE('Gaussian')
 NI.set_Num_Shuffles_oCSE(500)
 NI.set_Forward_oCSE_alpha(0.005)
 NI.set_Backward_oCSE_alpha(0.005)
 Range = np.arange(0.01,1,0.01)
 TPRs = np.zeros(len(Range))
 FPRs = np.zeros(len(Range))
 for i in range(len(Range)):
     NI.set_Forward_oCSE_alpha(Range[i])
     NI.set_Backward_oCSE_alpha(Range[i])
     B = NI.Estimate_Network()
     TPR,FPR = NI.Compute_TPR_FPR()
     print()
     print()
     print("This is the TPR and FPR: ",TPR,FPR)
     print()
     print()
     TPRs[i] = TPR
     FPRs[i] = FPR

 AUC = NI.Compute_AUC(TPRs,FPRs)
 print("This is the AUC: ", AUC)
 NI.Plot_ROC(TPRs,FPRs)
```

Another available inference method is the k nearest neighbors (KNN) version of oCSE. This takes a tremendous amount of data to converge however and a long time to run, so don't count on this method accurately finding reasonably dense network structures...
You can also generate logistic dynamics over the network as shown in the paper "Causation Entropy Identifies Indirect Influences, Dominance of Neighbors and Anticipatory Couplings" by Jie Sun and Erik Bollt.
Example code:

```
from NetworkInference import NetworkInference
import networkx as nx

 NI = NetworkInference()
 n = 20
 p = 0.1
 G = nx.erdos_renyi_graph(n,p)
 A = nx.adjacency_matrix(G)
 A = A.todense()
 NI.set_NetworkAdjacency(A)
 NI.set_T(250)
 #Generate logistic dynamics from the 2014 paper...
 NI.Gen_Logistic_Dynamics(r = 4, sigma = 0.5)
 
 #Set the inference method to KNN
 NI.set_InferenceMethod_oCSE('KNN')
 NI.set_Num_Shuffles_oCSE(500)
 
 #Set how many neareast neighbors...
 NI.set_KNN_K(8)
 B = NI.Estimate_Network()
 TPR,FPR = NI.Compute_TPR_FPR()
 print("This is the TPR and FPR: ",TPR,FPR)
 #Don't forget you can save the state if you wish..
 NI.save_state()
 
```
In the above code I may wish to come back and reload my previously saved state that can be achieved with the following code:
```
from NetworkInference import NetworkInference

NI.load_state()
#NOTE The above will load the previously saved state from TODAY. You can access other saved states by putting in the appropriate
#date it was saved and also the number of the save if you saved multiple in a day
#you would use NI.load_state(Date='2022-09-13') if you saved a single thing on that date or NI.load_state(Num=3) to access the 
#third one you saved today...


#Now I can access everything, for instance to find out what value of k I used
print(NI.return_KNN_K())

#Or to see the matrix of data I used...
print(NI.return_XY())

#I can even access the parameter I used for the Logistic map model
print(NI.return_Logistic_parameter_r())
```
You may also find yourself wishing that you could upload some data, rather than generating synthetic data. This can be achieved as well quite easily, but note that you can't get TPR or FPR anymore (unless you also add a network adjacency of the true network structure, but I am not sure you would be able to do that with real data). Note that the data is assumed to be input as a (T x n) matrix where T is the number of samples and n is the number of nodes or features if you wish.

```
import numpy as np
from NetworkInference import NetworkInference

#Load in the data you wish to bring in I am using np.load as an example but you could load in other ways, but to preserve b
#behavior it is best to feed the data in as a numpy array
Data = np.load(SOME FILENAME)

NI = NetworkInference()
NI.set_XY(Data)

B = NI.Estimate_Network()
```

That covers available functionality for now. Check in again later for more!
