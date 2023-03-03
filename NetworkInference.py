# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 08:39:51 2022

@author: fishja
"""

from sklearn.neighbors import KernelDensity
from scipy.spatial import distance
from scipy.special import gamma as Gamma
from scipy.special import digamma as Digamma
import scipy.stats as SStats
import numpy as np
import itertools
import copy
import datetime 
import glob
from matplotlib import pyplot as plt
from matplotlib import rc

"""Written by Jeremie Fish, 
   Last Update: April 6th 2022
   
   Depending on method used please cite the appropriate papers..."""

class NetworkInference:
    """A class for NetworkInference. Version number 0.1"""
    def __init__(self):
        
        #Version 
        self.VersionNum = 0.1
        
        #Time series stuff
        self.n = 100
        self.T = 200
        
        #Initialize time series to nothing, this must be entered or generated
        self.X = None
        self.Y = None
        self.Z = None
        self.XY = None
        
        #Network stuff
        self.NetworkAdjacency = None
        self.NetworkType = None
        self.Lin_Stoch_Gaussian_Adjacency = None
        self.Logistic_Adjacency = None
        
        #Rho, which is effectively how "noise dominated" the system is
        #It is the spectral radius of the input matrix
        self.Rho = None
        
        #Inference Stuff
        self.Overall_Inference_Method = 'Standard_oCSE'
        self.InferenceMethod_oCSE = 'KernelDensity'
        self.KernelBandWidth = None
        self.KernelType = 'gaussian'
        self.KNN_K = None
        self.Tau = int(1) #Tau is assumed to be 1 always but can be changed
        
        #oCSE stuff
        self.Num_Shuffles_oCSE = 100
        self.Forward_oCSE_alpha = 0.02
        self.Backward_oCSE_alpha = 0.02
        
        
        #Miscellaneous
        self.Pedeli_PoissonAlphas = None
        self.Correlation_XY = None
        self.Prctile = None
        self.Dict = None
        self.Sinit = None
        self.Sfinal = None
        #TO DO,Add all of the other listed methods. These will become available in future renditions of the software.
        self.Available_Inference_Methods = ['Standard_oCSE', 'Alternative_oCSE']#['Standard_oCSE', 'Alternative_oCSE', 'Lasso', 'Graphical_Lasso', 
                                           # 'GLM_Lasso', 'Kernel_Lasso','Entropic_Regression','Convergent_Cross_Mapping',
                                           # 'PCMCI','PCMCIplus','LPMCI','Partial_Correlation','GPDC']
        self.DataType = 'Continuous'
        self.font = None
        
        #Dynamics stuff
        self.Gaussian_Epsilon = None
        self.Logistic_parameter_r = None
        self.Logistic_Coupling = None
        self.Armillotta_Poisson_Betas = None
        self.Armillotta_Poisson_Lambda_0 = None
        
        #For Error computation
        self.B = None # The estimated Adjacency matrix
        self.TPR = None
        self.FPR = None
        self.AUC = None
    
    def set_font(self,font):
        self.font = font
    
    def set_AUC(self,AUC):
        self.AUC = AUC
    
    def set_FPR(self,FPR):
        self.FPR = FPR
    
    def set_TPR (self,TPR):
        self.TPR = TPR
    
    def set_B(self,B):
        self.B = B
        
    def set_Tau(self,Tau):
        self.Tau = int(Tau)
    
    def set_DataType(self,Type):
        Types = ['Discrete', 'Continuous']
        if Type not in Types:
            raise ValueError("Sorry the only types allowed for DataType are: ", Types)
        
        self.DataType = Type
    
    def set_Available_Inference_Methods(self,Methods):
        self.Available_Inference_Methods = Methods
    
    def set_Armillotta_Poisson_Betas(self,Betas):
        self.Armillotta_Poisson_Betas = Betas
        
    def set_Armillotta_Poisson_Lambda_0(self,Lam0):
        self.Armillotta_Poisson_Lambda_0 = Lam0
    
    def set_Lin_Stoch_Gaussian_Adjacency(self,A):
        self.Lin_Stoch_Gaussian_Adjacency = A
        
    def set_Logistic_Adjacency(self,A):
        self.Logistic_Adjacency = A
        
    def set_Pedeli_PoissonAlphas(self,alphas):
        self.Pedeli_PoissonAlphas = alphas
        
    def set_Prctile(self,Prctile):
        self.Prctile = Prctile
        
    def set_Dict(self,Dict):
        self.Dict = Dict
        
    def set_Sinit(self,Sin):
        self.Sinit = Sin
        
    def set_Sfinal(self,Sfin):
        self.Sfinal = Sfin
        
    def set_Gaussian_Epsilon(self,Eps):
        self.Gaussian_Epsilon = Eps
        
    def set_Logistic_parameter_r(self,r):
        self.Logistic_parameter_r = r
        
    def set_Logistic_Coupling(self,Sigma):
        self.Logistic_Coupling = Sigma
        
    def set_n(self,n):
        self.n = n
    
    def set_T(self,T):
        self.T = T
        
    def set_X(self,X):
        """The time series (T x n) which is the set of predictors"""
        
        self.X = X
        self.T = X.shape[0]
    
    def set_Y(self,Y):
        """The time series (T x m) of target variable(s)
           (note m usually is equal to 1 but for generalized 
           interactions between many nodes it may be useful 
           to have m>1) """
        
        self.Y = Y
        if self.Y is not None:
            self.T = Y.shape[0]
        
    def set_Z(self,Z):
        """Set the conditioning set of variables. It is possible that
           you do not want to condition on anything in which case it
           is fine that Z is None"""
        self.Z = Z
        
    def set_XY(self,XY):
        
        self.XY = XY
        if self.XY is not None:
            self.T = XY.shape[0]
            self.n = XY.shape[1]
        
    def set_Forward_oCSE_alpha(self,alpha):
        self.Forward_oCSE_alpha = alpha
        
    def set_Num_Shuffles_oCSE(self,Num_Shuffles):
        self.Num_Shuffles_oCSE = Num_Shuffles
        
    def set_Backward_oCSE_alpha(self,alpha):
        self.Backward_oCSE_alpha = alpha
        
    def set_Correlation_XY(self,Correlation_XY):
        self.Correlation_XY = Correlation_XY
        
    def set_Overall_Inference_Method(self,Method):
        self.Overall_Inference_Method = Method
    
    def set_InferenceMethod_oCSE(self,Method):
        self.InferenceMethod_oCSE = Method
        
    def set_KernelBandWidth(self,Bandwidth):
        self.KernelBandWidth = Bandwidth
    
    def set_KernelType(self,Type):
        self.KernelType = Type
        
    def set_KNN_K(self,K):
        self.KNN_K = K
        
    def set_NetworkAdjacency(self,A):
        """Adjacency matrix assumed to be (n x n)"""
        self.NetworkAdjacency = A
        self.n = A.shape[0]
        
    def set_NetworkType(self,Type):
        self.NetworkType = Type
        
    def set_Rho(self,Rho):
        self.Rho = Rho
        
    def return_font(self):
        return self.font
        
    def return_DataType(self):
        return self.DataType
        
    def return_VersionNum(self):
        return self.VersionNum
      
    def return_Available_Inference_Methods(self):
        return self.Available_Inference_Methods
      
    def return_n(self):
        return self.n
    
    def return_T(self):
        return self.T
        
    def return_Num_Shuffles_oCSE(self):
        return self.Num_Shuffles_oCSE
    
    def return_Z(self):
        return self.Z
    
    def return_X(self):
        return self.X
    
    def return_Y(self):
        return self.Y
         
    def return_XY(self):
        return self.XY
    
    def return_NetworkAdjacency(self):
        return self.NetworkAdjacency
    
    def return_NetworkType(self):
        return self.NetworkType
    
    def return_Rho(self):
        return self.Rho
    
    def return_Overall_Inference_Method(self):
        return self.Overall_Inference_Method
    
    def return_InferenceMethod_oCSE(self):
        return self.InferenceMethod_oCSE
    
    def return_KernelBandWidth(self):
        return self.KernelBandWidth
    
    def return_KernelType(self):
        return self.KernelType
    
    def return_KNN_K(self):
        return self.KNN_K
    
    def return_Tau(self):
        return self.Tau
    
    def return_Forward_oCSE_alpha(self):
        return self.Forward_oCSE_alpha
    
    def return_Backward_oCSE_alpha(self):
        return self.Backward_oCSE_alpha
    
    def return_Pedeli_PoissonAlphas(self):
        return self.Pedeli_PoissonAlphas
    
    def return_Correlation_XY(self):
        return self.Correlation_XY
    
    def return_Prctile(self):
        return self.Prctile
    
    def return_Sinit(self):
        return self.Sinit
    
    def return_Sfinal(self):
        return self.Sfinal
    
    def return_Dict(self):
        return self.Dict
    
    def return_Lin_Stoch_Gaussian_Adjacency(self):
        return self.Lin_Stoch_Gaussian_Adjacency
    
    def return_Logistic_Adjacency(self):
        return self.Logistic_Adjacency
    
    def return_Gaussian_Epsilon(self):
        return self.Gaussian_Epsilon
    
    def return_Logistic_parameter_r(self):
        return self.Logistic_parameter_r
    
    def return_Logistic_Coupling(self):
        return self.Logistic_Coupling
    
    def return_Armillotta_Poisson_Betas(self):
        return self.Armillotta_Poisson_Betas
    
    def return_Armillotta_Poisson_Lambda_0(self):
        return self.Armillotta_Poisson_Lambda_0
    
    def return_B(self):
        return self.B
    
    def return_TPR(self):
        return self.TPR
    
    def return_FPR(self):
        return self.FPR
    
    def return_AUC(self):
        return self.AUC
    
    def save_state(self):
        VersionNum = self.VersionNum
        
        Save_Dict = {}
        Save_Dict['VersionNum'] = self.VersionNum
        Save_Dict['T'] = self.T
        Save_Dict['n'] = self.n
        Save_Dict['X'] = self.X
        Save_Dict['Y'] = self.Y
        Save_Dict['Z'] = self.Z
        Save_Dict['XY'] = self.XY
        Save_Dict['NetworkAdjacency'] = self.NetworkAdjacency
        Save_Dict['NetworkType'] = self.NetworkType
        Save_Dict['Lin_Stoch_Gaussian_Adjacency'] = self.Lin_Stoch_Gaussian_Adjacency
        Save_Dict['Logistic_Adjacency'] = self.Logistic_Adjacency
        Save_Dict['Rho'] = self.Rho
        Save_Dict['Overall_Inference_Method'] = self.Overall_Inference_Method
        Save_Dict['InferenceMethod_oCSE'] = self.InferenceMethod_oCSE
        Save_Dict['KernelBandWidth'] = self.KernelBandWidth
        Save_Dict['KernelType'] = self.KernelType
        Save_Dict['KNN_K'] = self.KNN_K
        Save_Dict['Tau'] = self.Tau
        Save_Dict['Num_Shuffles_oCSE'] = self.Num_Shuffles_oCSE
        Save_Dict['Forward_oCSE_alpha'] = self.Forward_oCSE_alpha
        Save_Dict['Backward_oCSE_alpha'] = self.Backward_oCSE_alpha
        Save_Dict['Pedeli_PoissonAlphas'] = self.Pedeli_PoissonAlphas
        Save_Dict['Correlation_XY'] = self.Correlation_XY
        Save_Dict['Prctile'] = self.Prctile
        Save_Dict['Dict'] = self.Dict
        Save_Dict['Sinit'] = self.Sinit
        Save_Dict['Sfinal'] = self.Sfinal
        Save_Dict['DataType'] = self.DataType
        Save_Dict['Gaussian_Epsilon'] = self.Gaussian_Epsilon
        Save_Dict['Logistic_parameter_r'] = self.Logistic_parameter_r
        Save_Dict['Logistic_Coupling'] = self.Logistic_Coupling
        Save_Dict['Armillotta_Poisson_Betas'] = self.Armillotta_Poisson_Betas
        Save_Dict['Armillotta_Poisson_Lambda_0'] = self.Armillotta_Poisson_Lambda_0
        Save_Dict['Available_Inference_Methods'] = self.Available_Inference_Methods
        Save_Dict['font'] = self.font
        Save_Dict['B'] = self.B
        Save_Dict['TPR'] = self.TPR
        Save_Dict['FPR'] = self.FPR
        Save_Dict['AUC'] = self.AUC
        
        Date = datetime.datetime.today().strftime('%Y-%m-%d')
        F = glob.glob('NetworkInference_version_'+str(VersionNum)+'_Date_'+Date+'*.npz')
        num = len(F)+1
        
        np.savez('NetworkInference_version_'+str(VersionNum)+'_Date_'+Date+'_num_'+str(num),Save_Dict)
    
    def load_state(self,Date=None,Num=None,Version=None):
        if Date is None:
            #Default to today's date for attempt at loading
            Date = datetime.datetime.today().strftime('%Y-%m-%d')
         
        if Version is None:
            VersionNum = self.VersionNum    
         
        if Num is None:
            #Try finding the local file
            F = glob.glob('NetworkInference_version_'+str(VersionNum)+'_Date_'+Date+'*.npz')
            if len(F) == 0:
                raise ValueError("Sorry no NetworkInference file with date: ",Date, " can be found in the current directory, please specify the date!")
                
            else:
                #If the number is unspecified, then always choose the most recent (largests number)
                NumList = []
                for i in range(len(F)):
                    StringNum = F[i].split('_')[-1].split('.npz')[0]
                    NumList.append(int(StringNum))
                
                Num = max(NumList)
                if len(F)>1:
                    print("There were multiple files discovered for date: ",Date, " the most recent one was used by default")
                    print("If you wish to use a different filenumber please specify the file number to use")
        
        
        
        Filename = "NetworkInference_version_"+str(VersionNum)+"_Date_"+Date+"_num_"+str(Num)+".npz"
        npz = np.load(Filename,allow_pickle=True)
        Load_Dict = npz['arr_0'][()]
        #Now set all values to the loaded states
        self.set_T(Load_Dict['T'])
        self.set_n(Load_Dict['n'])
        self.set_X(Load_Dict['X'])
        self.set_Y(Load_Dict['Y'])
        self.set_Z(Load_Dict['Z'])
        self.set_XY(Load_Dict['XY'])
        self.set_NetworkAdjacency(Load_Dict['NetworkAdjacency'])
        self.set_NetworkType(Load_Dict['NetworkType'])
        self.set_Lin_Stoch_Gaussian_Adjacency(Load_Dict['Lin_Stoch_Gaussian_Adjacency'])
        self.set_Logistic_Adjacency(Load_Dict['Logistic_Adjacency'])
        self.set_Rho(Load_Dict['Rho'])
        self.set_Overall_Inference_Method(Load_Dict['Overall_Inference_Method'])
        self.set_InferenceMethod_oCSE(Load_Dict['InferenceMethod_oCSE'])
        self.set_KernelBandWidth(Load_Dict['KernelBandWidth'])
        self.set_KernelType(Load_Dict['KernelType'])
        self.set_KNN_K(Load_Dict['KNN_K'])
        self.set_Tau(Load_Dict['Tau'])
        self.set_Num_Shuffles_oCSE(Load_Dict['Num_Shuffles_oCSE'])
        self.set_Forward_oCSE_alpha(Load_Dict['Forward_oCSE_alpha'])
        self.set_Backward_oCSE_alpha(Load_Dict['Backward_oCSE_alpha'])
        self.set_Pedeli_PoissonAlphas(Load_Dict['Pedeli_PoissonAlphas'])
        self.set_Correlation_XY(Load_Dict['Correlation_XY'])
        self.set_Prctile(Load_Dict['Prctile'])
        self.set_Dict(Load_Dict['Dict'])
        self.set_Sinit(Load_Dict['Sinit'])
        self.set_Sfinal(Load_Dict['Sfinal'])
        self.set_DataType(Load_Dict['DataType'])
        self.set_Gaussian_Epsilon(Load_Dict['Gaussian_Epsilon'])
        self.set_Logistic_parameter_r(Load_Dict['Logistic_parameter_r'])
        self.set_Logistic_Coupling(Load_Dict['Logistic_Coupling'])
        self.set_Armillotta_Poisson_Betas(Load_Dict['Armillotta_Poisson_Betas'])
        self.set_Armillotta_Poisson_Lambda_0(Load_Dict['Armillotta_Poisson_Lambda_0'])
        self.set_Available_Inference_Methods(Load_Dict['Available_Inference_Methods'])
        self.set_font(Load_Dict['font'])
        self.set_B(Load_Dict['B'])
        self.set_TPR(Load_Dict['TPR'])
        self.set_FPR(Load_Dict['FPR'])
        self.set_AUC(Load_Dict['AUC'])
    
    def nchoosek(self,n,k):
        Range = range(1,n+1)
        comb = list(itertools.combinations(Range,k))
        return np.array(comb)
    
    def PermutationMatrix(self):
        p = int(self.n)
        Zp = int(p*(p-1)/2)
        P = np.zeros((p,Zp))
        NK = self.nchoosek(p,2)
        for i in range(p):
            Z = np.zeros(Zp)
            Wh = np.where(NK-1==i)
            Z[Wh[0]] = 1;
            P[i,:] = Z
        return P
    
    def sub2ind(self,Size,Arr):
        return np.ravel_multi_index(Arr,Size,order='F')
    
    def Logistic_Map(self,X,r):
        
        return r*X*(1-X)
        
    def Gen_Logistic_Dynamics(self,r=3.99,sigma=0.1):
        """Network coupled logistic map, r is the logistic map parameter
           and sigma is the coupling strength between oscillators"""
        
        if self.NetworkAdjacency is None:
            raise ValueError("Missing adjacency matrix, please add this using set_NetworkAdjacency")
        T = self.T
        self.Logistic_parameter_r = r
        self.Logistic_Coupling = sigma
        A = self.NetworkAdjacency
        #Must adjust the adjacency matrix so that dynamics stay in [0,1]
        A = A/np.sum(A,axis=1)
        A = A.T 
        A[np.isnan(A)] = 0
        A[np.isinf(A)] = 0
        #Since the row sums equal to 1 the Laplacian matrix is easy...
        L = np.eye(self.n)-A
        L = np.array(L)
        self.Logistic_Adjacency = A
        XY = np.zeros((T,self.n))
        XY[0,:] = np.random.rand(self.n)
        for i in range(1,T):
            XY[i,:] = self.Logistic_Map(XY[i-1,:],r)-sigma*np.dot(L,self.Logistic_Map(XY[i-1,:],r)).T
            
        self.XY = XY
         
        
    def Gen_Stochastic_Gaussian(self,Epsilon=1e-1):
        """Linear stochastic Gaussian process"""
        
        if self.NetworkAdjacency is None:
            raise ValueError("Missing adjacency matrix, please add this using set_NetworkAdjacency")
        
        if self.Rho is None:
            raise ValueError("Missing Rho, please set it using set_Rho")
        
        self.Gaussian_Epsilon = Epsilon
        R = 2*(np.random.rand(self.n,self.n)-0.5)
        A = np.array(self.NetworkAdjacency) * R
        A = A/np.max(np.abs(np.linalg.eigvals(A)))
        A = A*self.Rho
        self.Lin_Stoch_Gaussian_Adjacency = A
        XY = np.zeros((self.T,self.n))
        XY[0,:] = Epsilon*np.random.randn(1,self.n)
        for i in range(1,self.T):
            Xi = np.dot(A,np.matrix(XY[i-1,:]).T) + Epsilon*np.random.randn(self.n,1)
            XY[i,:] = Xi.T
        self.XY = XY
    
    def Gen_Poisson_Data(self,Epsilon=1,noiseLevel=1):
        if self.NetworkAdjacency is None:
            raise ValueError("Missing adjacency matrix, please add this using set_NetworkAdjacency")
        p = int(self.n)
        noiseLam = noiseLevel*np.ones((p,1))
        Zp = int(p*(p-1)/2)
        lambdas = Epsilon*np.ones((p+Zp,1))
        P = self.PermutationMatrix()
        One_p = np.ones((p,1))
        NK = self.nchoosek(p,2)
        #Size = (p,p)
        Tr = self.NetworkAdjacency[NK[:,0]-1,NK[:,1]-1]
        I_p = np.eye(p)
        T_p = np.dot(One_p,Tr)
        PTP = np.array(P)*np.array(T_p)
        B = np.concatenate((I_p,PTP),axis=1)
        B = B.T
        Lambs = lambdas.T*np.ones((self.T,Zp+p))
        Y = np.random.poisson(Lambs)
        
        Lambs = noiseLam.T*np.ones((self.T,p))
        E = np.random.poisson(Lambs)
        
        X = np.dot(Y,B)+E
        return X
        
    def Gen_Stochastic_Poisson_Pedeli(self,noiseLevel=0):
        """Stochastic Poisson process FROM THE PEDELI PAPER!!!!!"""
           
        if self.NetworkAdjacency is None:
            raise ValueError("Missing adjacency matrix, please add this using set_NetworkAdjacency")
        
        if self.Rho is None:
            raise ValueError("Missing Rho, please set it using set_Rho")
        
        
        p = self.n
        alpha = self.Rho*np.random.rand(p)
        self.PoissonAlphas = alpha
        X = np.zeros((self.T,self.n))
        X.astype('int32')
        T = self.T
        self.T = 1
        X[0,:] = self.Gen_Poisson_Data(noiseLevel=noiseLevel).astype('int32')
        
        for i in range(1,T):
            S = self.Gen_Poisson_Data(noiseLevel=noiseLevel)
            B = np.random.binomial(X[i-1,:].astype('int32'),alpha)
            X[i,:] = B.astype('int32')+S.astype('int32')
            
        self.XY = X
        self.T = T        
    
    def Gen_Stochastic_Poisson_Armillotta(self,Lambda_0=1,Betas=np.array([0.2,0.3,0.2])):
        """Stochastic Poisson process which is generally known as
           an INteger AutoRegressive (INAR) Process from the paper
           on Network INAR or in this case PNAR...   The name of the paper is:
               Poisson network autoregression
               and it is written by Armillotta and Fokianos"""
           
        if self.NetworkAdjacency is None:
            raise ValueError("Missing adjacency matrix, please add this using set_NetworkAdjacency")
        
        if len(Betas)!=3:
            raise ValueError("The Armillotta version of Stochastic Poisson only currently allows for exactly 3 beta values, and thus only allows Tau = 1...")
        beta_0, beta_1, beta_2 = Betas
        A = self.NetworkAdjacency
        self.Armillotta_Poisson_Betas= Betas
        self.Armillotta_Poisson_Lambda_0 = Lambda_0
        Y_init = np.random.poisson(Lambda_0,(self.n,1))
        #To ensure we don't get NaN's do below steps
        D = np.sum(A,axis=1)
        D[np.where(D==0)]=1
        C = A/D
        lam_0 = beta_0 + (beta_1*np.dot(C,Y_init)).T + beta_2*Y_init.T
        X = np.zeros((self.T,self.n))
        X[0,:] = np.random.poisson(lam_0)
        
        for i in range(1,self.T):
            
            lam_t = beta_0 + (beta_1*np.dot(C,X[i-1,:])) + beta_2*X[i-1,:]
            X[i,:] = np.random.poisson(lam_t)
        
        self.XY = X
       
       
       
    
    def Compute_CMI_Gaussian(self,X):
        """An implementation of the Gaussian conditional mutual information from
        the paper by Sun, Taylor and Bollt entitled: 
            Causal network inference by optimal causation entropy"""
        TupleX = X.shape
        TupleY = self.Y.shape
        if len(TupleX)==1:
            SX = [[1]]
        
        else:
            if TupleX[1] == 1:
                SX = [[1]]
            else:
                SX = np.corrcoef(X.T)
        
        if len(TupleY) ==1:
            SY = [[1]]
        else:
            if TupleY[1] == 1:
                SY = [[1]]
            else:
                SY = np.corrcoef(self.Y.T)
        
        SX = np.linalg.det(SX)
        SY = np.linalg.det(SY)        
        if self.Z is None:

            SXY = np.linalg.det(np.corrcoef(X.T,self.Y.T))
            
            return 0.5*np.log((SX*SY)/SXY)
            
        else:
            TupleZ = self.Z.shape
            if len(TupleZ)==1:
                SZ = [[1]]
            else:
                if TupleZ[1]==1:
                    SZ = [[1]]
                else:
                    SZ = np.corrcoef(self.Z.T)
            
            SZ = np.linalg.det(SZ)
            XZ = np.concatenate((X,self.Z),axis=1)
            YZ = np.concatenate((self.Y,self.Z),axis=1)
            XYZ = np.concatenate((X,self.Y,self.Z),axis=1)
            
            SXZ = np.linalg.det(np.corrcoef(XZ.T))
            SYZ = np.linalg.det(np.corrcoef(YZ.T))
            SXYZ = np.linalg.det(np.corrcoef(XYZ.T))
            
            return 0.5*np.log((SXZ*SYZ)/(SZ*SXYZ))
                
       
    def Compute_CMI_Hawkes(self,X):
        Blah = 1
    
    def Compute_CMI_VonMises(self,X):
        Blah = 1
    
    def Compute_CMI_Laplace(self,X):
        Blah = 1
    
    def Compute_CMI_Histogram(self,X):
        Blah = 1
    
    def Compute_CMI_KernelDensity(self,X):
        
        if self.KernelBandWidth is None:
            #This is approximately the optimal bandwidth according to
            #the rule of thumb in "Optimal Bandwidth Selection for Kernel Density Functionals Estimation"
            #By Su Chen for the Guassian Kernal
            BandWidth = 1.515717*self.T**(-2/5)
            if self.DataType == 'Discrete':
                BandWidth = 1+BandWidth
        else:
            BandWidth = self.KernelBandWidth
        
        kde = KernelDensity(bandwidth=BandWidth,kernel=self.KernelType).fit(X)
    
        
    def MutualInfo_KNN(self,X,Y):
        """The Max Norm version of the Kraskov, Stogbauer, Grassberger (KSG) K-
        nearest neighbors mutual information from the paper: 
            Estimating mutual information"""
        if self.KNN_K is None:
            print("Warning KNN_K was set to None, for KNN the default is k=10")
            print()
            print("If you wish to change this behavior please manually set using set_KNN_K")
            self.KNN_K=10 
            
        k = int(self.KNN_K)
        N = X.shape[0]
        DistX = distance.pdist(X)
        DistY = distance.pdist(Y)
        DistXY = np.zeros(len(DistX))
        Wh1 = np.where((DistX-DistY)>=0)
        Wh2 = np.where((DistX-DistY)<0)
        DistXY[Wh1] = DistX[Wh1]
        DistXY[Wh2] = DistY[Wh2]
        DistX = distance.squareform(DistX)
        DistY = distance.squareform(DistY)
        DistXY = distance.squareform(DistXY)
        AS = DistXY.argsort(axis=0)[k]
        Term1 = Digamma(N)+Digamma(k)
        Inner2X = np.sum(DistX<DistXY[AS[np.arange(N)],[np.arange(N)]].reshape(N,1),axis=0)+1
        Inner2Y = np.sum(DistY<DistXY[AS[np.arange(N)],[np.arange(N)]].reshape(N,1),axis=0)+1
        Term2 = np.mean(Digamma(Inner2X)+Digamma(Inner2Y))
        
        return Term1-Term2
        
    def Compute_CMI_KNN(self,X):
        """KNN version of Conditional mutual information for Causation
        entropy..."""
        if self.Z is None:
            return np.max([self.MutualInfo_KNN(X,self.Y),0])
        else:
            XcatY = np.concatenate((X,self.Y),axis=1)
            MIXYZ = self.MutualInfo_KNN(XcatY,self.Z)
            MIXZ = self.MutualInfo_KNN(X,self.Z)
           
            return np.max([MIXYZ-MIXZ,0])        
 

    def Entropy_GKNN(self,X):
        """An implementation of Geometric KNN from Lord, Sun and Bollt's paper:
            Geometric k-nearest neighbor estimation of entropy and mutual information """
        if self.KNN_K is None:
            print("Warning KNN_K was set to None, for GeometricKNN the default is k=20")
            print()
            print("If you wish to change this behavior please manually set using set_KNN_K")
            self.KNN_K=20
        k = int(self.KNN_K)
        d = X.shape[1]
        N = X.shape[0]
        #The first "term" (really this is multiple terms) is easy to calculate
        Term1 = np.log(N) + np.log(np.pi**(d/2))-np.log(Gamma(1+(d/2)))
        #Find the distance between points (this is the heavy memory portion)
        DF = distance.squareform(distance.pdist(X))
        AS = DF.argsort(axis=0)[k]
        
        #Geometric KNN requires finding the number of points inside the ellipsoid
        #KforMean is the variable those k values will be stored in
        KforMean = np.zeros(N)
        #The size of the epsilon ball for each point....
        Epsilons = DF[AS[np.arange(N)],np.arange(N)]
        
        Term3 = np.zeros(N)
        for i in range(N):
            xi = np.matrix(X[i,:])
            #Find the k nearest neighbors to point i
            Wh = np.where(DF[i,:]<=DF[AS[i],i])
            Xi = X[Wh[0],:]
            
            #print(xi.shape,Xi.shape)
            #First center the SVD on the centroid of the k nearest neighbors
            Z = np.mean(Xi,axis=0)
            Yi = Xi-Z
            
            U,Sigma,V = np.linalg.svd(Yi)
            
            #Scale the singular values for use inside the epsilon ball
            SS = Sigma/np.max(Sigma)
            #Now project onto the axis determined by V and recenter to the 
            #ith datapoint...
            Zi = Xi-xi
            V = np.matrix(V.T)
            Zi = (Zi*V)
            print("This is Xi: ", Xi)
            print("This is Xi*V: ",Xi*V)
            #Re-scale...
            SSE = SS*Epsilons[i]
           
            Zi = (Zi.T/SSE[:,np.newaxis]).T#np.divide(Zi,(SS*Epsilons[i]))
            #print(np.sum(np.isinf(Zi)))
            Zi[np.isinf(Zi)]=0
            #Now determine which points are inside the ellipsoid. This can be
            #done using the equation of an ellipse (x^2/a^2 + y^2/b^2 <1 implies
            #                                       it is inside the ellipse)
            Zi = np.square(Zi)
            SumZi = np.sum(Zi,axis=1)
            #How many of the k neighbors are inside the ellipse...
            Wh2 = np.where(SumZi<=1)
            KforMean[i] = np.max([len(Wh2[0]),1])
            #This is the final terms combined from the paper
            Log = np.log(SS*Epsilons[i])
            Log[np.isinf(Log)] = 0
            Term3[i] = np.sum(Log)
        
        Log2 = np.log(KforMean)
        Log2[np.isinf(Log2)] = 0
        Term2 = np.mean(Log2)
        Term3 = np.mean(Term3)
        #print("Here is term1,3 and 2: ",Term1,Term3,Term2)
        return Term1-Term2+Term3
    
    def Compute_CMI_Geometric_KNN(self,X):
        """Geometric KNN version of Conditional mutual information for Causation
        entropy..."""
        if self.Z is None:
            XcatY = np.concatenate((X,self.Y),axis=1)
            return np.max([self.Entropy_GKNN(X)+self.Entropy_GKNN(self.Y)-self.Entropy_GKNN(XcatY),0])
        else:
            XcatZ = np.concatenate((X,self.Z),axis=1)
            YcatZ = np.concatenate((self.Y,self.Z),axis=1)
            XYcatZ = np.concatenate((X,self.Y,self.Z),axis=1)
            HZ = self.Entropy_GKNN(self.Z)
            HXZ = self.Entropy_GKNN(XcatZ)
            HYZ = self.Entropy_GKNN(YcatZ)
            HXYZ = self.Entropy_GKNN(XYcatZ)
            return np.max([HXZ+HYZ-HXYZ-HZ,0])
            
    def Compute_CMI_NegativeBinomial(self,X):
        Blah = 1
    
    def PoissEnt(self,Lambdas):
        
        Lambdas = np.abs(Lambdas)
        First = np.exp(-Lambdas)
        Psum = First
        P = [np.matrix(First)]
        counter = 0
        small = 1
        i = 1
        while np.max(1-Psum)>1e-16 and small>1e-75:
            counter = counter+1
            prob = SStats.poisson.pmf(i,Lambdas)
            Psum = Psum+prob
            P.append(np.matrix(prob))
            if i >=np.max(Lambdas):
                small = np.min(prob)
            
            i = i+1
        
        P = np.array(P).squeeze()
        est_a = P*np.log(P)
        est_a[np.isinf(est_a)]=0
        est_a[np.isnan(est_a)]=0
        try:
            est = -np.sum(est_a,axis=0)
        except:
            est = -np.sum(est_a)
        return np.real(est)
    
    def PoissonJointEntropy_New(self,Cov):
        T = np.triu(Cov,1)
        T = np.matrix(T)
        U = np.matrix(np.diag(Cov))
        Ent1 = np.sum(self.PoissEnt(U))
        Ent2 = np.sum(T)
        return Ent1+Ent2
    
    def Compute_CMI_Poisson(self,X):
        """Estimate of conditional mutual information from Poisson marginals,
        from the paper by Fish, Sun and Bollt entitled: 
            Interaction Networks from Discrete Event Data by Poisson Multivariate 
            Mutual Information Estimation and Information Flow with Applications 
            from Gene Expression Data"""
        
        if self.Z is None:
            SXY = np.corrcoef(X.T,self.Y.T)
            l_est = SXY - np.diag(np.diag(SXY))
            np.fill_diagonal(SXY,np.diagonal(SXY)-np.sum(l_est,axis=0))
            Dcov = np.diag(SXY)+np.sum(l_est,axis=0)
            TF = self.PoissonJointEntropy_New(SXY)
            FT = np.sum(self.PoissEnt(Dcov))
            #print(FT-TF)
           
            return FT-TF
        else:
            SzX = X.shape[1]
            SzY = self.Y.shape[1]
            SzZ = self.Z.shape[1]
            indX = np.matrix(np.arange(SzX))
            indY = np.matrix(np.arange(SzY)+SzX)
            indZ = np.matrix(np.arange(SzZ)+SzX+SzY)
            #print(indX,indY,indZ)
            XYZ = np.concatenate((X,self.Y,self.Z),axis=1)
            SXYZ = np.corrcoef(XYZ.T)
            SS = SXYZ
            Sa = SXYZ-np.diag(np.diag(SXYZ))
            np.fill_diagonal(SS,np.diagonal(SS)-Sa)
            SS[0:SzX,0:SzX] = SS[0:SzX,0:SzX]+SXYZ[0:SzX,SzX:SzX+SzY]
            SS[SzX:SzX+SzY,SzX:SzX+SzY] = SS[SzX:SzX+SzY,SzX:SzX+SzY]+SXYZ[SzX:SzX+SzY,0:SzX]
            S_est1 = SS[np.concatenate((indY.T,indZ.T),axis=0),np.concatenate((indY.T,indZ.T),axis=0)]
            S_est2 = SS[np.concatenate((indX.T,indZ.T),axis=0),np.concatenate((indX.T,indZ.T),axis=0)]
            HYZ = self.PoissonJointEntropy_New(S_est1)
            SindZ = SS[indZ,indZ]
            HZ = self.PoissonJointEntropy_New(SindZ)
            HXYZ = self.PoissonJointEntropy_New(SXYZ-np.diag(Sa))
            HXZ = self.PoissonJointEntropy_New(S_est2)
            H_YZ = HYZ-HZ
            H_XYZ = HXYZ-HXZ
            return H_XYZ-H_YZ
    def Compute_CMI(self,X):
        """Compute the CMI based upon whichever method"""
        if self.InferenceMethod_oCSE == 'Gaussian':
            return self.Compute_CMI_Gaussian(X)
        
        elif self.InferenceMethod_oCSE == 'KernelDensity':
            return self.Compute_CMI_KernelDensity(X)
        
        elif self.InferenceMethod_oCSE == 'KNN':
            return self.Compute_CMI_KNN(X)
        
        elif self.InferenceMethod_oCSE == 'Poisson':
            return self.Compute_CMI_Poisson(X)
        
        elif self.InferenceMethod_oCSE == 'GeometricKNN':
            return self.Compute_CMI_Geometric_KNN(X)
        
        elif self.InferenceMethod_oCSE == 'Histogram':
            return self.Compute_CMI_Histogram(X)
        
        elif self.InferenceMethod_oCSE == 'Laplace':
            return self.Compute_CMI_Laplace(X)
        
        elif self.InferenceMethod_oCSE == 'NegativeBinomial':
            return self.Compute_CMI_NegativeBinomial(X)
        
        elif self.InferenceMethod_oCSE == 'VonMises':
            return self.Compute_CMI_VonMises(X)
        
        elif self.InferenceMethod_oCSE == 'Hawkes':
            return self.Compute_CMI_Hawkes(X)
        
        else:
            raise ValueError("Sorry oCSE inference method: ", self.InferenceMethod_oCSE, "is not available.")
    
    def Standard_Forward_oCSE(self):
        NotStop = True
        
        n = self.X.shape[1]
        TestVariables = np.arange(n)
        S = [] 
        loopnum = 0
        while NotStop:
            loopnum = loopnum+1
            #print(loopnum)
            SetCheck = np.setdiff1d(TestVariables,S)
            m = len(SetCheck)
            if m == 0:
                NotStop = False
                break
            Ents = np.zeros(m)
            for i in range(m):
                X = self.X[:,[SetCheck[i]]]
                
                Ents[i] = self.Compute_CMI(X)
            #print(self.Z,SetCheck[i])
            Argmax = Ents.argmax()
            X = self.X[:,[SetCheck[Argmax]]]
            Dict = self.Standard_Shuffle_Test_oCSE(X,Ents[Argmax],self.Forward_oCSE_alpha)
            #print(Dict)
            if Dict['Pass']:
                S.append(SetCheck[Argmax])
                if len(S)==1:
                    self.Z = self.X[:,[SetCheck[Argmax]]]
                else:
                    self.Z = np.concatenate((self.Z,self.X[:,[SetCheck[Argmax]]]),axis=1)
            else:
                NotStop = False
        
        return S
    
    def Alternative_Forward_oCSE(self):
        NotStop = True
        
        n = self.X.shape[1]
        TestVariables = np.arange(n)
        S = [] 
        loopnum = 0
        while NotStop:
            loopnum = loopnum+1
            #print(loopnum)
            SetCheck = np.setdiff1d(TestVariables,S)
            m = len(SetCheck)
            if m == 0:
                NotStop = False
                break
            Ents = np.zeros(m)
            Passes = np.zeros(m,dtype=bool)
            for i in range(m):
                X = self.X[:,[SetCheck[i]]]
                
                Ents[i] = self.Compute_CMI(X)
                Dict = self.Standard_Shuffle_Test_oCSE(X,Ents[i],self.Forward_oCSE_alpha)
                Passes[i] = Dict['Pass']
            
            NewEnts = Ents[Passes]
            if len(NewEnts)>0:
                Wh = np.where(Passes)
                Wh = Wh[0]
                Argmax = Wh[NewEnts.argmax()]
                S.append(SetCheck[Argmax])
                if len(S)==1:
                    self.Z = self.X[:,[SetCheck[Argmax]]]
                else:
                    self.Z = np.concatenate((self.Z,self.X[:,[SetCheck[Argmax]]]),axis=1)
            else:
                NotStop = False                
            
        return S    
        
    def Standard_Backward_oCSE(self,S):
        #Reset the conditioning matrix
        self.Z = None
        RP = np.random.permutation(len(S))
        Sn = copy.deepcopy(S)
        for i in range(len(S)):
            X = self.X[:,[S[RP[i]]]]
            
            Inds = np.setdiff1d(Sn,S[RP[i]])
            #print(Inds)
            Z = self.X[:,Inds]
            #if len(Inds)<=1:
            #    Z = self.X[:,[S[Inds]]]
            #else:
            #    Z = self.X[:,S[Inds]]
            self.Z = Z
            if Z.shape[1]==0:
                self.Z = None
            Ent = self.Compute_CMI(X)
            Dict = self.Standard_Shuffle_Test_oCSE(X,Ent,self.Backward_oCSE_alpha)
            if not Dict['Pass']:
                Sn = np.setdiff1d(Sn,S[RP[i]])
            
        return Sn
            
        
    def Standard_Shuffle_Test_oCSE(self,X,Ent,alpha):
        ns = self.Num_Shuffles_oCSE
        T = X.shape[0]
        Ents = np.zeros(ns)
        TupleX = X.shape
        for i in range(ns):
            RP = np.random.permutation(T)
            if len(TupleX)>1:
                Xshuff = X[RP,:]
            else:
                Xshuff = X[RP]
            Ents[i] = self.Compute_CMI(Xshuff)
        Prctile = int(100*np.floor(ns*(1-alpha))/ns)
        self.Prctile = Prctile
        #print(Ents)
        #print(Ents[Ents>=np.percentile(Ents,Prctile)])
        Threshold = np.min(Ents[Ents>=np.percentile(Ents,Prctile)])
        Dict ={'Threshold':Threshold}
        Dict['Value'] = Ent
        Dict['Pass'] = False
        if Ent>=Threshold:
            Dict['Pass'] = True
        
        self.Dict = Dict
        return Dict
    
    def Standard_oCSE(self):
        
        """Run the standard version of the oCSE algorithm. Note defaults to the
           KernelDensity plugin estimator if the method is not specified"""
        if self.X is None:
            raise ValueError("Missing the potential predictors please add this using set_X")
        
        if self.Y is None:
            raise ValueError("Missing the target(s) please add using set_Y")
        
        #Ensure initially Z is None
        self.Z = None
        
        #Set this to avoid potential issues
        self.Correlation_XY = None
        
        #Find the initial set of potential predictors
        S = self.Standard_Forward_oCSE()
        self.Sinit = S
        
        #The final set of predictors after removing spurious edges
        S = self.Standard_Backward_oCSE(S)
        
        self.Sfinal = S
        
        #Reset conditioning set in case other methods need it
        self.Z = None
        
        return S
    
    def Alternative_oCSE(self):
        
        """Run the standard version of the oCSE algorithm. Note defaults to the
           KernelDensity plugin estimator if the method is not specified"""
        if self.X is None:
            raise ValueError("Missing the potential predictors please add this using set_X")
        
        if self.Y is None:
            raise ValueError("Missing the target(s) please add using set_Y")
        
        #Ensure initially Z is None
        self.Z = None
        
        #Set this to avoid potential issues
        self.Correlation_XY = None
        
        #Find the initial set of potential predictors
        S = self.Alternative_Forward_oCSE()
        self.Sinit = S
        
        #The final set of predictors after removing spurious edges
        S = self.Standard_Backward_oCSE(S)
        
        self.Sfinal = S
        
        #Reset conditioning set in case other methods need it
        self.Z = None
        
        return S
    
    def Estimate_Network(self):
        #Initialize...
        self.Y = None
        self.Z = None
        self.X = None
        Method = self.Overall_Inference_Method
        MethodList = self.Available_Inference_Methods
        if Method not in MethodList:
            raise ValueError("Sorry the Method: ", Method, " is not currently implemented, the only available methods are: ", self.Available_Inference_Methods)
        
        if Method == 'Standard_oCSE':
            XY = self.XY
            XY_1 = XY[0:self.T-self.Tau,:]
            XY_2 = XY[self.Tau:,:]
            B = np.zeros((self.n,self.n))
            for i in range(self.n):
                print("Estimating edges for node number: ", i)
                self.Y = XY_2[:,[i]]
                self.X = XY_1
                S = self.Standard_oCSE()
                B[i,S] = 1
                
        elif Method=='Alternative_oCSE':
            XY = self.XY
            XY_1 = XY[0:self.T-self.Tau,:]
            XY_2 = XY[self.Tau:,:]
            B = np.zeros((self.n,self.n))
            for i in range(self.n):
                print("Estimating edges for node number: ", i)
                self.Y  = XY_2[:,[i]]
                self.X = XY_1
                S = self.Alternative_oCSE()
                B[i,S] = 1
                
        else:
            #Do nothing
            B = []
        
        self.B = B
        return B
    
    def Compute_TPR_FPR(self):
        if self.NetworkAdjacency is None:
            raise ValueError("Sorry cannot compute error if NetworkAdjacency is not specified, please set using set_NetworkAdjacency")
        
        if self.B is None:
            raise ValueError("Sorry cannot compute error if B (estimated network) is not specified, please set using set_B")
        
        A = self.NetworkAdjacency
        B = self.B
        n = self.n
        TPR = 1 - (np.sum((A-B)>0))/np.sum(A)
        FPR= np.sum((A-B)<0)/(n*(n-1)-np.sum(A))
        self.TPR = TPR
        self.FPR = FPR
        return (TPR,FPR)
    
    def Compute_AUC(self,TPRs, FPRs):
        """Estimate the area under the curve (AUC) using the trapezoidal rule
           for integration..."""
           
        AUC = np.trapz(TPRs,FPRs)
        self.AUC = AUC
        return AUC
    
    def Plot_ROC(self,TPRs,FPRs):
        if self.font is None:
            font = {'family' : 'normal',
                    'weight' : 'bold',
                    'size'   : 22}
        else:
            font = self.font
        rc('font', **font)
        
        plt.plot(FPRs,TPRs)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC')
        plt.xlim(0,1)
        plt.ylim(0,1)
        
        AUC = self.Compute_AUC(TPRs,FPRs)
        plt.text(0.4,0.1,'AUC = '+format(AUC,'.4f'))
        
