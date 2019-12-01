import numpy as np
import torch
import scipy.io as sio
import time
#import matplotlib
#import sklearn 
"""
    The following is a function f that contains the constraints for the objective to be optimizied
    mathematically defined as f(Z,Q,E,W,mu,Y_1,Y_2)= (mu/2)(||WX-WXZ-E+(Y_1/mu)||_F^{2} + ||Z-Q+(Y_2/mu)||_F^{2})
"""
def f(Z,Q,E,W,X,mu,Y1,Y2):
    projection=torch.mm(W,X)
    abs1=projection-torch.mm(projection,Z)-E+Y1/mu
    abs2=Z-Q+Y2/mu
    return (mu/2)*(torch.sum(abs1**2)+torch.sum(abs2**2))
"""
    Computes the laplacian matrix for a given graph L_A= D_A -A where D_A is the degree matrix of A
"""
def laplacian_matrix(A):
    D_A=torch.zeros(A.shape[0],A.shape[0], dtype=torch.double)
    for i in range(A.shape[0]):
        D_A[i][i]=A[i][:].nonzero().size()[0] # finds the number of outgoing egdes from the node aka its degree.
    return D_A - A



"""
    The following is a conversion function for the convex optimization procedure being employed
    for building the dynamic affinity matrix A.
    Input: LOMO feature matrix X ( should be reduced to an appropriate dimension using PCA)
    Intermediate Variables: k (denotes the number of iterations)
                            parameters {dimension of W:,alpha:,beta:,gamma:,theta:,lambda:,tau: } //dict with given entries             
                            Set Z = Q = A = Y2 = 0, E = Y1 = 0.                           
                            mu=0.1, mu_max=10**10, rho=1.1                                  
    Output: Z, Q, A, W, E 
"""
def convex_update(X,Z,Q,A,W,E,Y1,Y2,parameters):
    mu, mu_max, rho = 0.1, 10**10, 1.1
    for k in range(100):
        #update Z
        constraint = f(Z,Q,E,W,X,mu,Y1,Y2)
        L_A = laplacian_matrix(A)
        prod = torch.mm(Q,torch.mm(L_A,torch.transpose(Q,0,1)))
        loss = parameters['alpha']*torch.sum(Z**2) + parameters['beta']*torch.sum(E**2) + parameters['gamma']*torch.trace(prod) + parameters['lambda']*torch.sum(W**2) + parameters['theta']*torch.sum(A**2) + constraint
        print('loss: ',loss)
        loss.backward()
        with torch.no_grad():
            Z = Z - parameters['tau']*Z.grad
            Q = Q - parameters['tau']*Q.grad
            W = W - parameters['tau']*W.grad
            A = A - parameters['tau']*A.grad
            E = E - parameters['tau']*E.grad
            Y1= Y1- mu*(torch.mm(W,X)-torch.mm(torch.mm(W,X),Z)-E)
            Y2= Y2- mu*(Z-Q)
        #update mu
        mu=min(mu_max,rho*mu)
        #requires gradients
        Z.requires_grad = True
        Q.requires_grad = True
        W.requires_grad = True
        A.requires_grad = True
        E.requires_grad = True
    return Z,Q,A,W,E,Y1,Y2

    #stacked_tensor = torch.stack(tensor_list)
    #return stacked_tensor
#All the functions end here calling begins
parameters={'dimension of W': 200,'alpha': 0.1,'beta':0.1,'gamma':0.1,'theta':0.1,'lambda':0.1,'tau':0.01}
features = sio.loadmat('../features_pca.mat')
ft = features['X']
X = torch.tensor(ft,dtype=torch.double)
Z = torch.zeros(X.shape[1], X.shape[1], dtype=torch.double, requires_grad=True)
Q = torch.zeros(X.shape[1], X.shape[1], dtype=torch.double, requires_grad=True)
A =torch.zeros(X.shape[1], X.shape[1], dtype=torch.double, requires_grad=True)
W = torch.zeros(parameters['dimension of W'],X.shape[0],dtype=torch.double,requires_grad=True)
E =  torch.zeros(parameters['dimension of W'],X.shape[1],dtype=torch.double,requires_grad=True)
Y2 = torch.zeros(X.shape[1], X.shape[1], dtype=torch.double)
Y1 = torch.zeros(parameters['dimension of W'],X.shape[1],dtype=torch.double)
print('print X',X)
Z,Q,A,W,E,Y1,Y2 = convex_update(X,Z,Q,A,W,E,Y1,Y2,parameters)
print('print Z',Z)
print('print Q',Q)
print('print W',W)
print('print E',E)
"""
es = res.detach.numpy()
print(es)
np.save(es,'../../meta/data/pca/es.npy')
"""
