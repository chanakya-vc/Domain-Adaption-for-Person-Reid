import numpy as np
import torch
import scipy.io as sio
import time
from gradient_optim import gradient_optim
from laplacian_matrix import laplacian_matrix
"""
    The following is a function f that contains the constraints for the objective to be optimizied
    mathematically defined as f(Z,Q,E,W,X,A,mu,Y1,Y2)= (mu/2)(||WX-WXZ-E+(Y_1/mu)||_F^{2} + ||Z-Q+(Y_2/mu)||_F^{2})
"""
def grad_f(Z,Q,E,W,X,A,mu,Y1,Y2):
    projection=torch.mm(W,X)
    abs1=projection-torch.mm(projection,Z)-E+Y1/mu
    abs2=Z-Q+Y2/mu
    return (-mu)*(torch.mm(torch.transpose(projection,0,1),abs1)-abs2)

"""
    Computes the laplacian matrix for a given graph L_A= D_A -A where D_A is the degree matrix of A
"""
def laplacian_matrix(A):
    D_A=torch.zeros(A.size()[0],A.size()[0])
    for i in range(A.size()[0]):
        D_A[i][i]=A[i][:].nonzero().size()[0] # finds the number of outgoing egdes from the node aka its degree.
    return D_A - A


"""
    soft thresholding operator with definiton
    \hat{f(y)}= y+ \tau if y<-\tau
             0       if -\tau<y<\tau
             y- \tau if y>\tau
"""
def L21_operator(y,t):
    aux=torch.max(torch.abs(y)-t,torch.zeros_like(y))
    aux[aux!=0]=1
    return torch.mul(aux,y)
"""
    L21 minimization for the proximal operator  for the update rule of error matrix E
"""
def soft_operator(y,t):
    return torch.sign(y)*torch.max(torch.abs(y)-t,torch.zeros_like(y))
"""
    Functions u_vector and update_A are for the update rule of A which is row wise updated.
"""
def u_vector(Q,parameters):
    u=torch.zeros(Q.size()[0],Q.size()[0])
    for i in range(Q.size()[0]):
        for j in range(Q.size()[0]):
            u[i][j]=(torch.norm(Q[i]-Q[j]))*(parameters['gamma']/(2*parameters['lambda']))
    return u
def update_A(A,Q,parameters):
    u = u_vector(Q,parameters)
    for i in range(Q.size()[0]):
        res=0
        for j in range(parameters['epsilon']):
            res= res + u[i][j]
        
        A[i]=torch.max(((1+res)/parameters['epsilon'])-u[i],torch.zeros_like(u[i]))
    return A

"""
    The following is a conversion function for the convex optimization procedure being employed
    for building the dynamic affinity matrix A.
    Input: LOMO feature matrix X ( should be reduced to an appropriate dimension using PCA)
    Intermediate Variables: k (denotes the number of iterations)
                            parameters {dimension of W:,alpha:,beta:,gamma:,lambda3:,lambda4:,tau:,epsilon:,eta:} //dict with given entries             
                            Set Z = Q = A = Y2 = 0, E = Y1 = 0.                           
                            mu=0.1, mu_max=10**10, rho=1.1                                  
    Output: Z, Q, A, W, E
    form of objective is alpha||Z||_1 + beta||E||_{2,1} + gamma tr(QL_AQ^{T}) + lambda ||A||_{F} + thetha ||W||_{2,1}
"""
def convex_update(X,parameters):
    Z = Q = A = Y2 = torch.zeros(X.shape[1], X.shape[1])
    W = torch.ones(parameters['dimension of W'],X.shape[0])/X.shape[0]
    E = Y1 = torch.zeros(parameters['dimension of W'],X.shape[1])
    mu, mu_max, rho = 0.1, 10**10, 1.1
    for k in range(100):
        #update Z
        constraint = grad_f(Z,Q,E,W,X,A,mu,Y1,Y2)
        P = Z - parameters['tau']*constraint
        Z = soft_operator(P,parameters['alpha']/(parameters['eta']*mu))
        #update Q
        L_A=laplacian_matrix(A)
        part1 = (Z + Y2/mu)
        part2 = torch.inverse( torch.eye(X.shape[1]) + parameters['gamma']*(L_A+torch.transpose(L_A,0,1)))
        Q = torch.mm(part1,part2)
        #update E
        #aux = torch.mm(W,X)-torch.mm(torch.mm(W,X),Z)-E+Y1/mu
        #E = soft_operator(aux,parameters['beta']/mu)
        aux = torch.mm(W,X)-torch.mm(torch.mm(W,X),Z)+Y1/mu
        E=(mu/(parameters['beta']+mu))*aux
        #update A
        A = update_A(A,Q,parameters)
        #update W
        W=gradient_optim(Z,Q,E,W,X,A,mu,Y1,Y2,parameters['theta'])
        #update Y1 and Y2
        #Y1 = mu*aux
        Y1 = Y1-mu*(torch.mm(W,X)-torch.mm(torch.mm(W,X),Z)-E)
        Y2 = Y2-mu*(Z-Q)
        #update mu
        mu=min(mu_max,rho*mu)
        parameters['tau']=parameters['tau']/1.1
        print("print k Z and Q",k,Z,Q)
    tensor_list=[Z,Q,A,W,X,E,Y1,Y2]
    return tensor_list
#All the functions end here calling begins
parameters={'dimension of W': 200,'alpha': 1,'beta':0.0000000001,'gamma':0.001,'theta':0.001,'lambda':0.1,'tau':0.001,'epsilon':60,'eta':0.1}
features = sio.loadmat('../features_pca.mat')
ft = features['X']
X = torch.tensor(ft,dtype=torch.float)
tensor_list = convex_update(X,parameters)
a=[]
for item in tensor_list:
    a.append(item.data.numpy())
np.save('../../meta/data/pca/Z_Q_A_W_X_E_Y1_Y2_100.npy',np.array(a))
