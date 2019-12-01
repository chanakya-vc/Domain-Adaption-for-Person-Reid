import numpy as np
import scipy.io as sio
import time
from gradient_optim_np import gradient_optim
from scipy.sparse import csgraph
"""
    The following is a function f that contains the constraints for the objective to be optimizied
    mathematically defined as f(Z,Q,E,W,X,A,mu,Y1,Y2)= (mu/2)(||WX-WXZ-E+(Y_1/mu)||_F^{2} + ||Z-Q+(Y_2/mu)||_F^{2})
"""
def grad_f(Z,Q,E,W,X,A,mu,Y1,Y2):
    projection=np.matmul(W,X)
    abs1=projection-np.matmul(projection,Z)-E+Y1/mu
    abs2=Z-Q+Y2/mu
    return (-mu)*(np.matmul(np.transpose(projection),abs1)-abs2)


"""
    soft thresholding operator with definiton
    \hat{f(y)}= y+ \tau if y<-\tau
             0       if -\tau<y<\tau
             y- \tau if y>\tau
"""
def L21_operator(y,t):
    aux=np.max(np.abs(y)-t,np.zeros_like(y))
    aux[aux!=0]=1
    return np.multiply(aux,y)
"""
    L21 minimization for the proximal operator  for the update rule of error matrix E
"""
def soft_operator(y,t):
    return np.sign(y)*np.maximum(np.abs(y)-t,np.zeros((y.shape[0],y.shape[1])))
"""
    Functions u_vector and update_A are for the update rule of A which is row wise updated.
"""
def u_vector(Q,parameters):
    u=np.zeros((Q.shape[0],Q.shape[0]))
    for i in range(Q.shape[0]):
        for j in range(Q.shape[0]):
            u[i][j]=(np.linalg.norm(Q[i]-Q[j]))*(parameters['gamma']/(2*parameters['lambda']))
    return u
def update_A(A,Q,parameters):
    u = u_vector(Q,parameters)
    for i in range(Q.shape[0]):
        res=0
        for j in range(parameters['epsilon']):
            res= res + u[i][j]
        
        A[i]=np.maximum(((1+res)/parameters['epsilon'])-u[i],np.zeros_like(u[i]))
    return A

"""
    The following is a conversion function for the convex optimization procedure being employed
    for building the dynamic affinity matrix A.
    Input: LOMO feature matrix X ( should be reduced to an appropriate dimension using PCA)
    Intermediate Variables: k (denotes the number of iterations)
                            parameters {dimension of W:,alpha:,beta:,gamatmula:,lambda3:,lambda4:,tau:,epsilon:,eta:} //dict with given entries             
                            Set Z = Q = A = Y2 = 0, E = Y1 = 0.                           
                            mu=0.1, mu_max=10**10, rho=1.1                                  
    Output: Z, Q, A, W, E
    form of objective is alpha||Z||_1 + beta||E||_{2,1} + gamatmula tr(QL_AQ^{T}) + lambda ||A||_{F} + thetha ||W||_{2,1}
"""

parameters={'dimension of W': 200,'alpha': 0.1,'beta':0.0000000001,'gamma':0.001,'theta':0.001,'lambda':0.1,'tau':0.001,'epsilon':60,'eta':0.1}
def convex_update(X,parameters=parameters):
    Z = Q = A = Y2 = np.zeros((X.shape[1], X.shape[1]))
    W = np.ones((parameters['dimension of W'],X.shape[0]))/X.shape[0]
    E = Y1 = np.zeros((parameters['dimension of W'],X.shape[1]))
    mu, mu_max, rho = 0.1, 10**10, 1.1
    for k in range(100):
        #update Z
        constraint = grad_f(Z,Q,E,W,X,A,mu,Y1,Y2)
        P = Z - parameters['tau']*constraint
        Z = soft_operator(P,parameters['alpha']/(parameters['eta']*mu))
        #update Q
        L_A= csgraph.laplacian(A, normed=False)
        part1 = (Z + Y2/mu)
        part2 = np.linalg.inv(np.eye(X.shape[1]) + parameters['gamma']*(L_A+np.transpose(L_A)))
        Q = np.matmul(part1,part2)
        #update E
        #aux = np.matmul(W,X)-np.matmul(np.matmul(W,X),Z)-E+Y1/mu
        #E = soft_operator(aux,parameters['beta']/mu)
        aux = np.matmul(W,X)-np.matmul(np.matmul(W,X),Z)+Y1/mu
        E=(mu/(parameters['beta']+mu))*aux
        #update A
        A = update_A(A,Q,parameters)
        #update W
        W=gradient_optim(Z,Q,E,W,X,A,mu,Y1,Y2,parameters['theta'])
        #update Y1 and Y2
        #Y1 = mu*aux
        Y1 = Y1-mu*(np.matmul(W,X)-np.matmul(np.matmul(W,X),Z)-E)
        Y2 = Y2-mu*(Z-Q)
        #update mu
        mu=min(mu_max,rho*mu)
        print("print k Z and Q",k,Z,Q)
    tensor_dict={'Z':Z,'Q':Q,'A':A,'W':W,'X':X,'E':E,'Y1':Y1,'Y2':Y2}
    return tensor_list
