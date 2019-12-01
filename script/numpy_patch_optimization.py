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
def soft_operator(y,t):
    print("t:",t)
    return np.sign(y)*np.maximum(np.abs(y)-t,np.zeros((y.shape[0],y.shape[1])))

"""
    L21 minimization for the proximal operator  for the update rule of error matrix E
"""
def L21_operator(y,E,t):
    row_norm=np.sum(np.abs(E)**2,axis=-1)**(1./2)
    return t*y/(row_norm[:,np.newaxis]+t)
"""
    Functions u_vector and update_A are for the update rule of A which is row wise updated.
"""
def u_vector(Q,parameters):
    u=np.zeros_like(Q)
    for i in range(Q.shape[0]):
        for j in range(i):
            u[i][j]=u[j][i]=(parameters['gamma']/(2*parameters['lambda']))*(np.sum((Q[:][i]-Q[:][j])**2))
    print("u",u)
    return u
def update_A(A,Q,parameters):
    u = u_vector(Q,parameters)
    for i in range(A.shape[0]):
        res= sum(sorted(u[i][:parameters['epsilon']]))
        A[i]=np.maximum(((1+res)/parameters['epsilon'])*np.ones_like(u[i])-u[i],np.zeros_like(u[i]))
        A[i]=A[i]/np.sum(A[i])
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
def convex_update(X,parameters):
    Z = Q = A = Y2 = np.zeros((X.shape[1], X.shape[1]))
    W = np.eye(X.shape[0])
    E = Y1 = np.zeros((parameters['dimension of W'],X.shape[1]))
    mu, mu_max, rho = 0.0001, 10**6, 1.1
    for k in range(50):
        #update W
        W=gradient_optim(Z,Q,E,W,X,A,mu,Y1,Y2,parameters['theta'])
        #update Z
        constraint = grad_f(Z,Q,E,W,X,A,mu,Y1,Y2)
        P = Z - parameters['tau']*constraint
        Z = soft_operator(P,parameters['alpha']/(parameters['eta']*mu))
        Z = np.maximum(Z,np.zeros_like(Z))
        #update Q
        L_A= csgraph.laplacian(0.5*(A+np.transpose(A)), normed=False)
        part1 = (Z + Y2/mu)
        part2 = np.linalg.inv(np.eye(X.shape[1]) + (parameters['gamma'])*(L_A+np.transpose(L_A)))
        Q = np.matmul(part1,part2)
        #update E
        aux = np.matmul(np.matmul(W,X),Z)-np.matmul(W,X)-Y1/mu
        E=L21_operator(aux,E,mu/parameters['beta'])
        #update A
        A = update_A(A,Q,parameters)
        #update Y1 and Y2
        #Y1 = (mu**2/(2-mu))*(np.matmul(W,X)-np.matmul(np.matmul(W,X),Z)-E)
        #Y2 = (mu**2/(2-mu))*(Z-Q)
        Y1 = Y1 + mu*(np.matmul(W,X)-np.matmul(np.matmul(W,X),Z)-E)
        Y2 = Y2 + mu*(Z-Q)
        #update mu
        mu=min(mu_max,rho*mu)
        print("print k",k)
        print("Z",Z)
        print("Q",Q)
        print("A",A)
        print("E",E)
        print("Y1",Y1)
        if(k%20==0):
            parameters['tau']=parameters['tau']/2
        if(k>50):
            if(np.linalg.norm(Z-Q)<0.1 and np.linalg.norm(aux-Y1/mu-E)<1):
                print("Success")
                break
    tensor_dict={'Z':Z,'Q':Q,'A':A,'W':W,'X':X,'E':E,'Y1':Y1,'Y2':Y2}
    return tensor_dict
#All the functions end here calling begins
features = sio.loadmat('../features_pca.mat')
X = features['X']
X = np.transpose(X)
row_sums = X.sum(axis=0)
X = X/row_sums[np.newaxis,:]
parameters={'dimension of W': X.shape[0],'alpha': 0.0001,'beta':0.1,'gamma':1000,'theta':0.001,'lambda':1,'tau':0.01,'epsilon':6,'eta':0}
parameters['eta']=np.sum(X**2)
tensor_dict = convex_update(X,parameters)
a=[]
for key in tensor_dict.keys():
    a.append(tensor_dict[key])
np.save('../../meta/data/pca/Z_Q_A_W_Xhat_E_Y1_Y2_100.npy',np.array(a))
