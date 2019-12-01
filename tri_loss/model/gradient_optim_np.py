import numpy as np
"""
    The following is the derivative for loss wrt W for its gradient descent 2\thetaW+Y1XT -Y1ZTXT +(mu)[WX-WX(Z+Z^{T})X^{T}-EX^{T}+WXZZ^{T}X^{T}+EZTXT]
"""
def grad_W(Z,Q,E,W,X,A,mu,Y1,Y2,theta):
    WX = np.matmul(W,X)
    ZT = np.transpose(Z)
    Z_plus_ZT = Z + ZT
    ZZT = np.matmul(Z,ZT)
    XT = np.transpose(X)
    EXT= np.matmul(E,XT)
    EZTXT = np.matmul(E,np.matmul(ZT,XT))
    Y1XT = np.matmul(Y1,XT)
    Y1ZTXT = np.matmul(Y1,np.matmul(ZT,XT))
    WXZZTXT = np.matmul(WX,np.matmul(ZZT,XT))
    WXZplusZTXT = np.matmul(np.matmul(WX,Z_plus_ZT),XT)
    return (2*theta)*W+mu*(np.matmul(WX,XT) - WXZplusZTXT - EXT+ WXZZTXT + EZTXT) +Y1XT -Y1ZTXT

def gradient_optim(Z,Q,E,W,X,A,mu,Y1,Y2,theta):
    cur_x = W # The algorithm starts at x=W
    gamma = 0.00001 # step size multiplier
    precision = 0.00001
    previous_step_size = 1 
    max_iters = 10000 # maximum number of iterations
    iters = 0 #iteration counter

    while previous_step_size > precision and iters < max_iters:
        prev_x = cur_x
        cur_x -= gamma * grad_W(Z,Q,E,prev_x,X,A,mu,Y1,Y2,theta)
        previous_step_size = np.sum(np.abs(cur_x - prev_x))
        iters+=1
    return cur_x
