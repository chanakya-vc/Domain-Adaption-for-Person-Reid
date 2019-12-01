import numpy as np
import torch
"""
    The following is the derivative for loss wrt W for its gradient descent 2\thetaW+Y1XT -Y1ZTXT +(mu)[WX-WX(Z+Z^{T})X^{T}-EX^{T}+WXZZ^{T}X^{T}+EZTXT]
"""
def grad_W(Z,Q,E,W,X,A,mu,Y1,Y2,theta):
    WX = torch.mm(W,X)
    ZT = torch.transpose(Z,0,1)
    Z_plus_ZT = Z + ZT
    ZZT = torch.mm(Z,ZT)
    XT = torch.transpose(X,0,1)
    EXT= torch.mm(E,XT)
    EZTXT = torch.mm(E,torch.mm(ZT,XT))
    Y1XT = torch.mm(Y1,XT)
    Y1ZTXT = torch.mm(Y1,torch.mm(ZT,XT))
    WXZZTXT = torch.mm(WX,torch.mm(ZZT,XT))
    WXZplusZTXT = torch.mm(torch.mm(WX,Z_plus_ZT),XT)
    return (2*theta)*W+mu*(torch.mm(WX,XT) - WXZplusZTXT - EXT+ WXZZTXT + EZTXT) +Y1XT -Y1ZTXT

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
        previous_step_size = torch.sum(torch.abs(cur_x - prev_x))
        iters+=1
    return cur_x
