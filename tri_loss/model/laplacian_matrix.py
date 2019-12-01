import numpy as np
import torch
"""
    Computes the laplacian matrix for a given graph L_A= D_A -A where D_A is the degree matrix of A
"""
def laplacian_matrix(A):
    D_A=torch.zeros(A.size()[0],A.size()[0])
    for i in range(A.size()[0]):
        D_A[i][i]=A[i][:].nonzero().size()[0] # finds the number of outgoing egdes from the node aka its degree.
    return D_A - A

