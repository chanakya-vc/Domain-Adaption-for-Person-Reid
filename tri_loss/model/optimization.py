import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from numpy_patch_optimization import convex_update
import numpy as np
import scipy.io as sio
### optimization code for the variables
def optimization(feature_list):
    if feature_list==[]:
        ## call the pca features
        #features = sio.loadmat('../features_pca.mat')
        #X = features['X']
	X=[]
    else:
        X = np.array(feature_list)
    tensor_dict=convex_update(X)
    Z=torch.from_numpy(tensor_dict['Z'])
    np.save('~/A.npy',tensor_dict['A'])
    return Z
