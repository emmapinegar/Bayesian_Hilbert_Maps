"""
# 2D and 3D Bayesian Hilbert Maps with pytorch
# Ransalu Senanayake
"""
import torch as pt
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as pl
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
import pandas as pd
from .kernels import RBF

dtype_ = pt.float32
device_ = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
EXPANSION_COEF = 0.05
VERBOSE = True

#TODO: merge 2D and 3D classes into a single class
#TODO: get rid of all numpy operations and test on a GPU
#TODO: parallelizing the segmentations
#TODO: efficient querying
#TODO: batch training
#TODO: re-using parameters for moving vehicles


class BHM_PYTORCH():
    def __init__(self, gamma=0.05, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None, nIter=0, mu_sig=None, mu=None, sig=None, epsilon=None, torch_kernel_func=False, device=device_, file=None, limit_scale=1):
        """
        :param gamma: RBF bandwidth
        :param grid: if there are prespecified locations to hinge the RBF
        :param cell_resolution: if 'grid' is 'None', resolution to hinge RBFs
        :param cell_max_min: if 'grid' is 'None', realm of the RBF field
        :param X: a sample of lidar locations to use when both 'grid' and 'cell_max_min' are 'None'
        """
        self.device = device
        self.limits = cell_max_min
        self.limit_scale = limit_scale
        if file is not None:
            self.load(file)
        else:
            self.gamma = gamma
            if grid is not None:
                self.updateGrid(grid)
            else:
                if (X is not None and X.shape[1] > 2) or (cell_max_min is not None and len(cell_max_min) > 4):
                    self.grid = self.__calc_3d_grid_auto(cell_resolution, cell_max_min, X)
                else:
                    self.grid = self.__calc_2d_grid_auto(cell_resolution, cell_max_min, X)

            self.nIter = nIter
            if VERBOSE:
                print(f" Number of hinge points={self.grid.shape[0]}")

            #ADDED
            if mu_sig is not None:
                self.updateMuSig(mu_sig)
            else:
                if mu is not None:
                    self.mu = mu
                if sig is not None:
                    self.sig = sig
        self.torch_kernel_func = torch_kernel_func
        if torch_kernel_func:
            self.rbf_torch = RBF()

    def updateGrid(self, grid):
        if pt.is_tensor(grid):
            self.grid = grid
        else:
            self.grid = pt.tensor(grid, device=self.device)

    def updateMuSig(self, mu_sig):
        if pt.is_tensor(mu_sig):
            self.mu = mu_sig[:,0]
            self.sig = mu_sig[:,1]
        else:
            self.mu = pt.tensor(mu_sig[:,0], device=self.device)
            self.sig = pt.tensor(mu_sig[:,1], device=self.device)

    def __calc_2d_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """
        # X = X.numpy()
        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            x_min, x_max, y_min, y_max = X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()
            x_expansion, y_expansion = (x_max - x_min) * EXPANSION_COEF, (y_max - y_min) * EXPANSION_COEF
            x_min, x_max = x_min - x_expansion, x_max + x_expansion
            y_min, y_max = y_min - y_expansion, y_max + y_expansion
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]

        xx, yy, zz = pt.meshgrid([pt.arange(x_min, x_max + cell_resolution[0], cell_resolution[0], device=self.device),
                              pt.arange(y_min, y_max + cell_resolution[1], cell_resolution[1], device=self.device)], indexing="ij")
        grid = pt.stack((xx.reshape(-1,1), yy.reshape(-1,1)), dim=1).squeeze()
        return grid
    
    def __calc_3d_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max, z_min, z_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """
        # X = X.cpu.numpy()
        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            x_min, x_max, y_min, y_max, z_min, z_max = X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max(), X[:, 2].min(), X[:, 2].max()
            x_range, y_range, z_range = (x_max - x_min), (y_max - y_min), (z_max - z_min)

            # x_middle, y_middle, z_middle = x_range/2 + x_min, y_range/2 + y_min, z_range/2 + z_min
            # x_points, y_points, z_points = pt.round(x_range/cell_resolution[0] + 1.5)*cell_resolution[0]/2, pt.round(y_range/cell_resolution[1] + 1.5)*cell_resolution[1]/2, pt.round(z_range/cell_resolution[2] + 1.5)*cell_resolution[2]/2
            # x_min, x_max = x_middle - x_points, x_middle + x_points + cell_resolution[0]
            # y_min, y_max = y_middle - y_points, y_middle + y_points + cell_resolution[1]
            # z_min, z_max = z_middle - z_points, z_middle + z_points + cell_resolution[2]

            x_expansion, y_expansion, z_expansion = x_range * EXPANSION_COEF, y_range * EXPANSION_COEF, z_range * EXPANSION_COEF
            x_min, x_max = x_min - x_expansion, x_max + x_expansion
            y_min, y_max = y_min - y_expansion, y_max + y_expansion
            z_min, z_max = z_min - z_expansion, z_max + z_expansion
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]
            z_min, z_max = max_min[4], max_min[5]

        print(f"grid points x: {x_min} {x_max} y: {y_min} {y_max} z: {z_min} {z_max}")

        xx, yy, zz = pt.meshgrid([pt.arange(x_min, x_max + cell_resolution[0], cell_resolution[0], device=self.device),
                              pt.arange(y_min, y_max + cell_resolution[1], cell_resolution[1], device=self.device),
                              pt.arange(z_min, z_max + cell_resolution[2], cell_resolution[2], device=self.device)], indexing="ij")
        grid = pt.stack((xx.reshape(-1,1), yy.reshape(-1,1), zz.reshape(-1,1)), dim=1).squeeze()
        return grid

    def __sparse_features(self, X):
        """
        :param X: inputs of size (N,2)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        if self.torch_kernel_func:
            rbf_features, _, _ = self.rbf_torch.eval(X, self.grid, gamma=self.gamma)
            return rbf_features
        else:
            rbf_features = rbf_kernel(X.detach().cpu().numpy(), self.grid.detach().cpu().numpy(), gamma=self.gamma)
            # COMMENTED OUT BIAS TERM
            # rbf_features = np.hstack((np.ones(X.shape[0])[:, np.newaxis], rbf_features))
            return pt.tensor(rbf_features, device=self.device)

    def __calc_posterior(self, X, y, epsilon, mu0, sig0):
        """
        :param X: input features
        :param y: labels
        :param epsilon: per dimension local linear parameter
        :param mu0: mean
        :param sig0: variance
        :return: new_mean, new_varaiance
        """
        logit_inv = pt.sigmoid(epsilon)
        lam = 0.5 / epsilon * (logit_inv - 0.5)
        sig = 1/(1/sig0 + 2*pt.nan_to_num(pt.sum( (X.t()**2)*lam, dim=1)))
        mu = sig*(mu0/sig0 + pt.mm(X.t(), y - 0.5).squeeze())
        return mu, sig

    def save(self, save_path=None, filename='bhm.pt'):
        save_path.mkdir(parents=False, exist_ok=True)
        params = {
            'gamma': self.gamma,
            'grid': self.grid,
            'mu': self.mu,
            'sig': self.sig,
            'nIter': self.nIter,
        }
        pt.save(params, save_path / filename)

    def load(self, file_path=None):
        params_dict = pt.load(file_path)
        print(params_dict)
        self.gamma = params_dict['gamma']
        self.grid = params_dict['grid'].to(self.device)
        self.mu = params_dict['mu'].to(self.device)
        self.sig = params_dict['sig'].to(self.device)
        self.nIter = params_dict['nIter']

    def fit(self, X, y):
        """
        :param X: raw data
        :param y: labels
        """
        X = self.__sparse_features(X)
        N, D = X.shape[0], X.shape[1]

        self.epsilon = pt.ones(N, device=self.device)
        if not hasattr(self, 'mu'):
            self.mu = pt.zeros(D, device=self.device)
            self.sig = 10000 * pt.ones(D, device=self.device)

        for i in range(self.nIter):

            # E-step
            self.mu, self.sig = self.__calc_posterior(X, y, self.epsilon, self.mu, self.sig)

            # M-step
            self.epsilon = pt.sqrt(pt.sum((X**2)*self.sig, dim=1) + (X.mm(self.mu.reshape(-1, 1))**2).squeeze())
            if VERBOSE:
                print(f"  Parameter estimation: iter={i} \t mu: {self.mu.size()} sig: {self.sig.size()} x: {X.size()}")
        # print(self.mu)

        return self.mu, self.sig

    def log_prob_vacancy(self, Xq):
        """
        Log-probability of vacancy, where prob_vacant = 1 - prob_occupied
        :param Xq: raw in query points

        """
        prob_Xq = 1. - self.predict(Xq)
        return pt.log(prob_Xq)

    def grad_log_p_vacancy(self, Xq):
        """
        :param Xq: raw in query points
        """
        assert self.torch_kernel_func
        # Use torch kernels with analytic gradients
        with pt.no_grad():
            K, dK_dXq, _ = self.rbf_torch.eval(Xq, self.grid, gamma=self.gamma)

        # From predict function
        K.requires_grad = True
        mu_a = K.mm(self.mu.reshape(-1, 1)).squeeze()
        sig2_inv_a = pt.sum((K ** 2) * self.sig, dim=1)
        k = 1.0 / pt.sqrt(1 + np.pi * sig2_inv_a / 8)
        log_p = pt.log(1. - pt.sigmoid(k * mu_a))

        # TODO: add the limits alterations to log_p calculations?
        if self.limits is not None:
            # print(self.limits[0][0])
            # print(Xq[0,0])
            print_str = f"log_p: {log_p[0]:5.6f}"

            log_p_diff = pt.exp(-self.limit_scale*(Xq[:, 0] - self.limits[0][0]))
            log_p_diff += pt.exp( self.limit_scale*(Xq[:, 0] - self.limits[0][1]))
            log_p_diff += pt.exp(-self.limit_scale*(Xq[:, 1] - self.limits[1][0]))
            log_p_diff += pt.exp( self.limit_scale*(Xq[:, 1] - self.limits[1][1]))
            if len(self.limits) > 2:
                log_p_diff += pt.exp(-self.limit_scale*(Xq[:, 2] - self.limits[2][0]))
                log_p_diff += pt.exp( self.limit_scale*(Xq[:, 2] - self.limits[2][1]))            
            print(print_str + f" diff: {log_p_diff[0]:5.6f} new log_p: {log_p[0] - log_p_diff[0]:5.6f}  x: {Xq[0,0] - self.limits[0][0]:5.6f} {Xq[0,0] - self.limits[0][1]:5.6f} y: {Xq[0,1] - self.limits[1][0]:5.6f} {Xq[0,1] - self.limits[1][1]:5.6f} z: {Xq[0,2] - self.limits[2][0]:5.6f} {Xq[0,2] - self.limits[2][1]:5.6f}")
            log_p -= log_p_diff
        else:
            print(f"log_p: {log_p[0]:5.6f}")

        # Autodiff second term.
        dlog_p_dK = pt.autograd.grad(log_p.sum(), K,)[0] # batch x rbf_features

        # Chain rule gradients
        dlog_p_dXq = dlog_p_dK.unsqueeze(1) @ dK_dXq

        return dlog_p_dXq.squeeze(1)

    def predict(self, Xq):
        """
        :param Xq: raw in query points
        :return: mean occupancy (Laplace approximation)
        """
        Xq = self.__sparse_features(Xq)

        mu_a = Xq.mm(self.mu.reshape(-1, 1)).squeeze()
        sig2_inv_a = pt.sum((Xq ** 2) * self.sig, dim=1)
        k = 1.0 / pt.sqrt(1 + np.pi * sig2_inv_a / 8)

        return pt.sigmoid(k*mu_a)

    def predictSampling(self, Xq, nSamples=50):
        """
        :param Xq: raw inquery points
        :param nSamples: number of samples to take the average over
        :return: sample mean and standard deviation of occupancy
        """
        Xq = self.__sparse_features(Xq)

        qw = pt.distributions.MultivariateNormal(self.mu, pt.diag(self.sig))
        w = qw.sample((nSamples,)).t()

        mu_a = Xq.mm(w).squeeze()
        probs = pt.sigmoid(mu_a)

        mean = pt.mean(probs, dim=1).squeeze()
        std = pt.std(probs, dim=1).squeeze()

        return mean, std