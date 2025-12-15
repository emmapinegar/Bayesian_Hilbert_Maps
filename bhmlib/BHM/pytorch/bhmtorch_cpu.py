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
from bhmlib.BHM.pytorch.kernels import RBF

# dtype = pt.float32
# device = pt.device("cpu")
#device = pt.device("cuda:0") # Uncomment this to run on GPU

#TODO: merge 2D and 3D classes into a single class
#TODO: get rid of all numpy operations and test on a GPU
#TODO: parallelizing the segmentations
#TODO: efficient querying
#TODO: batch training
#TODO: re-using parameters for moving vehicles


class BHM2D_PYTORCH():
    def __init__(
            self,
            gamma=0.05,
            grid=None,
            cell_resolution=(5, 5),
            cell_max_min=None,
            X=None,
            nIter=0,
            mu_sig=None,
            mu=None,
            sig=None,
            epsilon=None,
            torch_kernel_func=False,
    ):
        """
        :param gamma: RBF bandwidth
        :param grid: if there are prespecified locations to hinge the RBF
        :param cell_resolution: if 'grid' is 'None', resolution to hinge RBFs
        :param cell_max_min: if 'grid' is 'None', realm of the RBF field
        :param X: a sample of lidar locations to use when both 'grid' and 'cell_max_min' are 'None'
        """
        self.gamma = gamma
        if grid is not None:
            self.grid = grid
        else:
            self.grid = self.__calc_grid_auto(cell_resolution, cell_max_min, X)
        self.nIter = nIter
        print(' Number of hinge points={}'.format(self.grid.shape[0]))

        #ADDED
        if mu_sig is not None:
            self.mu = pt.tensor(mu_sig[:,0])
            self.sig = pt.tensor(mu_sig[:,1])
        else:
            if mu is not None:
                self.mu = mu
            if sig is not None:
                self.sig = sig
        self.torch_kernel_func = torch_kernel_func
        if torch_kernel_func:
            self.rbf_torch = RBF()
    def updateGrid(self, grid):
        self.grid = grid

    def updateMuSig(self, mu_sig):
        self.mu = pt.tensor(mu_sig[:,0])
        self.sig = pt.tensor(mu_sig[:,1])

    def __calc_grid_auto(self, cell_resolution, max_min, X):
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
            expansion_coef = 1.2
            x_min, x_max = expansion_coef*X[:, 0].min(), expansion_coef*X[:, 0].max()
            y_min, y_max = expansion_coef*X[:, 1].min(), expansion_coef*X[:, 1].max()
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]

        xx, yy = np.meshgrid(np.arange(x_min, x_max, cell_resolution[0]), \
                             np.arange(y_min, y_max, cell_resolution[1]))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))

        return pt.tensor(grid)

    def __sparse_features(self, X):
        """
        :param X: inputs of size (N,2)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        if self.torch_kernel_func:
            rbf_features, _, _ = self.rbf_torch.eval(X, self.grid, gamma=self.gamma)
            return rbf_features
        else:
            rbf_features = rbf_kernel(X, self.grid, gamma=self.gamma)
            # COMMENTED OUT BIAS TERM
            # rbf_features = np.hstack((np.ones(X.shape[0])[:, np.newaxis], rbf_features))
            return pt.tensor(rbf_features)

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
        sig = 1/(1/sig0 + 2*pt.sum( (X.t()**2)*lam, dim=1))
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
        self.gamma = params_dict['gamma']
        self.grid = params_dict['grid']
        self.mu = params_dict['mu']
        self.sig = params_dict['sig']
        self.nIter = params_dict['nIter']

    def fit(self, X, y):
        """
        :param X: raw data
        :param y: labels
        """
        X = self.__sparse_features(X)
        N, D = X.shape[0], X.shape[1]

        self.epsilon = pt.ones(N)
        if not hasattr(self, 'mu'):
            self.mu = pt.zeros(D)
            self.sig = 10000 * pt.ones(D)

        for i in range(self.nIter):
            print("  Parameter estimation: iter={}".format(i))

            # E-step
            self.mu, self.sig = self.__calc_posterior(X, y, self.epsilon, self.mu, self.sig)

            # M-step
            self.epsilon = pt.sqrt(pt.sum((X**2)*self.sig, dim=1) + (X.mm(self.mu.reshape(-1, 1))**2).squeeze())

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

        # Autodiff second term.
        dlog_p_dK = pt.autograd.grad(  # batch x rbf_features
            log_p.sum(),
            K,
        )[0]

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

class BHM3D_PYTORCH():
    def __init__(self, gamma=0.05, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None, nIter=2, mu_sig=None):
        """
        :param gamma: RBF bandwidth
        :param grid: if there are prespecified locations to hinge the RBF
        :param cell_resolution: if 'grid' is 'None', resolution to hinge RBFs
        :param cell_max_min: if 'grid' is 'None', realm of the RBF field
        :param X: a sample of lidar locations to use when both 'grid' and 'cell_max_min' are 'None'
        """
        self.gamma = gamma
        if grid is not None:
            self.grid = grid
        else:
            self.grid = self.__calc_grid_auto(cell_resolution, cell_max_min, X)
        self.nIter = nIter
        print(' Number of hinge points={}'.format(self.grid.shape[0]))

    def updateGrid(self, grid):
        self.grid = grid

    def updateMuSig(self, mu_sig):
        self.mu = pt.tensor(mu_sig[:,0])
        self.sig = pt.tensor(mu_sig[:,1])

    def __calc_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max, z_min, z_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """
        X = X.numpy()

        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            expansion_coef = 1.2
            x_min, x_max = expansion_coef*X[:, 0].min(), expansion_coef*X[:, 0].max()
            y_min, y_max = expansion_coef*X[:, 1].min(), expansion_coef*X[:, 1].max()
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]
            z_min, z_max = max_min[4], max_min[5]

        xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, cell_resolution[0]), \
                             np.arange(y_min, y_max, cell_resolution[1]), \
                             np.arange(z_min, z_max, cell_resolution[2]))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis], zz.ravel()[:, np.newaxis]))

        return pt.tensor(grid)

    def __sparse_features(self, X):
        """
        :param X: inputs of size (N,3)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        rbf_features = rbf_kernel(X, self.grid, gamma=self.gamma)

        # rbf_features = np.hstack((np.ones(X.shape[0])[:, np.newaxis], rbf_features))
        return pt.tensor(rbf_features)

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

        sig = 1/(1/sig0 + 2*pt.sum( (X.t()**2)*lam, dim=1))

        mu = sig*(mu0/sig0 + pt.mm(X.t(), y - 0.5).squeeze())

        return mu, sig

    def fit(self, X, y):
        """
        :param X: raw data
        :param y: labels
        """
        X = self.__sparse_features(X)
        N, D = X.shape[0], X.shape[1]

        self.epsilon = pt.ones(N)
        if not hasattr(self, 'mu'):
            self.mu = pt.zeros(D)
            self.sig = 10000 * pt.ones(D)

        for i in range(self.nIter):
            print("  Parameter estimation: iter={}".format(i))

            # E-step
            self.mu, self.sig = self.__calc_posterior(X, y, self.epsilon, self.mu, self.sig)

            # M-step
            self.epsilon = pt.sqrt(pt.sum((X**2)*self.sig, dim=1) + (X.mm(self.mu.reshape(-1, 1))**2).squeeze())
        return self.mu, self.sig

    def predict(self, Xq):
        """
        :param Xq: raw inquery points
        :return: mean occupancy (Lapalce approximation)
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


class BHM_FULL_PYTORCH():
    #TODO: double check evrything
    def __init__(self, gamma=0.075*0.814, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None):
        """
        :param gamma: RBF bandwidth
        :param grid: if there are prespecified locations to hinge the RBF
        :param cell_resolution: if 'grid' is 'None', resolution to hinge RBFs
        :param cell_max_min: if 'grid' is 'None', realm of the RBF field
        :param X: a sample of lidar locations to use when both 'grid' and 'cell_max_min' are 'None'
        """
        self.gamma = gamma
        if grid is not None:
            self.grid = grid
        else:
            self.grid = self.__calc_grid_auto(cell_resolution, cell_max_min, X)
        print('D=', self.grid.shape[0])

    def __calc_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """
        X = X.numpy()

        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            expansion_coef = 1.2
            x_min, x_max = expansion_coef*X[:, 0].min(), expansion_coef*X[:, 0].max()
            y_min, y_max = expansion_coef*X[:, 1].min(), expansion_coef*X[:, 1].max()
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]

        xx, yy = np.meshgrid(np.arange(x_min, x_max, cell_resolution[0]), \
                             np.arange(y_min, y_max, cell_resolution[1]))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))

        return pt.tensor(grid)

    def __sparse_features(self, X):
        """
        :param X: inputs of size (N,2)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        rbf_features = rbf_kernel(X, self.grid, gamma=self.gamma)
        rbf_features = np.hstack((np.ones(X.shape[0])[:, np.newaxis], rbf_features))
        return pt.tensor(rbf_features)

    def __lambda(self, epsilon):
        logit_inv = pt.sigmoid(epsilon)
        return 0.5 / epsilon * (logit_inv - 0.5)

    def __calc_posterior(self, X, y, epsilon, mu0, sig0_inv):
        lam = self.__lambda(epsilon)
        mu0 = mu0.reshape(-1,1)

        #sig = 1/(1/sig0 + 2*pt.sum( (X.t()**2)*lam, dim=1))
        sig_inv = sig0_inv + 2*pt.mm(X.t() * lam, X)
        sig = pt.inverse(sig_inv)
        #sig = pt.diag(1/pt.diag(sig_inv))

        print("sum=", pt.sum(sig_inv.mm(sig)))

        pl.subplot(121)
        pl.imshow(sig_inv[1:,1:], cmap='jet', interpolation=None); pl.colorbar()
        pl.subplot(122)
        pl.imshow(sig[1:,1:], cmap='jet', interpolation=None); pl.colorbar()
        pl.show()

        #mu = sig*(mu0/sig0 + pt.mm(X.t(), y - 0.5).squeeze())
        #sig_inv = sig0_inv + 2 * pt.diag(pt.mm(X.t() * lam, X)).squeeze()
        #mu = pt.mm((1 / sig_inv).reshape(1, -1), (sig_inv * mu0 + pt.mm(X.t(), y - 0.5))).squeeze()
        mu = sig.mm(sig0_inv.mm(mu0) + pt.mm(X.t(), (y - 0.5)))

        pl.close('all')
        pl.scatter(self.grid[:,0], self.grid[:,1], c=mu.squeeze()[1:], cmap='jet', s=5, edgecolor='')
        pl.colorbar()
        pl.show()

        return mu.squeeze(), sig_inv, sig

    def fit(self, X, y):
        X = self.__sparse_features(X)
        N, D = X.shape[0], X.shape[1]

        epsilon = pt.ones(N)
        mu = pt.zeros(D)
        sig_inv = 0.0001 * pt.eye(D)

        for i in range(1):
            print("i=", i)

            # E-step
            mu, sig_inv, sig = self.__calc_posterior(X, y, epsilon, mu, sig_inv)
            print('d', mu.shape)

            # M-step
            XMX = pt.mv(X, mu)**2
            XSX = pt.sum(pt.mm(X, pt.mm(sig, X.t())), dim=1)
            print(XMX.shape, XSX.shape)
            epsilon = pt.sqrt(XMX + XSX) #TODO - Bug

        self.mu, self.sig_inv = mu, sig_inv

    def predict(self, Xq):
        Xq = self.__sparse_features(Xq)

        sig_diag = 1/pt.diagonal(self.sig_inv)

        mu_a = Xq.mm(self.mu.reshape(-1, 1)).squeeze()
        sig2_inv_a = pt.sum((Xq ** 2) * sig_diag, dim=1)
        k = 1.0 / pt.sqrt(1 + np.pi * sig2_inv_a / 8)

        return pt.sigmoid(k*mu_a)
