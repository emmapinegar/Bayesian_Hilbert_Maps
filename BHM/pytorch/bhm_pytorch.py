"""
# Bayesian Hilbert Maps with pytorch
# Ransalu Senanayake
# edited by Emma Pinegar
"""
import torch as pt
import numpy as np


dtype = pt.float32
device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu") 
VERBOSE = False
EXPANSION_COEF = 0.1
#TODO: merege 2D and 3D classes into a single class
#TODO: get rid of all numpy operations and test on a GPU
#TODO: parallelizing the segmentations
#TODO: efficient querying
#TODO: batch training
#TODO: re-using parameters for moving vehicles
   
class BHM_PYTORCH():
    def __init__(self, gamma=0.05, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None, nIter=0, verbose=False, mu_sig=None):
        """
        :param gamma: RBF bandwidth
        :param grid: if there are prespecified locations to hinge the RBF
        :param cell_resolution: if 'grid' is 'None', resolution to hinge RBFs
        :param cell_max_min: if 'grid' is 'None', realm of the RBF field
        :param X: a sample of lidar locations to use when both 'grid' and 'cell_max_min' are 'None'
        """
        self.gamma = gamma
        if grid is not None:
            self.grid = grid if pt.is_tensor(grid) else pt.tensor(grid, dtype=dtype, device=device)
        else:
            if (cell_max_min is not None and len(cell_max_min) > 4) or (X is not None and X.shape[1] > 2):
                self.grid = self.__calc_3d_grid_auto(cell_resolution, cell_max_min, X)
            else:
                self.grid = self.__calc_2d_grid_auto(cell_resolution, cell_max_min, X)
        self.nIter = nIter
        if mu_sig is not None:
            if pt.is_tensor(mu_sig):
                self.mu = mu_sig[:,0]
                self.sig = mu_sig[:,1]
            else:
                self.mu = pt.tensor(mu_sig[:,0], dtype=dtype, device=device)
                self.sig = pt.tensor(mu_sig[:,1], dtype=dtype, device=device)

        VERBOSE = verbose
        if VERBOSE:
            print(' Number of hinge points={}'.format(self.grid.shape[0]))

    def updateGrid(self, grid):
        if pt.is_tensor(grid):
            self.grid = grid
        else:
            self.grid = pt.tensor(grid, dtype=dtype, device=device)

    def updateMuSig(self, mu_sig):
        if pt.is_tensor(mu_sig):
            self.mu = mu_sig[:,0]
            self.sig = mu_sig[:,1]
        else:
            self.mu = pt.tensor(mu_sig[:,0], dtype=dtype, device=device)
            self.sig = pt.tensor(mu_sig[:,1], dtype=dtype, device=device)

    def merge_weights():
        return None

    def __calc_3d_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """
        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            x_min, x_max, y_min, y_max, z_min, z_max = X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max(), X[:, 2].min(), X[:, 2].max()
            x_expansion, y_expansion, z_expansion = (x_max - x_min) * EXPANSION_COEF, (y_max - y_min) * EXPANSION_COEF, (z_max - z_min) * EXPANSION_COEF
            x_min, x_max = x_min - x_expansion, x_max + x_expansion
            y_min, y_max = y_min - y_expansion, y_max + y_expansion
            z_min, z_max = z_min - z_expansion, z_max + z_expansion
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]
            z_min, z_max = max_min[4], max_min[5]

        xx, yy, zz = pt.meshgrid([pt.arange(x_min, x_max, cell_resolution[0], device=device),
                              pt.arange(y_min, y_max, cell_resolution[1], device=device),
                              pt.arange(z_min, z_max, cell_resolution[2], device=device)])
        grid = pt.stack((xx.reshape(-1,1), yy.reshape(-1,1), zz.reshape(-1,1)), dim=1).squeeze()
        return grid

    def __calc_2d_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """
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

        xx, yy = pt.meshgrid([pt.arange(x_min, x_max, cell_resolution[0], device=device),
                              pt.arange(y_min, y_max, cell_resolution[1], device=device)])
        grid = pt.stack((xx.reshape(-1,1), yy.reshape(-1,1)), dim=1).squeeze()
        return grid

    def __rbf_kernel(self, X1, X2, gamma):
        K = pt.norm(X1[:, None] - X2, dim=-1, p=2).pow(2)
        K = pt.exp(-gamma*K)
        # K[pt.where(K < 1e-2)] = 0
        zero = pt.zeros((1,1), dtype=dtype, device=device)
        return pt.where(K < 1e-2, zero[0], K)

    def __sparse_features(self, X):
        """
        :param X: inputs of size (N,2)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        rbf_features = self.__rbf_kernel(X, self.grid, gamma=self.gamma)
        # rbf_features = pt.cat((pt.ones(X.shape[0],1, device=device), rbf_features), dim=1)
        # rbf_features = rbf_kernel(X, self.grid, gamma=self.gamma)
        # rbf_features = pt.tensor(rbf_features, dtype=dtype, device=device)
        return rbf_features

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
        sig_add = pt.nan_to_num(pt.sum( (X.t()**2)*lam, dim=1))
        sig = 1/(1/sig0 + 2*sig_add)       
        mu = sig*(mu0/sig0 + pt.mm(X.t(), y - 0.5).squeeze())
        return mu, sig

    def fit(self, X, y):
        """
        :param X: raw data
        :param y: labels
        """
        X = self.__sparse_features(X)
        N, D = X.shape[0], X.shape[1]
        self.epsilon = pt.ones(N, dtype=dtype, device=device)
        if not hasattr(self, 'mu'):
            self.mu = pt.zeros(D, dtype=dtype, device=device)
            self.sig = 10000 * pt.ones(D, dtype=dtype, device=device)

        for i in range(self.nIter):
            if VERBOSE:
                print("\n  Parameter estimation: iter={}\n".format(i))
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
        param Xq: raw inquery points
        :param nSamples: number of samples to take the average over
        :return: sample mean and standard deviation of occupancy
        """
        Xq = self.__sparse_features(Xq)

        qw = pt.distributions.MultivariateNormal(self.mu, pt.diag(self.sig))
        w = qw.sample((nSamples,)).t()

        mu_a = Xq.mm(w).squeeze()
        probs = pt.sigmoid(mu_a)

        mean = pt.std(probs, dim=1).squeeze()
        std = pt.std(probs, dim=1).squeeze()

        return mean, std

      
      
def _save_new_weights(weights, opt = 'pick_highest_confidence'):
    """
    Determines which mu and sigma should be kept if a hinge point has more than one.
    Parameters:
    weights (n,5): x, y, z, mu, sigma for each hinge point evaluation
    opt (string): 'weight_equally', 'remove_low_confidence', 'pick_highest_confidence', 'pick_first_weight', 'conflate'
    Returns:
    kernel_weight (m,5): x, y, z, mu, sigma with each hinge point being unique
    """
    weights_dict = {}

    for weight in weights:
        x1, x2, x3, mu, sig = weight
        kernel = (x1, x2, x3)

        if kernel in weights_dict:
            # replace
            if np.size(weights_dict[kernel]) == 2 and _initial_value(weights_dict[kernel]):
                weights_dict[kernel] = [mu, sig]
            elif not _initial_value([mu, sig]):
                weights_dict[kernel] = np.vstack((weights_dict[kernel], [mu, sig]))
                # take only last two, by now guaranteed at least two elements
                
                # weights_dict[kernel] = weights_dict[kernel][-2:]
        else:
            weights_dict[kernel] = [mu, sig]

    kernel_weight = np.empty([0,5], dtype=np.float)
    for kernel in weights_dict:
        x1, x2, x3 = kernel
        params = weights_dict[kernel]
        
        # params should not be empty
        if np.size(params) > 2:
            num_param = (np.size(params))/2

            if opt == 'weight_equally':
                mu = sum(params[:,0]) / (num_param)
                sig = math.sqrt(sum([s**2 for s in params[:,1]]) / (num_param**2))
            elif opt == 'remove_low_confidence':
                confident_paras_indx = np.log(params[:,1]) <= 3 #3-4 in log-scale seems to be a good value. visualize sig and see.
                mu = sum(params[confident_paras_indx,0]) / (num_param)
                sig = math.sqrt(sum([s**2 for s in params[confident_paras_indx,1]]) / (num_param**2))
            elif opt == 'pick_highest_confidence':
                hightest_conf_indx = np.argmin(params[:,1])
                mu = params[hightest_conf_indx, 0]
                sig = params[hightest_conf_indx, 1]
            elif opt == 'pick_first_weight':
                mu = params[0, 0]
                sig = params[0, 1]
            elif opt == 'conflate':
                max_mu, min_sigma = params[0, 0], params[0,1]
                valid_mu, valid_sigma = [], []
                for i in range(0, params.shape[0]):
                    if params[i, 1] < 11000:
                        valid_mu.append(params[i,0])
                        valid_sigma.append(params[i,1])

                    if (np.abs(max_mu) < np.abs(params[i,0])):
                        max_mu = params[i,0]
                    min_sigma = min(params[i,1], min_sigma)
                if (len(valid_mu) > 1):
                    resultant_mu_nom, resultant_mu_denom, resultant_sigma_denom = 0, 0, 0
                    for k in range(0, len(valid_mu)):
                        sigma_squared = valid_sigma[k]**2
                        resultant_mu_nom += (valid_mu[k])/sigma_squared
                        resultant_mu_denom += 1/sigma_squared
                        resultant_sigma_denom += 1/sigma_squared
                    mu = resultant_mu_nom/resultant_mu_denom
                    sig = (1/resultant_sigma_denom)
                else:
                    mu, sig = max_mu, min_sigma
            kernel_weight = np.vstack((kernel_weight, [x1, x2, x3, mu, sig]))
        elif np.size(params) > 0:
            mu, sig = params
            kernel_weight = np.vstack((kernel_weight, [x1, x2, x3, mu, sig]))

    return kernel_weight      
