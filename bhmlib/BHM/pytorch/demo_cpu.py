"""
# 2D kitti dataset test
# In this demo, 1) how to run kitti in cpu/gpu with pytorch, 2) how to partition the map and run parallelly
# TODO: setup for cuda
"""
import os
import time
import numpy as np
import pandas as pd
import torch as pt
import matplotlib.pyplot as pl
from pathlib import Path
from bhmtorch_cpu import BHM2D_PYTORCH

def getPartitions(cell_max_min, nPartx1, nPartx2):
    """
    :param cell_max_min: The size of the entire area
    :param nPartx1: How many partitions along the longitude
    :param nPartx2: How many partitions along the latitude
    :return: a list of all partitions
    """
    width = cell_max_min[1] - cell_max_min[0]
    height = cell_max_min[3] - cell_max_min[2]
    cell_max_min_segs = []
    for x in range(nPartx1):
        for y in range(nPartx2):
            seg_i = (cell_max_min[0] + width / nPartx1 * x, cell_max_min[0] + width / nPartx1 * (x + 1), \
                     cell_max_min[2] + height / nPartx2 * y, cell_max_min[2] + height / nPartx2 * (y + 1))
            cell_max_min_segs.append(seg_i)

    return cell_max_min_segs

def load_parameters(case):
    parameters = \
        {'kitti1': \
             ( os.path.abspath('../../Datasets/kitti/kitti2011_09_26_drive0001_frame'),
              (2, 2), #hinge point resolution
              (-80, 80, -80, 80), #area [min1, max1, min2, max2]
              None,
              None,
              0.5, #gamma
              ),

        'intel': \
             ('../../Datasets/intel.csv',
              (0.5, 0.5), #x1 and x2 resolutions for positioning hinge points
              (-20, 20, -25, 10), #area to be mapped [x1_min, x1_max, x2_min, x2_max]
              1, #N/A
              0.01, #threshold for filtering data
              6.71 #gamma: kernel parameter
            ),

         }

    return parameters[case]

base_path =  Path(__file__).resolve().parents[2]

# Settings
# dtype = pt.float32
device = pt.device("cpu")
# pt.set_default_tensor_type(pt.FloatTensor)
pt.set_default_tensor_type(pt.DoubleTensor)
# dataset =  'kitti1'
dataset = 'intel'
# save_path = base_path / 'Outputs' / 'saved_models'
save_path = None
save_iter = 10
plot_iter = 10
#device = pt.device("cuda:0") # Uncomment this to run on GPU

# Read the file
(fn_train,
 cell_resolution,
 cell_max_min,
 skip,
 thresh,
 gamma,) = load_parameters(dataset)

#read data
g = pd.read_csv(fn_train, delimiter=',').values
print('shapes:', g.shape)
g = pt.tensor(g)
X_train = g[:, 0:3]
Y_train = g[:, 3].reshape(-1, 1)

max_t = len(pt.unique(X_train[:, 0]))
print(max_t)

for ith_scan in range(0, max_t, skip):

    # extract data points of the ith scan
    ith_scan_indx = X_train[:, 0] == ith_scan
    print('{}th scan:\n  N={}'.format(ith_scan, pt.sum(ith_scan_indx)))
    X_new = X_train[ith_scan_indx, 1:]
    y_new = Y_train[ith_scan_indx]

    if ith_scan == 0:
        # get all data for the first scan and initialize the model
        X, y = X_new, y_new
        bhm_mdl = BHM2D_PYTORCH(
        	gamma=gamma,
        	torch_kernel_func=True,
        	grid=None,
        	cell_resolution=cell_resolution,
        	cell_max_min=cell_max_min,
        	X=X,
        	nIter=1,
        )
    else:
        # information filtering
        q_new = bhm_mdl.predict(X_new).reshape(-1, 1)
        info_val_indx = pt.absolute(q_new - y_new) > thresh
        info_val_indx = info_val_indx.flatten()
        X, y = X_new[info_val_indx, :], y_new[info_val_indx]
        print('  {:.2f}% points were used.'.format(X.shape[0]/X_new.shape[0]*100))

    # Fit the model
    t1 = time.time()
    bhm_mdl.fit(X, y)
    t2 = time.time()

    q_resolution = 0.25
    if ith_scan % plot_iter == 0:
        # query the model
        xx, yy= np.meshgrid(np.arange(cell_max_min[0], cell_max_min[1] - 1, q_resolution),
                             np.arange(cell_max_min[2], cell_max_min[3] - 1, q_resolution))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
        Xq = pt.tensor(grid)
        # Predict
        t3 = time.time()
        yq = bhm_mdl.predict(Xq)
        t4 = time.time()

        print('Fit time: {}'.format(t2 - t1))
        print('Pred time: {}'.format(t4 - t3))
        print('iter time: {}\n'.format(t4 - t1))

        Xq = Xq.cpu().numpy()
        yq = yq.cpu().numpy()

        print('PLotting...')
        pl.figure(figsize=(13,5))
        pl.subplot(121)
        ones_ = np.where(y==1)
        # pl.scatter(X[ones_, 0], X[ones_, 1], c='r', cmap='jet', s=5, edgecolors='')
        pl.scatter(X[ones_, 0], X[ones_, 1], c='r', cmap='jet', s=5)
        pl.title('Laser hit points at t={}'.format(ith_scan))
        pl.xlim([cell_max_min[0], cell_max_min[1]]); pl.ylim([cell_max_min[2], cell_max_min[3]])
        pl.subplot(122)
        pl.title('SBHM at t={}'.format(ith_scan))
        # pl.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=10, marker='8',edgecolors='')
        pl.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=10, marker='8',)
        # pl.scatter(Xq[:, 0], Xq[:, 1], cmap='jet', marker='8',edgecolors='')
        # pl.scatter(Xq[:, 0], Xq[:, 1])
        #pl.imshow(Y_query.reshape(xx.shape))
        pl.colorbar()
        pl.xlim([cell_max_min[0], cell_max_min[1]]); pl.ylim([cell_max_min[2], cell_max_min[3]])
        # pl.savefig('Output/step' + str(ith_scan) + '.png', bbox_inches='tight')
        pl.savefig(os.path.abspath('../../Outputs/intel_{:03d}.png'.format(ith_scan)), bbox_inches='tight')

    if save_path is not None and \
       ith_scan % save_iter == 0:
        print('Saving map...')
        filename = 'bhm_{}_res{}_iter{:03d}.pt'.format(dataset, q_resolution, ith_scan)
        bhm_mdl.save(save_path, filename)

# # Partition the environment into to 4 areas
# # TODO: We can parallelize this
# cell_max_min_segments = getPartitions(cell_max_min, 2, 2)
#
# # Read data
# for framei in range(108):
#     print('\nReading '+fn_train+'{}.csv...'.format(framei))
#     g = pd.read_csv(
#     		fn_train+'{}.csv'.format(framei),
#     		delimiter=',',
#     	).values[:, :]
#
#     # Filter data
#     layer = np.logical_and(g[:,2] >= 0.02, g[:,2] <= 0.125)
#     #layer = np.logical_and(g[:, 2] >= -0.6, g[:, 2] <= -0.5)
#     g = pt.tensor(g[layer, :], dtype=pt.float32)
#     X = g[:, :2]
#     y = g[:, 3].reshape(-1, 1)
#     # if pt.cuda.is_available():
#     #     X = X.cuda()
#     #     y = y.cuda()
#
#     toPlot = []
#     totalTime = 0
#     for segi in range(len(cell_max_min_segments)):
#         print(' Mapping segment {} of {}...'.format(segi+1,len(cell_max_min_segments)))
#         cell_max_min = cell_max_min_segments[segi]
#
#         bhm_mdl = BHM2D_PYTORCH(
#         	gamma=gamma,
#         	grid=None,
#         	cell_resolution=cell_resolution,
#         	cell_max_min=cell_max_min,
#         	X=X,
#         	nIter=1,
#         )
#
#         t1 = time.time()
#         bhm_mdl.fit(X, y)
#         t2 = time.time()
#         totalTime += (t2-t1)
#
#         # query the model
#         q_resolution = 0.5
#         xx, yy= np.meshgrid(np.arange(cell_max_min[0], cell_max_min[1] - 1, q_resolution),
#                              np.arange(cell_max_min[2], cell_max_min[3] - 1, q_resolution))
#         grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
#         Xq = pt.tensor(grid, dtype=pt.float32)
#         yq = bhm_mdl.predict(Xq)
#         toPlot.append((Xq,yq))
#     print(' Total training time={} s'.format(np.round(totalTime, 2)))
#
#     # Plot frame i
#     pl.close('all')
#     for segi in range(len(cell_max_min_segments)):
#         ploti = toPlot[segi]
#         Xq, yq = ploti[0], ploti[1]
#         pl.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=5, vmin=0, vmax=1, edgecolors='')
#     pl.colorbar()
#     pl.xlim([-80,80]); pl.ylim([-80,80])
#     # pl.title('kitti2011_09_26_drive0001_frame{}'.format(framei))
#     pl.title('{}_frame{}'.format(dataset,framei))
#     #pl.savefig(os.path.abspath('../../Outputs/kitti2011_09_26_drive0001_frame{}.png'.format(framei)))
#     pl.show()