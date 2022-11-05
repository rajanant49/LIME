import numpy as np
from scipy.fft import *
from skimage import exposure
from scipy.ndimage.filters import convolve
import cv2
from scipy.spatial import distance
from tqdm import trange
import bm3d

def firstOrderDerivative(n, k=1):
    return -np.eye(n) + np.eye(n, k=k)

def toeplitizMatrix(row,col):
    dx = np.zeros((row, col))
    dy = np.zeros((row, col))
    dx[1, 0] = 1
    dx[1, 1] = -1
    dy[0, 1] = 1
    dy[1, 1] = -1

    return dx,dy

class LIME:
    # Initialization parameters
    def __init__(self, iter, alpha, rho, gamma, strategy, eps, sigma):
        self.iter = iter
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.strategy = strategy
        self.eps = eps
        self.sigma = sigma

    # Load the image and normalize it to 0~1
    # Initialize Dx, Dy, DD
    def loadimage(self,imgPath):

        img = cv2.imread(imgPath)

        # # resize image if very large
        # if img.shape[0]>1000 or img.shape[1]>1000:
        #     img = cv2.resize(img,(1000,1000), interpolation=cv2.INTER_AREA)

        self.L = img / 255.0
        self.row, self.col = self.L.shape[0], self.L.shape[1]

        # equation 2
        self.T_hat = np.max(self.L, axis=2)

        #  And D contains Dh and Dv , which are the Toeplitz matrices from the discrete gradient operators with forward difference.
        self.Dv = firstOrderDerivative(self.row,k=1)
        self.Dh = firstOrderDerivative(self.col,k=-1)
        self.W = self.Strategy()

    def gaussian_kernel(self, spatial_sigma, size = 5):

        kernel = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                kernel[i, j] = np.exp(-0.5 * (distance.euclidean((i, j), (size // 2, size // 2)) ** 2) / (spatial_sigma ** 2))

        # gaussian kernel
        return kernel

    def compute_weights(self, Lp, L, kernel, eps = 1e-3):

        T = convolve(np.ones_like(L)/(L.shape[0]*L.shape[1]), kernel, mode='constant')         
        # equation 22
        # print(L.shape,T.shape,Lp.shape,kernel.shape)
        try:
            T = T / (np.abs(convolve(Lp @ L, kernel, mode='constant')) + eps) 
        except:
            T = T / (np.abs(convolve(L @ Lp, kernel, mode='constant')) + eps)

        return T

    def Strategy(self):
        if self.strategy == 2:

            # strategy 2 : equation 21
            self.Wv = 1 / (np.abs(self.Dv @ self.T_hat) + 1)
            self.Wh = 1 / (np.abs(self.T_hat @ self.Dh) + 1)
            return np.vstack((self.Wv, self.Wh))
        elif self.strategy == 1:

            #strategy 1
            return np.ones((self.row * 2, self.col))
        else:

            #strategy 3
            self.kernel = self.gaussian_kernel(self.sigma)
            self.Wv = self.compute_weights(L = self.T_hat, Lp = self.Dv, kernel=self.kernel, eps=self.eps)
            self.Wh = self.compute_weights(L = self.T_hat, Lp = self.Dh, kernel=self.kernel, eps=self.eps)
            return np.vstack((self.Wv, self.Wh))

    # T subproblem
    def T_sub(self, G, Z, miu):
        X = G - (Z / miu)
        Xv = X[:self.row, :]
        Xh = X[self.row:, :]

        dx,dy = toeplitizMatrix(self.row,self.col)
        dxf,dyf = fft2(dx),fft2(dy)

        # calculating using FFT
        DD = np.conj(dxf) * dxf + np.conj(dyf) * dyf

        # equation 13
        T = np.real(ifft2(fft2(2 * self.T_hat + miu * (self.Dv @ Xv + Xh @ self.Dh))/ 
                        (DD * miu + 2)))

        return exposure.rescale_intensity(T, (0, 1), (0.001, 1))

    # shrinkage operator
    def shrink(self, X, e):
        return np.sign(X) * np.maximum(np.abs(X) - e, 0)
        
    # G subproblem
    def G_sub(self, T, Z, miu, W):

        # equation 15
        return self.shrink(np.vstack((self.Dv @ T, T @ self.Dh)) + Z / miu, self.alpha * W / miu)

    # Z subproblem
    def Z_sub(self, T, G, Z, miu):
        return Z + miu * (np.vstack((self.Dv @ T, T @ self.Dh)) - G)

    # miu sub problem
    def miu_sub(self, miu):
        return miu * self.rho

    def run(self):
        # exact algorithm

        # initialize T, G, Z, Miu
        T = np.zeros((self.row, self.col))
        G = np.zeros((self.row * 2, self.col))
        Z = np.zeros((self.row * 2, self.col))
        miu = 1

        for i in trange(self.iter):
            T = self.T_sub(G, Z, miu)
            G = self.G_sub(T, Z, miu, self.W)
            Z = self.Z_sub(T, G, Z, miu)
            miu = self.miu_sub(miu)

        self.T = T ** self.gamma

        # equation 3
        self.T = np.repeat(self.T[..., None], 3, axis = -1)
        self.R = self.L / self.T

        # image denoising
        self.Rd = bm3d.bm3d(self.R, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        self.Rf = self.R*self.T+self.Rd*(1-self.T)

        return exposure.rescale_intensity(self.Rf,(0,1)) * 255
