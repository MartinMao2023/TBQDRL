import numpy as np

class IntegrateMatern:
    def __init__(
        self,
        l: float, # lengthscale
        ):

        self.l = l
        def derivative_kernel(t1, t2):
            
            return np.exp(-np.abs(t1.reshape(-1, 1) - t2.reshape(1, -1))/l)

        self.derivative_kernel = derivative_kernel

        def kernel(t1, t2):
            t1 = t1.reshape(-1, 1)
            t2 = t2.reshape(1, -1)
            diff_t = np.abs(t1 - t2)
            min_t = np.where(t1 < t2, t1, t2)
            cov = l**2 * (
                np.exp(-t1/l) + np.exp(-t2/l) -np.exp(-diff_t/l) - 1
                ) + 2*l*min_t
            return cov
        
        self.kernel = kernel

        def cross_kernel(xt, vt):
            xt = xt.reshape(-1, 1)
            vt = vt.reshape(1, -1)
            cov = np.where(
                xt > vt,
                2*l - l * np.exp((vt - xt)/l),
                l * np.exp((xt - vt)/l),
                ) - l * np.exp(-vt/l)

            return cov
        
        self.xv_kernel = cross_kernel


    def covariance(
        self,
        t1: np.ndarray, # shape N1 x 1
        t2: np.ndarray, # shape N2 x 1
        ) -> np.ndarray: # shape N1 x N2

        cov = self.kernel(t1, t2)

        return cov
    

    def xv_cross_covariance(
        self, 
        xt: np.ndarray, 
        vt: np.ndarray,
        ) -> np.ndarray:

        cov = self.xv_kernel(xt, vt)
        return cov