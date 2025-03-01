import numpy as np
import astropy.io.fits as fits
import math, random
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

class MinimumEnergyDisambiguator:
    
    def __init__(self, Bx, By, Bz, az=None, dx=1.0, dy=1.0, lambda_factor=1.0, pad_width=3):
        self.Bx = np.array(Bx, copy=True)
        self.By = np.array(By, copy=True)
        self.Bz = np.array(Bz, copy=True)
        if az is not None:
            self.az = np.array(az, copy=True) 
        else:
            self.az = np.arctan2(self.By, self.Bx)
        self.dx = dx
        self.dy = dy
        self.lambda_factor = lambda_factor
        self.pad_width = pad_width
        self.nx, self.ny = self.Bx.shape
        self.DivB = np.zeros_like(self.Bx)
        self.Jz = np.zeros_like(self.Bx)
        self.compute_derivatives()

    def cos_taper(self, N, width):
        taper = np.ones(N)
        for i in range(width):
            taper[i] = 0.5 * (1 - math.cos(math.pi * (i + 1) / width))
            taper[-(i + 1)] = 0.5 * (1 - math.cos(math.pi * (i + 1) / width))
        return taper

    def apodize_array(self, arr):
        wx = self.cos_taper(self.nx, self.pad_width)
        wy = self.cos_taper(self.ny, self.pad_width)
        window = np.outer(wx, wy)
        return arr * window

    def compute_dBz_dz(self):
        pad = self.pad_width
        Bz_padded = np.pad(self.Bz, pad_width=pad, mode='constant', constant_values=0)
        N0, N1 = Bz_padded.shape
        taper0 = self.cos_taper(N0, pad)
        taper1 = self.cos_taper(N1, pad)
        window = np.outer(taper0, taper1)
        Bz_padded = Bz_padded * window
        Bz_fft = np.fft.fft2(Bz_padded)
        kx = np.fft.fftfreq(N0, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(N1, d=self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2 + KY**2
        K2[0, 0] = 1.0
        dBz_dz_fft = -np.sqrt(K2) * Bz_fft
        dBz_dz_padded = np.real(np.fft.ifft2(dBz_dz_fft))
        dBz_dz = dBz_dz_padded[pad:pad + self.nx, pad:pad + self.ny]
        return dBz_dz

    def compute_derivatives(self):
        Bx_ap = self.apodize_array(self.Bx)
        By_ap = self.apodize_array(self.By)
        dBx_dx = np.zeros_like(Bx_ap)
        dBx_dx[1:-1, :] = (Bx_ap[2:, :] - Bx_ap[:-2, :]) / (2.0 * self.dx)
        dBx_dx[0, :] = (Bx_ap[1, :] - Bx_ap[0, :]) / self.dx
        dBx_dx[-1, :] = (Bx_ap[-1, :] - Bx_ap[-2, :]) / self.dx
        dBy_dy = np.zeros_like(By_ap)
        dBy_dy[:, 1:-1] = (By_ap[:, 2:] - By_ap[:, :-2]) / (2.0 * self.dy)
        dBy_dy[:, 0] = (By_ap[:, 1] - By_ap[:, 0]) / self.dy
        dBy_dy[:, -1] = (By_ap[:, -1] - By_ap[:, -2]) / self.dy
        dBz_dz = self.compute_dBz_dz()
        self.DivB = dBx_dx + dBy_dy + dBz_dz
        dBy_dx = np.zeros_like(By_ap)
        dBy_dx[1:-1, :] = (By_ap[2:, :] - By_ap[:-2, :]) / (2.0 * self.dx)
        dBy_dx[0, :] = (By_ap[1, :] - By_ap[0, :]) / self.dx
        dBy_dx[-1, :] = (By_ap[-1, :] - By_ap[-2, :]) / self.dx
        dBx_dy = np.zeros_like(Bx_ap)
        dBx_dy[:, 1:-1] = (Bx_ap[:, 2:] - Bx_ap[:, :-2]) / (2.0 * self.dy)
        dBx_dy[:, 0] = (Bx_ap[:, 1] - Bx_ap[:, 0]) / self.dy
        dBx_dy[:, -1] = (Bx_ap[:, -1] - Bx_ap[:, -2]) / self.dy
        self.Jz = dBy_dx - dBx_dy

    def compute_energy(self):
        self.compute_derivatives()
        return np.sum(np.abs(self.DivB) + self.lambda_factor * np.abs(self.Jz))

    def calc_de_reconfig(self, i, j, window=3):
        orig_Bx = self.Bx[i, j]
        orig_By = self.By[i, j]
        half = window // 2
        i_min = max(i - half, 0)
        i_max = min(i + half + 1, self.nx)
        j_min = max(j - half, 0)
        j_max = min(j + half + 1, self.ny)
        self.compute_derivatives()
        E_before = np.sum(np.abs(self.DivB[i_min:i_max, j_min:j_max]) +
                          self.lambda_factor * np.abs(self.Jz[i_min:i_max, j_min:j_max]))
        self.Bx[i, j] = -self.Bx[i, j]
        self.By[i, j] = -self.By[i, j]
        self.compute_derivatives()
        E_after = np.sum(np.abs(self.DivB[i_min:i_max, j_min:j_max]) +
                         self.lambda_factor * np.abs(self.Jz[i_min:i_max, j_min:j_max]))
        de = E_after - E_before
        self.Bx[i, j] = orig_Bx
        self.By[i, j] = orig_By
        self.compute_derivatives()
        return de

    def reconfig(self, i, j):
        self.Bx[i, j] = -self.Bx[i, j]
        self.By[i, j] = -self.By[i, j]
        self.compute_derivatives()

    def simulated_annealing(self, neq=100, tfac0=0.1, tfactr=0.9,
                              tol_conv=1e-5, nconv_min=10, tstop_par=1e-3, max_iter=1000, Btr_threshold=100.0):
        E = self.compute_energy()
        max_de = 0.0
        for i in range(self.nx):
            for j in range(self.ny):
                Bt = math.sqrt(self.Bx[i, j]**2 + self.By[i, j]**2)
                if Bt < Btr_threshold:
                    continue
                de = abs(self.calc_de_reconfig(i, j))
                if de > max_de:
                    max_de = de
        t = tfac0 * max_de
        t_stop = tstop_par * t
        E_prev = E
        nconv = 0
        for _ in tqdm(range(max_iter), desc='Disambiguation [strong-masked fields] - simulated_annealing procedure'):
            nsucc = 0
            for _ in range(neq):
                i = random.randint(0, self.nx - 1)
                j = random.randint(0, self.ny - 1)
                Bt = math.sqrt(self.Bx[i, j]**2 + self.By[i, j]**2)
                if Bt < Btr_threshold:
                    continue
                de = self.calc_de_reconfig(i, j)
                if de < 0 or random.random() < math.exp(-de / t):
                    self.reconfig(i, j)
                    E += de
                    nsucc += 1
            t *= tfactr
            E_new = self.compute_energy()
            rel_change = abs(E_new - E_prev) / (abs(E_new) + abs(E_prev) + 1e-10)
            if rel_change < tol_conv:
                nconv += 1
            else:
                nconv = 0
            E_prev = E_new
            if nsucc == 0 or t < t_stop or nconv >= nconv_min:
                break

    def acute_angle_weak(self, Btr_threshold=100.0, max_iterations=10, conv_threshold=1e-2):

        updated = np.degrees(self.az.copy()) 
        def acute_diff(a, b):
            return abs(a - b)
        shape = updated.shape
        for it in range(max_iterations):
            prev = updated.copy()
            for i in range(1, shape[0]-1):
                for j in range(1, shape[1]-1):
                    Bt = math.sqrt(self.Bx[i, j]**2 + self.By[i, j]**2)
                    if Bt >= Btr_threshold:
                        continue
                    neighbors = [
                        updated[i-1, j], updated[i+1, j],
                        updated[i, j-1], updated[i, j+1],
                        updated[i-1, j-1], updated[i-1, j+1],
                        updated[i+1, j-1], updated[i+1, j+1]
                    ]
                    current_angle = updated[i, j]
                    flipped_angle = -current_angle  
                    cost_current = sum((acute_diff(current_angle, n))**2 for n in neighbors)
                    cost_flipped = sum((acute_diff(flipped_angle, n))**2 for n in neighbors)
                    if cost_flipped < cost_current:
                        updated[i, j] = flipped_angle
            change = np.max(np.abs(updated - prev))
            if change < conv_threshold:
                print(f'Acute-angle method [on weak-masked fields] converged after {it + 1} iterations')
                break
        self.az = np.radians(updated)
        for i in range(shape[0]):
            for j in range(shape[1]):
                Bt = math.sqrt(self.Bx[i, j]**2 + self.By[i, j]**2)
                if Bt < Btr_threshold:
                    self.Bx[i, j] = Bt * math.cos(self.az[i, j])
                    self.By[i, j] = Bt * math.sin(self.az[i, j])
        self.compute_derivatives()
    
    def smooth_fields(self, sigma=1.0):
        self.Bx = gaussian_filter(self.Bx, sigma)
        self.By = gaussian_filter(self.By, sigma)
        self.Bz = gaussian_filter(self.Bz, sigma)
        self.compute_derivatives()


    def save_disambiguation_results(self):
        hdu1 = fits.PrimaryHDU(self.Bx)
        hdu2 = fits.ImageHDU(self.By)
        hdu3 = fits.ImageHDU(np.degrees(self.az)) 
        hdu4 = fits.ImageHDU(self.Jz)
    
        hdu1.header['COMMENT'] = "Bx"
        hdu2.header['COMMENT'] = "By"
        hdu3.header['COMMENT'] = "Azimuth (deg)"
        hdu4.header['COMMENT'] = "Vertical current density, Jz"
    
        hdul = fits.HDUList([hdu1, hdu2, hdu3, hdu4])
        hdul.writeto('MEM_disambiguation_results.fits', overwrite=True)
        print('Minimum energy disambiguation is done and results are saved!')