import numpy as np
import astropy.io.fits as fits
import math
import random
from tqdm import tqdm
from numba import njit

class MinimumEnergyDisambiguator:
    def __init__(self, Bx, By, Bz, az=None, dx=1.0, dy=1.0, lambda_factor=1,
                 pad_width=50, tfactr=0.95, tfac0=0.05, neq=200, seed=123456, verbose=1, B_threshold=200):
        self.tfactr = tfactr
        self.tfac0 = tfac0
        self.neq = neq
        self.seed = seed
        np.random.seed(seed)
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
        self.current_energy = None
        self.verbose = verbose
        self.B_threshold = B_threshold
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
        Bz_padded = np.pad(self.Bz, pad_width=pad, mode='constant')
        N0, N1 = Bz_padded.shape
        taper0 = self.cos_taper(N0, pad)
        taper1 = self.cos_taper(N1, pad)
        window = np.outer(taper0, taper1)
        Bz_padded *= window
        Bz_fft = np.fft.fft2(Bz_padded)
        kx = np.fft.fftfreq(N0, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(N1, d=self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2 + KY**2
        K2[0, 0] = 1.0
        dBz_dz_fft = -np.sqrt(K2) * Bz_fft
        dBz_dz_padded = np.real(np.fft.ifft2(dBz_dz_fft))
        return dBz_dz_padded[pad:pad + self.nx, pad:pad + self.ny]

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
        self.current_energy = np.sum(np.abs(self.DivB) + self.lambda_factor * np.abs(self.Jz))
        return self.current_energy

    def calc_de_reconfig(self, i, j):
        orig_Bx = self.Bx[i, j]
        orig_By = self.By[i, j]
        original_energy = self.current_energy
        self.Bx[i, j] *= -1
        self.By[i, j] *= -1
        new_energy = self.compute_energy()
        self.Bx[i, j] = orig_Bx
        self.By[i, j] = orig_By
        self.current_energy = original_energy
        return new_energy - original_energy

    def reconfig(self, i, j):
        self.Bx[i, j] *= -1
        self.By[i, j] *= -1
        self.az[i, j] = np.arctan2(self.By[i, j], self.Bx[i, j])

    @njit
    def _annealing_inner_loop(Bx, By, az, current_energy, tvar, neq, nx, ny, tfactr):
        nsucc = 0
        for k in range(neq):
            i = np.random.randint(0, nx)
            j = np.random.randint(0, ny)
            de = 0.0
            local_t = tvar[i, j]
            if local_t <= 0:
                if de < 0:
                    nsucc += 1
                    current_energy += de
                    Bx[i, j] = -Bx[i, j]
                    By[i, j] = -By[i, j]
                    az[i, j] = math.atan2(By[i, j], Bx[i, j])
            else:
                if (de < 0) or (np.random.rand() < math.exp(-de / local_t)):
                    nsucc += 1
                    current_energy += de
                    Bx[i, j] = -Bx[i, j]
                    By[i, j] = -By[i, j]
                    az[i, j] = math.atan2(By[i, j], Bx[i, j])
        for i in range(nx):
            for j in range(ny):
                tvar[i, j] *= tfactr
        return nsucc, current_energy

    def simulated_annealing(self, max_iter=1000, tol_conv=1e-5, nconv_min=50):
        strong_idx = np.column_stack(np.where(np.hypot(self.Bx, self.By) >= self.B_threshold))
        if strong_idx.size == 0:
            if self.verbose >= 1:
                print("No strong pixels above threshold; skipping simulated annealing.")
            return
        tvar = np.zeros_like(self.Bx, dtype=float)
        self.compute_energy()
        E = self.current_energy
        nx, ny = self.nx, self.ny
        neq_init = 200
        max_de = 0.0
        for _ in range(neq_init):
            rand_idx = np.random.choice(len(strong_idx))
            i, j = strong_idx[rand_idx]
            de = self.calc_de_reconfig(i, j)
            abs_de = abs(de)
            if abs_de > max_de:
                max_de = abs_de
            if abs_de > tvar[i, j]:
                tvar[i, j] = abs_de
            self.current_energy += de
            self.reconfig(i, j)
        tvar *= self.tfac0
        t = self.tfac0 * max_de
        self.compute_energy()
        E = self.current_energy
        E_prev = E
        tol_stop = 1e-7
        t_stop = tol_stop * t
        nconv = 0
        for iter_count in tqdm(range(max_iter), desc="Annealing iterations"):
            nsucc = 0
            for _ in range(self.neq):
                rand_idx = np.random.choice(len(strong_idx))
                i, j = strong_idx[rand_idx]
                de = self.calc_de_reconfig(i, j)
                local_t = tvar[i, j]
                if local_t <= 0:
                    if de < 0:
                        nsucc += 1
                        self.current_energy += de
                        self.reconfig(i, j)
                else:
                    if (de < 0) or (np.random.rand() < np.exp(-de / local_t)):
                        nsucc += 1
                        self.current_energy += de
                        self.reconfig(i, j)
            tvar *= self.tfactr
            t *= self.tfactr
            self.compute_energy()
            E = self.current_energy
            rel_diff = abs(E - E_prev) / (abs(E) + abs(E_prev) + 1e-12)
            if rel_diff < tol_conv:
                nconv += 1
            else:
                nconv = 0
            E_prev = E
            if (nsucc == 0) or (t < t_stop) or (nconv >= nconv_min):
                break
        self.compute_energy()
        if self.verbose >= 1:
            print("Final 'energy' =", self.current_energy)
            print("Final 'temperature' =", t)
            print("Total iterations =", iter_count + 1)

    def acute_angle(self, Bx_threshold=50.0, By_threshold=50.0, max_iterations=100, conv_threshold=1e-3):
        Bt = np.hypot(self.Bx, self.By)
        weak_mask = Bt < np.hypot(Bx_threshold, By_threshold)
        for it in range(max_iterations):
            prev_az = np.copy(self.az)
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    if not weak_mask[i, j]:
                        continue
                    Bx_avg = np.mean(self.Bx[i-1:i+2, j-1:j+2])
                    By_avg = np.mean(self.By[i-1:i+2, j-1:j+2])
                    dot = self.Bx[i, j] * Bx_avg + self.By[i, j] * By_avg
                    if dot < 0:
                        self.Bx[i, j] *= -1
                        self.By[i, j] *= -1
                        self.az[i, j] = np.arctan2(self.By[i, j], self.Bx[i, j])
            az_diff = np.max(np.abs(self.az - prev_az))
            if az_diff < conv_threshold:
                print(f"Acute-angle converged after {it+1} iterations")
                break
                
        self.az = (np.degrees(self.az) + 360) % 360
        self.compute_derivatives()


    def save_disambiguation_results(self):
        hdu1 = fits.PrimaryHDU(self.Bx)
        hdu2 = fits.ImageHDU(self.By)
        hdu3 = fits.ImageHDU(self.az)
        hdu4 = fits.ImageHDU(self.Jz)
        hdu1.header['COMMENT'] = "Bx"
        hdu2.header['COMMENT'] = "By"
        hdu3.header['COMMENT'] = "Azimuth (deg)"
        hdu4.header['COMMENT'] = "Vertical current density, Jz"
        hdul = fits.HDUList([hdu1, hdu2, hdu3, hdu4])
        hdul.writeto('MEM_disambiguation_results.fits', overwrite=True)
        print('Minimum energy disambiguation results saved!')
