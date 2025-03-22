import numpy as np
import numba as nb
from tqdm import tqdm
from scipy.ndimage import binary_erosion, binary_dilation, uniform_filter, median_filter
from astropy.io import fits

@nb.njit
def anneal_loop(Bx, By, DivB, Jz, lambda_w, T, max_attempts):
    accepted = 0
    nxi, nyi = Bx.shape
    for _ in range(max_attempts):
        i = np.random.randint(1, nxi - 1)
        j = np.random.randint(1, nyi - 1)
        old_Bx = Bx[i, j]
        old_By = By[i, j]
        cells = [(i-1, j-1), (i-1, j), (i, j-1), (i, j)]
        div_old = np.empty(4, dtype=DivB.dtype)
        jz_old = np.empty(4, dtype=Jz.dtype)
        for idx in range(4):
            ic, jc = cells[idx]
            div_old[idx] = DivB[ic, jc]
            jz_old[idx] = Jz[ic, jc]
        Bx[i, j] = -old_Bx
        By[i, j] = -old_By
        local_dE = 0.0
        for idx in range(4):
            ic, jc = cells[idx]
            div_new = 0.5 * ((Bx[ic+1, jc] - Bx[ic, jc] + Bx[ic+1, jc+1] - Bx[ic, jc+1]) +
                             (By[ic, jc+1] - By[ic, jc] + By[ic+1, jc+1] - By[ic+1, jc]))
            jz_new = 0.5 * ((Bx[ic+1, jc] - Bx[ic, jc] + Bx[ic+1, jc+1] - Bx[ic, jc+1]) -
                            (By[ic, jc+1] - By[ic, jc] + By[ic+1, jc+1] - By[ic+1, jc]))
            DivB[ic, jc] = div_new
            Jz[ic, jc] = jz_new
            local_dE += (abs(div_new) + lambda_w * abs(jz_new)) - (abs(div_old[idx]) + lambda_w * abs(jz_old[idx]))
        if local_dE <= 0 or np.random.rand() < np.exp(-local_dE / T):
            accepted += 1
        else:
            Bx[i, j] = old_Bx
            By[i, j] = old_By
            for idx in range(4):
                ic, jc = cells[idx]
                DivB[ic, jc] = div_old[idx]
                Jz[ic, jc] = jz_old[idx]
    return accepted

class MinimumEnergyDisambiguator:
    def __init__(self, Bx, By, Bz, *, weight_current=None, initial_T=None, cooling_rate=0.9, 
                 max_iter=500, neq=100, pad=100, vert_method='FFT', polar=False, theta=30, phi=0, p_angle=0, b_angle=0):
        self.Bx = np.array(Bx, dtype=np.float64)
        self.By = np.array(By, dtype=np.float64)
        self.Bz = np.array(Bz, dtype=np.float64)
        self.nx, self.ny = self.Bx.shape
        self.weight_current = weight_current
        self.initial_T = initial_T
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.neq = neq
        self.pad = pad
        self.vert_method = vert_method.upper()
        self.polar = polar
        self.theta = theta
        self.phi = phi
        self.p_angle = p_angle
        self.b_angle = b_angle
        self.az = None
        self.DivB = np.zeros_like(self.Bx)
        self.Jz = np.zeros_like(self.Bx)
        
    def set_polar_coordinates(self, theta, phi, p_angle, b_angle):
        T = self.compute_transformation_matrix(theta, phi, p_angle, b_angle)
        self.Bx, self.By, self.Bz = self.transform_field(self.Bx, self.By, self.Bz, T)
        self.az = np.arctan2(self.By, self.Bx)
        self._compute_divergence_current()
        
    def compute_transformation_matrix(self, theta, phi, p_angle, b_angle):
        theta_r = np.radians(theta)
        phi_r = np.radians(phi)
        P_r = np.radians(p_angle)
        B0_r = np.radians(b_angle)
        a11 = -np.sin(B0_r)*np.sin(P_r)*np.sin(theta_r) + np.cos(P_r)*np.cos(theta_r)
        a12 = -np.sin(phi_r)*(np.sin(B0_r)*np.sin(P_r)*np.cos(theta_r) + np.cos(P_r)*np.sin(theta_r)) - np.cos(phi_r)*np.cos(B0_r)*np.sin(P_r)
        a13 = -np.cos(B0_r)*np.sin(theta_r)
        a21 = np.sin(B0_r)*np.cos(P_r)*np.sin(theta_r) + np.sin(P_r)*np.cos(theta_r)
        a22 = np.sin(phi_r)*(np.sin(B0_r)*np.cos(P_r)*np.cos(theta_r) - np.sin(P_r)*np.sin(theta_r)) + np.cos(phi_r)*np.cos(B0_r)*np.cos(P_r)
        a23 = -np.cos(B0_r)*np.sin(phi_r)*np.cos(theta_r) + np.sin(B0_r)*np.cos(phi_r)
        a31 = np.cos(phi_r)*(np.sin(B0_r)*np.sin(P_r)*np.cos(theta_r) + np.cos(P_r)*np.sin(theta_r)) - np.sin(phi_r)*np.cos(B0_r)*np.sin(P_r)
        a32 = -np.cos(phi_r)*(np.sin(B0_r)*np.cos(P_r)*np.cos(theta_r) - np.sin(P_r)*np.sin(theta_r)) + np.sin(phi_r)*np.cos(B0_r)*np.cos(P_r)
        a33 = np.cos(phi_r)*np.cos(B0_r)*np.cos(theta_r) + np.sin(phi_r)*np.sin(B0_r)
        return {'a11': a11, 'a12': a12, 'a13': a13,
                'a21': a21, 'a22': a22, 'a23': a23,
                'a31': a31, 'a32': a32, 'a33': a33}
        
    def transform_field(self, Bx, By, Bz, T):
        Bx_new = T['a11']*Bx + T['a12']*By + T['a13']*Bz
        By_new = T['a21']*Bx + T['a22']*By + T['a23']*Bz
        Bz_new = T['a31']*Bx + T['a32']*By + T['a33']*Bz
        return Bx_new, By_new, Bz_new
        
    def _cosine_taper(self, N, width):
        taper = np.ones(N)
        for i in range(width):
            taper[i] = 0.5 * (1 - np.cos(np.pi * (i+1) / width))
            taper[-(i+1)] = 0.5 * (1 - np.cos(np.pi * (i+1) / width))
        return taper
        
    def _pad_and_apodize(self, field, pad=None):
        if pad is None:
            pad = self.pad
        if pad <= 0:
            return field
        field_np = field
        padded = np.pad(field_np, pad_width=pad, mode='constant')
        ny_p, nx_p = padded.shape
        taper_x = self._cosine_taper(nx_p, pad)
        taper_y = self._cosine_taper(ny_p, pad)
        window = np.outer(taper_y, taper_x)
        padded *= window
        return padded
        
    def _compute_vertical_derivative_Bz(self, Bz):
        if self.vert_method == "FFT":
            pad = max(1, min(self.nx // 8, self.ny // 8))
            Bz0 = Bz - np.mean(Bz)
            Bz_padded = self._pad_and_apodize(Bz0, pad=pad)
            Bz_fft = np.fft.rfft2(Bz_padded)
            ny_p, nx_p = Bz_padded.shape
            kx = np.fft.fftfreq(nx_p) * 2 * np.pi
            ky = np.fft.fftfreq(ny_p) * 2 * np.pi
            kx_grid, ky_grid = np.meshgrid(kx[:nx_p//2+1], ky)
            k = np.sqrt(kx_grid**2 + ky_grid**2)
            k[0, 0] = 1e-10
            dBz_dz_fft = -k * Bz_fft
            dBz_dz_padded = np.fft.irfft2(dBz_dz_fft, s=Bz_padded.shape)
            dBz_dz = dBz_dz_padded[pad:pad+self.nx, pad:pad+self.ny]
            return dBz_dz
        elif self.vert_method == "FD":
            dz = self.dz
            dBz_dz = np.zeros_like(Bz)
            dBz_dz[1:-1, :] = (Bz[2:, :] - Bz[:-2, :]) / (2*dz)
            dBz_dz[0, :] = (Bz[1, :] - Bz[0, :]) / dz
            dBz_dz[-1, :] = (Bz[-1, :] - Bz[-2, :]) / dz
            return dBz_dz
        else:
            return np.zeros_like(Bz)
            
    def _compute_divergence_current(self):
        dBx_dx_top = self.Bx[1:, :-1] - self.Bx[:-1, :-1]
        dBx_dx_bottom = self.Bx[1:, 1:] - self.Bx[:-1, 1:]
        dBy_dy_left = self.By[:-1, 1:] - self.By[:-1, :-1]
        dBy_dy_right = self.By[1:, 1:] - self.By[1:, :-1]
        DivB_xy = 0.5 * (dBx_dx_top + dBx_dx_bottom + dBy_dy_left + dBy_dy_right)
        Jz_xy = 0.5 * (dBx_dx_top + dBx_dx_bottom) - 0.5 * (dBy_dy_left + dBy_dy_right)
        dBz_dz = self._compute_vertical_derivative_Bz(self.Bz)
        dBz_dz_int = dBz_dz[:-1, :-1]
        self.DivB = DivB_xy + dBz_dz_int
        self.Jz = Jz_xy
        if self.weight_current is None:
            div_std = np.std(self.DivB)
            jz_std = np.std(self.Jz)
            self.weight_current = div_std / jz_std if jz_std > 0 else 1.0
            print('Chosen weight current is :', self.weight_current)
            
    def compute_energy(self):
        self._compute_divergence_current()
        return np.sum(np.abs(self.DivB)) + self.weight_current * np.sum(np.abs(self.Jz))
        
    def simulated_annealing_disambiguation(self, strong_threshold=50.0, morphology=True):
        self._compute_divergence_current()
        Bt = np.hypot(self.Bx, self.By)
        strong_mask = Bt >= strong_threshold
        if morphology:
            strong_mask_cpu = strong_mask.copy()
            core_mask = binary_erosion(strong_mask_cpu, structure=np.ones((7,7)))
            strong_mask_cpu = binary_dilation(core_mask, structure=np.ones((7,7)), iterations=3)
            strong_mask = strong_mask_cpu
        dEs = []
        for _ in range(1000):
            i = np.random.randint(1, self.nx - 1)
            j = np.random.randint(1, self.ny - 1)
            if not strong_mask[i, j]:
                continue
            orig_val = (self.Bx[i, j], self.By[i, j])
            cells = [(i-1, j-1), (i-1, j), (i, j-1), (i, j)]
            old_vals = [(self.DivB[c], self.Jz[c]) for c in cells]
            self.Bx[i, j] = -orig_val[0]
            self.By[i, j] = -orig_val[1]
            local_dE = 0.0
            for idx, (ic, jc) in enumerate(cells):
                div_new = 0.5 * ((self.Bx[ic+1, jc] - self.Bx[ic, jc] + self.Bx[ic+1, jc+1] - self.Bx[ic, jc+1]) +
                                 (self.By[ic, jc+1] - self.By[ic, jc] + self.By[ic+1, jc+1] - self.By[ic+1, jc]))
                jz_new = 0.5 * ((self.Bx[ic+1, jc] - self.Bx[ic, jc] + self.Bx[ic+1, jc+1] - self.Bx[ic, jc+1]) -
                                (self.By[ic, jc+1] - self.By[ic, jc] + self.By[ic+1, jc+1] - self.By[ic+1, jc]))
                self.DivB[ic, jc] = div_new
                self.Jz[ic, jc] = jz_new
                div_old, jz_old = old_vals[idx]
                local_dE += (abs(div_new) + self.weight_current * abs(jz_new)) - (abs(div_old) + self.weight_current * abs(jz_old))
            self.Bx[i, j], self.By[i, j] = orig_val
            for idx, (ic, jc) in enumerate(cells):
                self.DivB[ic, jc], self.Jz[ic, jc] = old_vals[idx]
            if local_dE != 0:
                dEs.append(local_dE)
        dEs = np.array(dEs)
        T_est = 3.0 * np.std(dEs) if dEs.size > 0 else 1.0
        if T_est <= 0:
            T_est = 1.0
        if self.initial_T is None:
            self.initial_T = T_est
            print('Chosen initial temperature is: ', self.initial_T)
        max_attempts = self.neq**2
        current_T = self.initial_T
        E_prev = self.compute_energy()
        tol_conv = 1e-5
        stagnant_count = 0
        for iteration in tqdm(range(self.max_iter), desc="Simulated Annealing"):
            accepted = anneal_loop(self.Bx, self.By, self.DivB, self.Jz, self.weight_current, current_T, max_attempts)
            self._compute_divergence_current()
            E_new = self.compute_energy()
            rel_change = abs(E_new - E_prev) / (abs(E_new) + abs(E_prev) + 1e-12)
            stagnant_count = stagnant_count + 1 if rel_change < tol_conv else 0
            E_prev = E_new
            current_T *= self.cooling_rate
            if accepted == 0 or current_T < 1e-6 * self.initial_T or stagnant_count >= 5:
                break
        self.az = np.degrees(np.arctan2(self.By, self.Bx)) % 360.0
        return self.Bx, self.By, self.Bz
        
    def acute_angle_disambiguation(self, weak_threshold, max_iterations=1000):
        Bx_pot, By_pot = self.potential_field()
        if self.az is None:
            self.az = np.degrees(np.arctan2(self.By, self.Bx)) % 360.0
        B_perp = np.hypot(self.Bx, self.By)
        weak_mask = B_perp < weak_threshold
        for _ in range(max_iterations):
            prev_az = self.az.copy()
            dot = self.Bx * Bx_pot + self.By * By_pot
            flip_idx = weak_mask & (dot < 0)
            if not flip_idx.any():
                break
            self.Bx[flip_idx] *= -1
            self.By[flip_idx] *= -1
            self.az[flip_idx] = (self.az[flip_idx] + 180.0) % 360.0
            if np.max(np.abs(self.az - prev_az)) < 1e-4:
                break
        self._compute_divergence_current()
        
    def potential_field(self):
        pad = max(1, min(self.nx // 8, self.ny // 8))
        Bz0 = self.Bz - np.mean(self.Bz)
        Bz_padded = self._pad_and_apodize(Bz0, pad=pad)
        Bz_fft = np.fft.rfft2(Bz_padded)
        ny_p, nx_p = Bz_padded.shape  
        kx = np.fft.fftfreq(nx_p) * 2 * np.pi
        ky = np.fft.fftfreq(ny_p) * 2 * np.pi
        kx_grid, ky_grid = np.meshgrid(kx[:nx_p//2+1], ky)
        k = np.sqrt(kx_grid**2 + ky_grid**2)
        k[0, 0] = 1e-10
        Bx_pot_fft = 1j * (kx_grid / k) * Bz_fft
        By_pot_fft = 1j * (ky_grid / k) * Bz_fft
        Bx_pot_padded = np.fft.irfft2(Bx_pot_fft, s=Bz_padded.shape)
        By_pot_padded = np.fft.irfft2(By_pot_fft, s=Bz_padded.shape)
        Bx_pot = Bx_pot_padded[pad:pad+self.nx, pad:pad+self.ny]
        By_pot = By_pot_padded[pad:pad+self.nx, pad:pad+self.ny]
        return Bx_pot, By_pot

    def clean_boxcar(self, n_iter=1, window_size=3):
        for _ in range(n_iter):
            sum_Bx = uniform_filter(self.Bx, size=window_size, mode='reflect') * (window_size**2)
            sum_By = uniform_filter(self.By, size=window_size, mode='reflect') * (window_size**2)
            smooth_Bx = (sum_Bx - self.Bx) / (window_size**2 - 1)
            smooth_By = (sum_By - self.By) / (window_size**2 - 1)
            dot = smooth_Bx * self.Bx + smooth_By * self.By
            flip_idx = dot <= 0
            self.Bx[flip_idx] *= -1
            self.By[flip_idx] *= -1
            self.az = np.degrees(np.arctan2(self.By, self.Bx)) % 360.0
            self._compute_divergence_current()
            
    def clean_median(self, n_iter=1, window_size=3):
        footprint = np.ones((window_size, window_size))
        footprint[window_size//2, window_size//2] = 0
        for _ in range(n_iter):
            smooth_Bx = median_filter(self.Bx, footprint=footprint, mode='reflect')
            smooth_By = median_filter(self.By, footprint=footprint, mode='reflect')
            dot = smooth_Bx * self.Bx + smooth_By * self.By
            flip_idx = dot <= 0
            self.Bx[flip_idx] *= -1
            self.By[flip_idx] *= -1
            self.az = np.degrees(np.arctan2(self.By, self.Bx)) % 360.0
            self._compute_divergence_current()
            
    def clean_disambiguation(self, cleaning_method=None, n_iter=1, window_size=3):
        if cleaning_method is None:
            return
        if cleaning_method.lower() == 'boxcar':
            self.clean_boxcar(n_iter=n_iter, window_size=window_size)
        elif cleaning_method.lower() == 'median':
            self.clean_median(n_iter=n_iter, window_size=window_size)
        else:
            print("Unknown cleaning method. Use 'boxcar' or 'median'.")
            
    def save_disambiguation_results(self, filename='MEM_disambiguation_results.fits'):
        Bx_out = self.Bx
        By_out = self.By
        az_out = self.az
        Jz_out = self.Jz
        hdu_primary = fits.PrimaryHDU(Bx_out)
        hdu_by = fits.ImageHDU(By_out, name='By')
        hdu_az = fits.ImageHDU(az_out, name='Azimuth')
        hdu_jz = fits.ImageHDU(Jz_out, name='Jz')
        hdu_primary.header['COMMENT'] = "Disambiguated Bx"
        hdu_by.header['COMMENT'] = "Disambiguated By"
        hdu_az.header['COMMENT'] = "Azimuth (deg) of B vector"
        hdu_jz.header['COMMENT'] = "Vertical current density Jz"
        hdul = fits.HDUList([hdu_primary, hdu_by, hdu_az, hdu_jz])
        hdul.writeto(filename, overwrite=True)
        print(f"Results saved to {filename}")
