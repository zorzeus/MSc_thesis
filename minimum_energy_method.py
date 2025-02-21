import numpy as np
from scipy.fft import dst, idst  
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

def compute_current_density_and_divergence(Bx, By, Bz):

    dBy_dx = np.gradient(By, axis=1)
    dBx_dy = np.gradient(Bx, axis=0)
    Jz = dBy_dx - dBx_dy 
    
    dBr_dx = np.gradient(Bz, axis=0)
    dBtheta_dy = np.gradient(By, axis=1)
    dBphi_dphi = np.gradient(Bx, axis=1)
    DivB = dBr_dx + dBtheta_dy + dBphi_dphi
    
    return DivB, Jz

def solve_poisson_dst(Jz):

    Jz_transformed = dst(dst(Jz, type=1, axis=0), type=1, axis=1)
    
    nx, ny = Jz.shape
    kx = np.arange(1, nx + 1)[:, None]
    ky = np.arange(1, ny + 1)[None, :]
    k2 = kx**2 + ky**2
    k2[0, 0] = 1  
    
    Bz_transformed = Jz_transformed / k2
    Bz_corrected = idst(idst(Bz_transformed, type=1, axis=0), type=1, axis=1)
    
    return Bz_corrected

def iterative_local_adjustment(Bx, By, azimuth, strong_field_mask, iterations=10):

    Bx_new, By_new, az_new = np.copy(Bx), np.copy(By), np.copy(azimuth)
    nx, ny = Bx.shape

    for _ in range(iterations):
        az_smoothed = gaussian_filter(az_new, sigma=2)  
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                if not strong_field_mask[i, j]:
                    continue
                
                avg_neighbor_az = az_smoothed[i, j]
                flipped_angle = (az_new[i, j] + np.pi) % (2 * np.pi)

                if abs(flipped_angle - avg_neighbor_az) < abs(az_new[i, j] - avg_neighbor_az):
                    Bx_new[i, j] *= -1
                    By_new[i, j] *= -1
                    az_new[i, j] = flipped_angle

    return Bx_new, By_new, az_new

def smooth_field(Bz):

    x, y = np.arange(Bz.shape[0]), np.arange(Bz.shape[1])
    spline = RectBivariateSpline(x, y, Bz)
    return spline(x, y)

def compute_azimuth(Bx, By):
 
    azimuth = np.arctan2(By, Bx)
    return (azimuth + 2 * np.pi) % (2 * np.pi)

def MEM(Bx, By, Bz, strong_field_mask, dataset_type):

    DivB, Jz = compute_current_density_and_divergence(Bx, By, Bz)
    Bz_correction = solve_poisson_dst(Jz)

    Bz_corrected = np.copy(Bz)
    Bz_corrected[strong_field_mask] -= Bz_correction[strong_field_mask]

    Bz_smooth = smooth_field(Bz_corrected)

    azimuth = compute_azimuth(Bx, By)

    Bx_new, By_new, az_new = np.copy(Bx), np.copy(By), np.copy(azimuth)
    Bx_strong, By_strong, az_strong = iterative_local_adjustment(Bx, By, azimuth, strong_field_mask)

    Bx_new[strong_field_mask] = Bx_strong[strong_field_mask]
    By_new[strong_field_mask] = By_strong[strong_field_mask]
    az_new[strong_field_mask] = az_strong[strong_field_mask]

    print('MEM completed!')
    return Bx_new, By_new, Bz_smooth, az_new