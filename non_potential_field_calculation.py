import numpy as np
from scipy.ndimage import gaussian_filter

def calculate_Jz(Bx, By):

    dBy_dx = np.gradient(By, axis=1)
    dBx_dy = np.gradient(Bx, axis=0)
    Jz = dBy_dx - dBx_dy
    return Jz

def NPFC_fields(B, inc, az, lambda_param, max_iterations=100, min_changes=1, smoothing_sigma=1.5):

    B_r = B * np.cos(inc)
    B_theta = B * np.sin(inc) * np.cos(az)
    B_phi = B * np.sin(inc) * np.sin(az)

    Bx, By, Bz = B_theta, B_phi, B_r

    Jz = calculate_Jz(Bx, By)

    iteration_count = 0
    while iteration_count < max_iterations:
        iteration_count += 1
        changes = 0

        Bx_smooth = gaussian_filter(Bx, sigma=smoothing_sigma)
        By_smooth = gaussian_filter(By, sigma=smoothing_sigma)

        for i in range(1, Bz.shape[0] - 1):
            for j in range(1, Bz.shape[1] - 1):
         
                Bp_x = (Bx_smooth[i - 1, j] + Bx_smooth[i + 1, j]) / 2
                Bp_y = (By_smooth[i, j - 1] + By_smooth[i, j + 1]) / 2

                energy_1 = (Bp_x - Bx[i, j]) ** 2 + (Bp_y - By[i, j]) ** 2
                energy_2 = (Bp_x + Bx[i, j]) ** 2 + (Bp_y + By[i, j]) ** 2

                if energy_2 < energy_1:
                    Bz[i, j] *= -1
                    changes += 1

        if changes < min_changes:
            break

    print(f"NPFC converged after {iteration_count} iterations with {changes} pixel flips.")
    return Bx, By, Bz

def NPFC_azimuth(azimuth, max_iterations=10, threshold=1e-3):

    azimuth = np.array(azimuth, dtype=np.float64)
    azimuth = np.atleast_2d(azimuth)
    shape = azimuth.shape

    for iteration in range(int(max_iterations)):
        az_prev = azimuth.copy()

        for i in range(1, shape[0] - 1):
            for j in range(1, shape[1] - 1):
     
                neighbors = azimuth[i-1:i+2, j-1:j+2].ravel()
                valid_neighbors = neighbors[neighbors != azimuth[i, j]]

          
                if len(valid_neighbors) == 0:
                    continue

                # Compute current and flipped cost
                current_angle = azimuth[i, j]
                flipped_angle = (current_angle + 180) % 360

                current_cost = np.sum(np.abs(np.mod(valid_neighbors - current_angle + 180, 360) - 180))
                flipped_cost = np.sum(np.abs(np.mod(valid_neighbors - flipped_angle + 180, 360) - 180))

                # Apply flip if it reduces local inconsistency
                if flipped_cost < current_cost:
                    azimuth[i, j] = flipped_angle

        # Compute the max change to check convergence
        change = np.max(np.abs(azimuth - az_prev))
        if change < threshold:
            print(f"NPFC azimuth converged after {iteration+1} iterations.")
            break

    return azimuth