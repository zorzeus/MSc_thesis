import numpy as np
from disambiguation_helper import weak_field_mask

def AAM_azimuth(azimuth, max_iterations=10, threshold=1e-2):

    shape = azimuth.shape
    updated_azimuth = azimuth.copy()

    for _ in range(max_iterations):
        azimuth_prev = updated_azimuth.copy()

        for i in range(1, shape[0] - 1):
            for j in range(1, shape[1] - 1):
                neighbors = [
                    updated_azimuth[i-1, j], updated_azimuth[i+1, j],
                    updated_azimuth[i, j-1], updated_azimuth[i, j+1],
                    updated_azimuth[i-1, j-1], updated_azimuth[i-1, j+1],
                    updated_azimuth[i+1, j-1], updated_azimuth[i+1, j+1]
                ]

                neighbors = [n for n in neighbors if not np.isnan(n)]
                if not neighbors:
                    continue

                current_angle = updated_azimuth[i, j]
                flipped_angle = (current_angle + 180) % 360

                current_cost = sum(1.0 / (abs(current_angle - n) + 1e-3) for n in neighbors)
                flipped_cost = sum(1.0 / (abs(flipped_angle - n) + 1e-3) for n in neighbors)

                if flipped_cost < current_cost:
                    updated_azimuth[i, j] = flipped_angle

        change = np.max(np.abs(updated_azimuth - azimuth_prev))
        if change < threshold:
            print(f"AAM converged after {_+1} iterations")
            break
            
    return updated_azimuth