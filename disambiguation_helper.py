import numpy as np
import astropy.io.fits as fits

def load_fits_data(filepath, is_original=False):

    data = fits.open(filepath)[0].data
    if is_original:
        return data[4,30], np.radians(data[6,30]), np.radians(data[7,30])
    else:
        return data[:,:,0], data[:,:,1], data[:,:,2]

def compute_B_perp(B, inc):

    return B * np.sin(inc)

def weak_field_mask(B_perp, dataset_type):

    thresholds = {
        "orig": 70,
        "ptpc": 40,
        "spc": 55
    }

    if dataset_type not in thresholds:
        raise ValueError(f"Invalid dataset type '{dataset_type}'. Choose from 'orig', 'ptpc', 'spc'.")

    threshold = thresholds[dataset_type]
    return B_perp < threshold, threshold

def strong_field_mask(Bz, dataset_type):

    thresholds = {
        "orig": 150,
        "ptpc": 100,
        "spc": 120
    }

    if dataset_type not in thresholds:
        raise ValueError(f"Invalid dataset type '{dataset_type}'. Choose from 'orig', 'ptpc', 'spc'.")

    return np.abs(Bz) > thresholds[dataset_type]  # 

def conversion_to_heliographic(B, inc, az):

    B_r = B * np.cos(inc)
    B_theta = B * np.sin(inc) * np.cos(az)
    B_phi = B * np.sin(inc) * np.sin(az)
    
    return B_r, B_theta, B_phi

def compute_Bx_By(B_perp, az):

    Bx = B_perp * np.cos(az)
    By = B_perp * np.sin(az)
    return Bx, By




""" 
korisno
# Import necessary functions
from helper_functions import load_fits_data, compute_B_perp, weak_field_mask, strong_field_mask, conversion_to_heliographic, compute_Bx_By
from acute_angle_method import AAM
from minimum_energy_method import MEM
from non_potential_field_calculation import NPFC

# Define dataset names and file paths
datasets = {
    "orig": "pole_atm_original.fits",
    "ptpc": "deprojected_ptp_convolved.fits",
    "spc": "deprojected_spatially_coupled.fits"
}

# Initialize empty variables
B, inc, az, Btr, wr, strong_mask = {}, {}, {}, {}, {}, {}
B_r, B_theta, B_phi, Bx, By = {}, {}, {}, {}, {}

# Load data and compute necessary fields
for key, filename in datasets.items():
    B[key], inc[key], az[key] = load_fits_data(filename, is_original=(key == "orig"))
    Btr[key] = compute_B_perp(B[key], inc[key])
    wr[key], _ = weak_field_mask(Btr[key], key)
    strong_mask[key] = strong_field_mask(B[key], key)
    B_r[key], B_theta[key], B_phi[key] = conversion_to_heliographic(B[key], inc[key], az[key])
    Bx[key], By[key] = compute_Bx_By(Btr[key], az[key])

# Set pixel size for NPFC (1 pixel = 16 km → cm)
lamda = 16 * 1e5  

# Apply AAM, MEM, and NPFC using loops
Bx_aam, By_aam, Bx_mem, By_mem, Bz_mem, Bx_npfc, By_npfc, Bz_npfc, Jz = {}, {}, {}, {}, {}, {}, {}, {}, {}

for key in datasets.keys():
    # AAM
    Bx_aam[key], By_aam[key] = AAM(Bx[key], By[key], wr[key])

    # MEM
    Bx_mem[key], By_mem[key], Bz_mem[key] = MEM(Bx_aam[key], By_aam[key], B_r[key], key)

    # NPFC
    Bx_npfc[key], By_npfc[key], Bz_npfc[key], Jz[key] = NPFC(B[key], inc[key], az[key], lamda)




""" 