import numpy as np
import time
from astropy.io import fits
from save_disambiguation_results import save_all_results 
from acute_angle_method import AAM_fields, AAM_azimuth
from minimum_energy_method import MEM_fields, MEM_azimuth
from non_potential_field_calculation import NPFC_fields, NPFC_azimuth
from disambiguation_helper import calculate_Jz, weak_field_mask

start_time = time.time()

B_orig = fits.open("pole_atm_original.fits")[0].data[4,30]
inc_orig = np.radians(fits.open("pole_atm_original.fits")[0].data[6,30])
az_orig = np.radians(fits.open("pole_atm_original.fits")[0].data[7,30])

B_ptpc = fits.open("deprojected_ptp_convolved.fits")[0].data[:,:,0]
inc_ptpc = fits.open("deprojected_ptp_convolved.fits")[0].data[:,:,1]
az_ptpc = fits.open("deprojected_ptp_convolved.fits")[0].data[:,:,2]

B_spc = fits.open("deprojected_spatially_coupled.fits")[0].data[:,:,0]
inc_spc = fits.open("deprojected_spatially_coupled.fits")[0].data[:,:,1]
az_spc = fits.open("deprojected_spatially_coupled.fits")[0].data[:,:,2]

lamda = 16 * 1e5

start_aam = time.time()
Bx_aam_orig, By_aam_orig = AAM_fields(B_orig, inc_orig, az_orig)
Bx_aam_ptpc, By_aam_ptpc = AAM_fields(B_ptpc, inc_ptpc, az_ptpc)
Bx_aam_spc, By_aam_spc = AAM_fields(B_spc, inc_spc, az_spc)
az_aam_orig = AAM_azimuth(az_orig)
print(f'AAM disambiguation completed! Time: {time.time() - start_aam:.2f} seconds')

start_mem = time.time()
Bx_mem_orig, By_mem_orig, Bz_mem_orig = MEM_fields(B_orig, inc_orig, az_orig, "orig")
Bx_mem_ptpc, By_mem_ptpc, Bz_mem_ptpc = MEM_fields(B_ptpc, inc_ptpc, az_ptpc, "ptpc")
Bx_mem_spc, By_mem_spc, Bz_mem_spc = MEM_fields(B_spc, inc_spc, az_spc, "spc")
az_mem_orig = MEM_azimuth(az_orig, Bx_mem_orig, By_mem_orig)
print(f'MEM disambiguation completed! Time: {time.time() - start_mem:.2f} seconds')

start_npfc = time.time()
Bx_npfc_orig, By_npfc_orig, Bz_npfc_orig = NPFC_fields(B_orig, inc_orig, az_orig, lamda)
Jz_orig = calculate_Jz(Bx_npfc_orig, By_npfc_orig, lamda)
Bx_npfc_ptpc, By_npfc_ptpc, Bz_npfc_ptpc = NPFC_fields(B_ptpc, inc_ptpc, az_ptpc, lamda)
Jz_ptpc = calculate_Jz(Bx_npfc_ptpc, By_npfc_ptpc, lamda)
Bx_npfc_spc, By_npfc_spc, Bz_npfc_spc = NPFC_fields(B_spc, inc_spc, az_spc, lamda)
Jz_spc = calculate_Jz(Bx_npfc_spc, By_npfc_spc, lamda)
az_npfc_orig = NPFC_azimuth(az_orig)
print(f'NPFC disambiguation completed! Time: {time.time() - start_npfc:.2f} seconds')

aam_results = {
    'Bx_aam_orig': Bx_aam_orig, 'By_aam_orig': By_aam_orig,
    'Bx_aam_ptpc': Bx_aam_ptpc, 'By_aam_ptpc': By_aam_ptpc,
    'Bx_aam_spc': Bx_aam_spc, 'By_aam_spc': By_aam_spc,
    'az_aam_orig': az_aam_orig
}

mem_results = {
    'Bx_mem_orig': Bx_mem_orig, 'By_mem_orig': By_mem_orig, 'Bz_mem_orig': Bz_mem_orig,
    'Bx_mem_ptpc': Bx_mem_ptpc, 'By_mem_ptpc': By_mem_ptpc, 'Bz_mem_ptpc': Bz_mem_ptpc,
    'Bx_mem_spc': Bx_mem_spc, 'By_mem_spc': By_mem_spc, 'Bz_mem_spc': Bz_mem_spc,
    'az_mem_orig': az_mem_orig
}

npfc_results = {
    'Bx_npfc_orig': Bx_npfc_orig, 'By_npfc_orig': By_npfc_orig, 'Bz_npfc_orig': Bz_npfc_orig, 'Jz_orig': Jz_orig,
    'Bx_npfc_ptpc': Bx_npfc_ptpc, 'By_npfc_ptpc': By_npfc_ptpc, 'Bz_npfc_ptpc': Bz_npfc_ptpc, 'Jz_ptpc': Jz_ptpc,
    'Bx_npfc_spc': Bx_npfc_spc, 'By_npfc_spc': By_npfc_spc, 'Bz_npfc_spc': Bz_npfc_spc, 'Jz_spc': Jz_spc,
    'az_npfc_orig': az_npfc_orig
}

save_all_results(aam_results, mem_results, npfc_results)
print(f'Disambiguation completed and results saved! Total Time: {time.time() - start_time:.2f} seconds')
