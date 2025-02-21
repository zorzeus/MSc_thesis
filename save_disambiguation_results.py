import numpy as np
from astropy.io import fits

def save_fits(filename, data_dict, description_dict):

    hdul = fits.HDUList([fits.PrimaryHDU()]) 
    
    for key, data in data_dict.items():
        hdu = fits.ImageHDU(data.astype(np.float32), name=key)  
        hdu.header['COMMENT'] = description_dict.get(key, "No description provided")
        hdul.append(hdu)
    
    hdul.writeto(filename, overwrite=True)
    print(f"Saved: {filename}")

def save_all_results(aam_results, mem_results, npfc_results):

    aam_descriptions = {
        'Bx_aam_orig': 'Bx disambiguated (AAM) - Original Data',
        'By_aam_orig': 'By disambiguated (AAM) - Original Data',
        'Bx_aam_ptpc': 'Bx disambiguated (AAM) - PTP Convolved',
        'By_aam_ptpc': 'By disambiguated (AAM) - PTP Convolved',
        'Bx_aam_spc': 'Bx disambiguated (AAM) - Spatially Coupled',
        'By_aam_spc': 'By disambiguated (AAM) - Spatially Coupled',
        'az_aam_orig': 'Azimuth disambiguated (AAM)'
    }
    save_fits('aam_disambiguated_results.fits', aam_results, aam_descriptions)

    mem_descriptions = {
        'Bx_mem_orig': 'Bx disambiguated (MEM) - Original Data',
        'By_mem_orig': 'By disambiguated (MEM) - Original Data',
        'Bz_mem_orig': 'Bz disambiguated (MEM) - Original Data',
        'Bx_mem_ptpc': 'Bx disambiguated (MEM) - PTP Convolved',
        'By_mem_ptpc': 'By disambiguated (MEM) - PTP Convolved',
        'Bz_mem_ptpc': 'Bz disambiguated (MEM) - PTP Convolved',
        'Bx_mem_spc': 'Bx disambiguated (MEM) - Spatially Coupled',
        'By_mem_spc': 'By disambiguated (MEM) - Spatially Coupled',
        'Bz_mem_spc': 'Bz disambiguated (MEM) - Spatially Coupled',
        'az_mem_orig': 'Azimuth disambiguated (MEM)'
    }
    save_fits('mem_disambiguated_results.fits', mem_results, mem_descriptions)

    npfc_descriptions = {
        'Bx_npfc_orig': 'Bx disambiguated (NPFC) - Original Data',
        'By_npfc_orig': 'By disambiguated (NPFC) - Original Data',
        'Bz_npfc_orig': 'Bz disambiguated (NPFC) - Original Data',
        'Jz_orig': 'Jz computed (NPFC) - Original Data',
        'Bx_npfc_ptpc': 'Bx disambiguated (NPFC) - PTP Convolved',
        'By_npfc_ptpc': 'By disambiguated (NPFC) - PTP Convolved',
        'Bz_npfc_ptpc': 'Bz disambiguated (NPFC) - PTP Convolved',
        'Jz_ptpc': 'Jz computed (NPFC) - PTP Convolved',
        'Bx_npfc_spc': 'Bx disambiguated (NPFC) - Spatially Coupled',
        'By_npfc_spc': 'By disambiguated (NPFC) - Spatially Coupled',
        'Bz_npfc_spc': 'Bz disambiguated (NPFC) - Spatially Coupled',
        'Jz_spc': 'Jz computed (NPFC) - Spatially Coupled',
        'az_npfc_orig': 'Azimuth disambiguated (NPFC)'
    }
    save_fits('npfc_disambiguated_results.fits', npfc_results, npfc_descriptions)

"""
fits_files = {
    "aam": "aam_disambiguated_results.fits",
    "mem": "mem_disambiguated_results.fits",
    "npfc": "npfc_disambiguated_results.fits"
}

results = {}

for method, filename in fits_files.items():
    with fits.open(filename) as hdul:
        results[f"Bx_{method}_orig"] = hdul[f"Bx_{method}_orig"].data
        results[f"By_{method}_orig"] = hdul[f"By_{method}_orig"].data
        results[f"az_{method}_orig"] = hdul[f"az_{method}_orig"].data
        if method != "aam":  
            results[f"Bz_{method}_orig"] = hdul[f"Bz_{method}_orig"].data

Bx_aam_orig, By_aam_orig, az_aam_orig = results["Bx_aam_orig"], results["By_aam_orig"], results["az_aam_orig"]
Bx_mem_orig, By_mem_orig, Bz_mem_orig, az_mem_orig = results["Bx_mem_orig"], results["By_mem_orig"], results["Bz_mem_orig"], results["az_mem_orig"]
Bx_npfc_orig, By_npfc_orig, Bz_npfc_orig, az_npfc_orig = results["Bx_npfc_orig"], results["By_npfc_orig"], results["Bz_npfc_orig"], results["az_npfc_orig"]
"""


