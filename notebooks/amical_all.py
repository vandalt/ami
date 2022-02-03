import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import glob
from tqdm import tqdm_notebook as tqdm

import jax.numpy as jnp

from astropy.table import Table
import pandas as pd
from astropy.io import fits, ascii

import pygtc
import pdb

from readcol import * 
from make_mask import *
from naming_script import *

import nrm_analysis as implaneia
from nrm_analysis.misctools import mask_definitions
from nrm_analysis.misctools import oifits
from nrm_analysis.misctools.implane2oifits import calibrate_oifits

import amical

from itertools import product

import yaml

import os

f2f,mask = mask_definitions.jwst_g7s6c()


[files] = readcol('../data/noises_c.txt', twod=False)
[files2] = readcol('../data/noises_t.txt', twod=False)

import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False


import warnings
warnings.filterwarnings("ignore")


'''---------------------------------------------
amical_all.py

Wrapper for scripts by Rachel Cooper for using
ImPlaneIA and Candid to model injected binaries
in simulated JWST NIRISS AMI data.

---------------------------------------------'''

params_ami = {
    "peakmethod": "fft",
    "bs_multi_tri": False,
    "maskname": "g7",
    "fw_splodge": 0.7,
}

# thin wrapper to take in a .fits file with amical and produce a calibrated oifits

def amical_oifits(file_t,file_c,outname):
    
    hdu = fits.open(file_t)
    cube_t = hdu[1].data
    hdu.close()

    hdu = fits.open(file_c)
    cube_c = hdu[1].data
    hdu.close()

    # Extract raw complex observables for the target and the calibrator:
    # It's the core of the pipeline (amical/mf_pipeline/bispect.py)
    bs_t = amical.extract_bs(
        cube_t, file_t, targetname="fakebinary", **params_ami, display=False
    )
    bs_c = amical.extract_bs(
        cube_c, file_c, targetname="fakepsf", **params_ami, display=False
    )

    cal = amical.calibrate(bs_t, bs_c)
    dic, fname = amical.save(cal, oifits_file=outname,datadir='.')
    return dic, fname

# assemble lists of all data

indir = '../data/lower_contrast/'
odir = '../data/calibrated_low/'

if not os.path.exists(odir):
    os.makedirs(odir)
    print('Created output directory: %s' % odir)

targfiles1 = sorted(glob.glob(os.path.join(indir,'t_pa0*.fits'))) # or 'multi_t*oifits
targfiles2 = sorted(glob.glob(os.path.join(indir,'t_pa45*.fits')))
targfiles3 = sorted(glob.glob(os.path.join(indir,'t_pa90*.fits'))) 
calfiles = sorted(glob.glob(os.path.join(indir,'c_*.fits')))


# produce calibrated oifits files

for targfiles in [targfiles1,targfiles2,targfiles3]:
    for targ in targfiles:
        cal = targ.replace('/t_','/c_')
        sub1 = os.path.basename(targ).split('flat_x11')[-1]
        noise_str = sub1.split('_00')[0]
        blah = os.path.basename(targ).split('F')[0]
        cont_str = blah.split('_')[-2]
        pa_str = os.path.basename(targ).split('_')[1]
        outstr = pa_str + '_' + cont_str + noise_str
        # skip if already done
        if os.path.exists(odir+outstr+'.oifits'):
            print(odir+outstr+'.oifits already done')
            pass
        else:
            amical_oifits(targ,cal,odir+outstr+'.oifits')

calib_dir = odir

all_calib = sorted(glob.glob(os.path.join(calib_dir,'*.oifits')))

# now use candid

contrasts = [0.01,0.003,0.001]
pas = [0,45,90]

combinations = list(product(contrasts,pas))      
print(combinations)

odir = 'implaneia_candid_lowcon/'
if not os.path.exists(odir):
    os.mkdir(odir)
        
param_candid = {
    "rmin": 10,  # inner radius of the grid
    "rmax": 300,  # outer radius of the grid
    "step": 75,  # grid sampling
    "ncore": 1,  # multiprocessing.cpu_count()  # core for multiprocessing
}
def dm_to_cr(dm):
    # mag_diff = -2.5 * np.log10(flux_ratio)
    return 100**(-dm/5.)

fit_dict = {}

for (cont,pa) in combinations:
    print(cont,pa)
    fit_dict[str(cont)+'_'+str(pa)] = {}
    # set up table
    colnames = ["Noise Types", "Sep/mas", "PA/deg", "CR", "chi^2", "nsigma"]
    # make lists of strings for table columns
    noise_strs = []
    sep_fit = []
    theta_fit = []
    cr_fit = []
    nsigma, chi2 = [], []
    
    fn_list = sorted(glob.glob(os.path.join(calib_dir,'pa%s_con%s_*.oifits' % (str(pa),str(cont)))))
    for fn in fn_list:
        bn = os.path.basename(fn)
        if (str(cont) in bn) & (str(pa) in bn):
            #print('matched')
            sub = bn.split('_',2)[-1]
            noise_str = sub.split('_.')[0]
            
            fit = amical.candid_grid(fn, **param_candid, diam=0, doNotFit=["diam*"], save=True)
            
            fit_dict[str(cont)+'_'+str(pa)][noise_str] = fit
            dm, dm_uncer = fit["best"]["dm"], fit["uncer"]["dm"]
            cr, cr_uncer = dm_to_cr(dm), dm_to_cr(dm_uncer)
            noise_strs.append(noise_str)
            sep_fit.append("%.3f +/- %.2f" % (fit["best"]["sep"], fit["uncer"]["sep"]))
            theta_fit.append("%.3f +/- %.2f" % (fit["best"]["theta"], fit["uncer"]["theta"]))
            cr_fit.append("%.3e +/- %.2f" % (cr,cr_uncer))
            nsigma.append("%.2f" % fit["nsigma"])
            chi2.append("%.2f" % fit["chi2"])
    t = Table([noise_strs, sep_fit, theta_fit, cr_fit, chi2, nsigma], names=colnames)
    print(t)
    tablefn = os.path.join(odir,'contrast%s_pa%s_noisysims_fit.dat' % (cont,pa))
    ascii.write(
        t, tablefn, format="fixed_width_two_line", delimiter_pad=" ", overwrite=True
    )


# dump to yaml
yml_file = "all_lowcon_candid_fits.yml"
with open(yml_file, "w") as f:
    yaml.dump(fit_dict, f)  