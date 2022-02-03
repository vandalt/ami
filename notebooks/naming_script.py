from itertools import chain, combinations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nrm_analysis.misctools import oifits

noise_types = ['read','flat','dark','bkgd','jitt']


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

noise_comb = list(powerset(noise_types))

odir = '/ifs/jwst/wit/niriss/rcooper/ami_noise_sims/'

# with open("make_all_sims.sh", "w") as f:
#     f.write('#!/bin/sh')
#     f.write('\n')

def get_tag(noisetup):

        tag = ''
        if 'read' in noisetup:
            read = 1
            tag += 'read_'
        else:
            read = 0
        if 'dark' in noisetup:
            dark = 1
            tag += 'dark_'
        else:
            dark = 0
        if 'bkgd' in noisetup:
            bkgd = 1
            tag += 'bkgd_'
        else:
            bkgd = 0
        if 'jitt' in noisetup:
            jitt = 1
            tag += 'jitt_'
        else:
            jitt = 0
 
        if 'flat' in noisetup:
            flat = 0 # reverse of the others
            tag += 'flat_'
        else:
            flat = 1
        if tag == '':
            tag = 'photonly'

        return tag
 
        # for scene in binary_scenes:
        #     line1 = 'python driver_scene.py -t %s --tag %s -o 1 -utr 0 -f F380M' % (odir,tag)
        #     line2 = ' -p ../noise_tests/simulatedpsfs_for_ami_sim/F380M_81_flat_x11.fits -s %s' % scene
        #     line3 = ' -os 11 -I 10000 -G 1 -c 1 -cr 1e10 --random_seed 10 -v 0'
        #     line4 = ' --apply_dither 0 --apply_jitter %i --include_photnoise 1 --uniform_flatfield %i' % (jitt,flat)
        #     line5 = ' --include_readnoise %i --include_darkcurrent %i --include_background %i' % (read,dark,bkgd)
        #     command = line1+line2+line3+line4+line5
        #     print(command)
        #     # print(command)
        #     # print('')
        #     # f.write(command)
        #     # f.write('\n')

def fname_generator(targ,pa,sep,pipeline,noisetup):
    tag = get_tag(noisetup)
    pipe = ''
    if pipeline is not None:
        pipe = '_'+pipeline

    fname = '../data/corr_%s_pa%d_sep%d_F380M_sky_81px_x11__F380M_81_flat_x11_%s_00_mir%s.npz' % (targ,pa,sep,tag,
                                                                                                    pipe)
    return fname


def fname_generator_implaneia(targ,pa,sep,noisetup):
    tag = get_tag(noisetup)

    fname = '../data/ImPlaneIA/multi_%s_pa%d_sep%d_F380M_sky_81px_x11__F380M_81_flat_x11_%s_00_mir.oifits' % (targ,pa,sep,tag)
    return fname


def load_data(pa,sep,noisetup,pipeline,calibrated=False,verbose=False):
    assert pipeline.lower() in ['sampip','amical','implaneia']
    if pipeline.lower()=='sampip':
        calfile_sampip = fname_generator('c',pa,sep,None,noisetup)    
        targfile_sampip = fname_generator('t',pa,sep,None,noisetup)
        if verbose:
            print(calfile_sampip,targfile_sampip)
        cal_sampip = np.load(calfile_sampip)
        targ_sampip = np.load(targfile_sampip)

        v2_cal = cal_sampip['V2']
        cp_cal = cal_sampip['CP']

        v2_targ = targ_sampip['V2']
        cp_targ = targ_sampip['CP']

    elif pipeline.lower()=='amical':
        calfile_amical = fname_generator('c',pa,sep,'amical',noisetup)    
        targfile_amical = fname_generator('t',pa,sep,'amical',noisetup)
        if verbose:
            print(calfile_amical,targfile_amical)

        cal_amical = np.load(calfile_amical)
        targ_amical = np.load(targfile_amical)
        v2_cal = cal_amical['V2']
        cp_cal = cal_amical['CP']

        v2_targ = targ_amical['V2']
        cp_targ = targ_amical['CP']

    elif pipeline.lower()=='implaneia':
        calfile_implaneia = fname_generator_implaneia('c',pa,sep,noisetup)    
        targfile_implaneia = fname_generator_implaneia('t',pa,sep,noisetup)    
        if verbose:
            print(calfile_implaneia,targfile_implaneia)

        cal_implaneia = oifits.load(calfile_implaneia)
        targ_implaneia = oifits.load(targfile_implaneia)

        v2_cal = cal_implaneia['OI_VIS2']['VIS2DATA'].T
        cp_cal = cal_implaneia['OI_T3']['T3PHI'].T

        v2_targ = targ_implaneia['OI_VIS2']['VIS2DATA'].T
        cp_targ = targ_implaneia['OI_T3']['T3PHI'].T

    #amical

    if not calibrated:
        v2_calmean, v2_calstd, v2_calcov = np.mean(v2_cal,axis=0), np.std(v2_cal,axis=0), np.cov(v2_cal.T)
        cp_calmean, cp_calstd, cp_calcov = np.mean(cp_cal,axis=0), np.std(cp_cal, axis=0), np.cov(cp_cal.T)

        v2_targmean, v2_targstd, v2_targcov = np.mean(v2_targ,axis=0), np.std(v2_targ,axis=0), np.cov(v2_targ.T)
        cp_targmean, cp_targstd, cp_targcov = np.mean(cp_targ,axis=0), np.std(cp_targ, axis=0), np.cov(cp_targ.T)

        v2_mean = v2_calmean
        cp_mean = cp_calmean
        v2_std = v2_calstd
        cp_std = cp_calstd
        v2_cov = v2_calcov
        cp_cov = cp_calcov
    else:
        minshape = np.min([v2_cal.shape[0],v2_targ.shape[0]])
        v2_cal, v2_targ, cp_cal, cp_targ = v2_cal[:minshape,:], v2_targ[:minshape,:], cp_cal[:minshape,:], cp_targ[:minshape,:]

        v2_mean, v2_std, v2_cov = np.mean(v2_targ/v2_cal,axis=0), np.std(v2_targ/v2_cal,axis=0), np.cov((v2_targ/v2_cal).T)
        cp_mean, cp_std, cp_cov = np.mean(cp_targ-cp_cal,axis=0), np.std(cp_targ-cp_cal, axis=0), np.cov((cp_targ-cp_cal).T)

    return v2_mean, v2_std, v2_cov, cp_mean, cp_std, cp_cov
