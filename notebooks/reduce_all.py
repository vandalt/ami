import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import glob
from tqdm import tqdm

import jax.numpy as jnp

from astropy.table import Table
import pandas as pd
from astropy.io import fits

import pygtc
import pdb

from readcol import * 
from make_mask import *
from naming_script import *

import nrm_analysis as implaneia
from nrm_analysis.misctools import mask_definitions
from nrm_analysis.misctools import oifits

f2f,mask = mask_definitions.jwst_g7s6c()

import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False


'''-----------------------------------------------------------------------------
reduce_all.py

Run the scripts in covariance.ipynb automatically for all datasets

-----------------------------------------------------------------------------'''


# load in sampip metadata
sampip_baselines = Table.read('../data/sampip_baselines.csv',format='ascii')
sampip_mask = Table.read('../data/sampip_mask.csv',format='ascii')
sampip_baselines['indices']=list(sampip_baselines['indices'])

sampip_u1=np.array([1.139,1.139,1.139,1.139,1.139,1.139,1.139,1.139,1.139,
           -1.143,-1.143,-1.143,-3.425,-3.425,-3.425,-0.,-0.,-0.,
           -0.,-2.282,-2.282,-2.282,-4.564,-4.564,-4.564,-2.282,
           -2.282,-2.282,-4.564,-4.564,-4.564,-2.282,-2.282,-2.282,0.,])
sampip_v1=np.array([-0.663,-0.663,-0.663,-0.663,-0.663,-1.98,-1.98,-1.98,
           -1.98,-4.615,-4.615,-4.615,-3.297,-3.297,-0.663,-1.317,
           -1.317,-1.317,-1.317,-3.952,-3.952,-3.952,-2.634,-2.634,
           0.,-2.635,-2.635,-2.635,-1.317,-1.317,1.317,1.318,1.318,3.952,2.634])
sampip_u2=np.array([-0.,-2.282,-4.564,-4.564,-3.425,-2.282,-4.564,-4.564,-3.425,
           -2.282,-2.282,-1.143,0.,1.139,1.139,-2.282,-4.564,-4.564,
           -3.425,-2.282,-2.282,-1.143,0.,1.139,1.139,-2.282,-2.282,
           -1.143,0.,1.139,1.139,0.,1.139,1.139,1.139])
sampip_v2=np.array([-1.317,-3.952,-2.634,0.,0.663,-2.635,-1.317,1.317,1.98,1.318,
           3.952,4.615,2.634,3.297,0.663,-2.635,-1.317,1.317,1.98,1.318,
           3.952,4.615,2.634,3.297,0.663,1.318,3.952,4.615,2.634,3.297,
           0.663,2.634,3.297,0.663,0.663])

sampip_b = np.sqrt(sampip_baselines['x']**2 + sampip_baselines['y']**2)
sampip_sumbl = np.sqrt(sampip_u1**2+sampip_v1**2) + np.sqrt(sampip_u2**2+sampip_v2**2)

sampip_indices = Table.read('../data/sampip_indices.txt',format='ascii')
sampip_triangles = Table.read('../data/sampip_triangles.txt',format='ascii')

sampip_hole_triangles = np.zeros((35,3),dtype='int')
for k,triangle in enumerate(sampip_triangles):
#     print(np.array(triangle))
    b1 = set(sampip_indices[triangle[0]-1])
    b2 = set(sampip_indices[triangle[1]-1])
    b3 = set(sampip_indices[triangle[2]-1])
    this_tri = b1.union(b2).union(b3)
    sampip_hole_triangles[k,:] = list(this_tri)

# generate implaneia metadata
barray, bls = makebaselines(mask)
triples, uvs = maketriples_all(mask)

b = np.sqrt(np.sum(bls**2,axis=1))
sumbl = np.sum(np.sqrt((uvs)**2).sum(axis=2),axis=1)

'''-----------------------------------------------------------------------------
-----------------------------------------------------------------------------'''


all_data = glob.glob('../data/*.npz')
ia_data = glob.glob('../data/ImPlaneIA/*.oifits')


def baseline_plots(pa,sep,noisetup,calibrated=False):
	fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14.0,6.0))

	# get filenames

	v2_mean, v2_std, v2_cov, cp_mean, cp_std, cp_cov = load_data(pa,sep,noisetup,'amical',calibrated=calibrated)


	ax1.errorbar(b,v2_mean,yerr=v2_std,linestyle='',fmt='.',capsize=2,alpha=0.8,markersize=12,label='AMICal')
	ax2.errorbar(sumbl,cp_mean,yerr=cp_std,linestyle='',fmt='.',capsize=2,alpha=0.8,markersize=12);

	#sampip

	v2_mean, v2_std, v2_cov, cp_mean, cp_std, cp_cov = load_data(pa,sep,noisetup,'sampip',calibrated=calibrated)

	ax1.errorbar(sampip_b+0.075,v2_mean,yerr=v2_std,linestyle='',fmt='.',capsize=2,alpha=0.8,markersize=12,label='SAMPip')
	ax2.errorbar(sumbl+0.075,cp_mean,yerr=cp_std,linestyle='',fmt='.',capsize=2,alpha=0.8,markersize=12);

	#implaneia

	v2_mean, v2_std, v2_cov, cp_mean, cp_std, cp_cov = load_data(pa,sep,noisetup,'implaneia',calibrated=calibrated)

	ax1.errorbar(b+0.15,v2_mean,yerr=v2_std,linestyle='',fmt='.',capsize=2,alpha=0.8,markersize=12,label='ImPlaneIA')
	ax2.errorbar(sumbl+0.15,cp_mean,yerr=cp_std,linestyle='',fmt='.',capsize=2,alpha=0.8,markersize=12);

	# labels
	ax1.set_xlabel('Baseline (m)',fontsize=16)
	ax1.set_ylabel('Vis2',fontsize=16)

	ax2.set_xlabel('B1 + B2 (m)',fontsize=16)
	ax2.set_ylabel('Closure Phase (deg)',fontsize=16)

	ax1.axhline(1.0,linestyle='--')
	ax2.axhline(0.0,linestyle='--')

	# ax3.imshow()

	ax1.legend()

	# ax1.legend((ami,imp),('Amical','ImPlaneIA'))
	if calibrated:
		calname='calibrated'
		plt.suptitle('Calibrated Data')
	else:
		calname='uncalibrated'
		plt.suptitle('Raw Target Data')
	plt.savefig('../outputs/%s_%s_%s_all_pipelines_vs_baselines_%s.png' % (pa,sep,get_tag(noisetup),calname),bbox_inches='tight')

def covariance_plots(pa,sep,noisetup):
	fig, axes = plt.subplots(3,2,figsize=(16.0,24.0))

	# get filenames
	calfile_sampip = fname_generator('c',pa,sep,None,noisetup)    
	targfile_sampip = fname_generator('t',pa,sep,None,noisetup)

	calfile_amical = fname_generator('c',pa,sep,'amical',noisetup)    
	targfile_amical = fname_generator('t',pa,sep,'amical',noisetup)

	calfile_implaneia = fname_generator_implaneia('c',pa,sep,noisetup)    
	targfile_implaneia = fname_generator_implaneia('t',pa,sep,noisetup)    

	# load data
	cal_amical = np.load(calfile_amical)
	targ_amical = np.load(targfile_amical)

	cal_sampip = np.load(calfile_sampip)
	targ_sampip = np.load(targfile_sampip)

	cal_implaneia = oifits.load(calfile_implaneia)
	targ_implaneia = oifits.load(targfile_implaneia)


	#sampip

	v2_cal = cal_sampip['V2']
	cp_cal = cal_sampip['CP']

	v2_targ = targ_sampip['V2']
	cp_targ = targ_sampip['CP']

	v2_calmean, v2_calstd, v2_calcov = np.mean(v2_cal,axis=0), np.std(v2_cal,axis=0), np.cov(v2_cal.T)
	cp_calmean, cp_calstd, cp_calcov = np.mean(cp_cal,axis=0), np.std(cp_cal, axis=0), np.cov(cp_cal.T)

	v2_targmean, v2_targstd, v2_targcov = np.mean(v2_targ,axis=0), np.std(v2_targ,axis=0), np.cov(v2_targ.T)
	cp_targmean, cp_targstd, cp_targcov = np.mean(cp_targ,axis=0), np.std(cp_targ, axis=0), np.cov(cp_targ.T)

	vmin_v2, vmax_v2 = np.min(v2_targcov),np.max(v2_targcov)
	vmin_cp, vmax_cp = np.min(cp_targcov),np.max(cp_targcov)
	vmin_cp, vmax_cp = -np.max([np.abs(vmin_cp),np.abs(vmax_cp)]), np.max([np.abs(vmin_cp),np.abs(vmax_cp)])

	## imshow v2
	img_v2_targcov = axes[1,0].imshow(v2_targcov,vmin=vmin_v2,vmax=vmax_v2)
	ax = axes[1,0]
	cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
	plt.colorbar(img_v2_targcov,cax=cax)


	img_cp_targcov = axes[1,1].imshow(cp_targcov,vmin=vmin_cp,vmax=vmax_cp, cmap=mpl.cm.seismic_r)
	ax = axes[1,1]
	cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
	plt.colorbar(img_cp_targcov,cax=cax)


	axes[1,0].set_title('SAMPip V2 Covariance',fontsize=16)
	axes[1,1].set_title('SAMPip CP Covariance',fontsize=16)

	#amical

	v2_cal = cal_amical['V2']
	cp_cal = cal_amical['CP']

	v2_targ = targ_amical['V2']
	cp_targ = targ_amical['CP']

	v2_calmean, v2_calstd, v2_calcov = np.mean(v2_cal,axis=0), np.std(v2_cal,axis=0), np.cov(v2_cal.T)
	cp_calmean, cp_calstd, cp_calcov = np.mean(cp_cal,axis=0), np.std(cp_cal, axis=0), np.cov(cp_cal.T)

	v2_targmean, v2_targstd, v2_targcov = np.mean(v2_targ,axis=0), np.std(v2_targ,axis=0), np.cov(v2_targ.T)
	cp_targmean, cp_targstd, cp_targcov = np.mean(cp_targ,axis=0), np.std(cp_targ, axis=0), np.cov(cp_targ.T)


	## imshow v2
	img_v2_targcov = axes[0,0].imshow(v2_targcov,vmin=vmin_v2,vmax=vmax_v2)
	ax = axes[0,0]
	cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
	plt.colorbar(img_v2_targcov,cax=cax)

	axes[0,0].set_title('AMICal V2 Covariance',fontsize=16)
	img_cp_targcov = axes[0,1].imshow(cp_calcov,vmin=vmin_cp,vmax=vmax_cp, cmap=mpl.cm.seismic_r)
	ax = axes[0,1]
	cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
	plt.colorbar(img_cp_targcov,cax=cax)

	axes[0,1].set_title('AMICal CP Covariance',fontsize=16)


	#implaneia 
	v2_cal = cal_implaneia['OI_VIS2']['VIS2DATA'].T
	cp_cal = cal_implaneia['OI_T3']['T3PHI'].T

	v2_targ = targ_implaneia['OI_VIS2']['VIS2DATA'].T
	cp_targ = targ_implaneia['OI_T3']['T3PHI'].T

	v2_calmean, v2_calstd, v2_calcov = np.mean(v2_cal,axis=0), np.std(v2_cal,axis=0), np.cov(v2_cal.T)
	cp_calmean, cp_calstd, cp_calcov = np.mean(cp_cal,axis=0), np.std(cp_cal, axis=0), np.cov(cp_cal.T)

	v2_targmean, v2_targstd, v2_targcov = np.mean(v2_targ,axis=0), np.std(v2_targ,axis=0), np.cov(v2_targ.T)
	cp_targmean, cp_targstd, cp_targcov = np.mean(cp_targ,axis=0), np.std(cp_targ, axis=0), np.cov(cp_targ.T)

	## imshow v2
	img_v2_targcov = axes[2,0].imshow(v2_calcov,vmin=vmin_v2,vmax=vmax_v2)
	ax = axes[2,0]
	cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
	plt.colorbar(img_v2_targcov,cax=cax)

	axes[2,0].set_title('ImPlaneIA V2 Covariance',fontsize=16)
	img_cp_targcov = axes[2,1].imshow(cp_calcov,vmin=vmin_cp,vmax=vmax_cp, cmap=mpl.cm.seismic_r)
	ax = axes[2,1]
	cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
	plt.colorbar(img_cp_targcov,cax=cax)

	axes[2,1].set_title('ImPlaneIA CP Covariance',fontsize=16)

	# ax3.imshow()

	# ax1.legend()

	# ax1.legend((ami,imp),('Amical','ImPlaneIA'))
	plt.savefig('../outputs/%s_%s_%s_cov.png' % (pa,sep,get_tag(noisetup)),bbox_inches='tight')

# for j, noisetup in enumerate(tqdm(noise_comb)):

# 	baseline_plots(90,200,noisetup,calibrated=False)
# 	baseline_plots(90,200,noisetup,calibrated=True)
# 	covariance_plots(90,200,noisetup)