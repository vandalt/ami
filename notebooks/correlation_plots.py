import pygtc
import numpy as np
import matplotlib.pyplot as plt
import pdb
from readcol import * 

[files] = readcol('noises_c.txt', twod=False)
[files2] = readcol('noises_t.txt', twod=False)

for i in range(len(files)):

    filename = files[i]
    filename2 = files2[i]

    data = np.load(filename)
    data2 = np.load(filename2)
    
    names = ['BL1','BL2','BL3','BL4','BL5','BL6','BL7','BL8','BL9','BL10',\
                 'BL111','BL12','BL13','BL14','BL15','BL16','BL17','BL18',\
                 'BL19','BL20','BL21']

    chainLabels = ["Calibrator", "Target"]
    GTC = pygtc.plotGTC(chains=[data['V2'], data2['V2']], paramNames=names, chainLabels=chainLabels)
    GTC.suptitle('V2_'+filename[:-5])
    GTC.savefig('V2_'+filename[:-5]+'.pdf', bbox_inches='tight')

    names = ['CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CP7', 'CP8', 'CP9', 'CP10', \
             'CP11', 'CP12', 'CP13', 'CP14', 'CP15', 'CP16', 'CP17', 'CP18', 'CP19', 'CP20',\
             'CP21', 'CP22', 'CP23', 'CP24', 'CP25', 'CP26', 'CP27', 'CP28', 'CP29', 'CP30',\
             'CP31', 'CP32', 'CP33', 'CP34', 'CP35']

    chainLabels = ["Calibrator", "Target"]
    GTC = pygtc.plotGTC(chains=[data['CP'], data2['CP']], paramNames=names, chainLabels=chainLabels)
    GTC.suptitle('CP_'+filename[:-5])
    GTC.savefig('CP_'+filename[:-5]+'.pdf', bbox_inches='tight')

    plt.close('all')
