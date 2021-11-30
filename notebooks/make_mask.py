import numpy as np

def maketriples_all(mask,verbose=False):
    """ returns int array of triple hole indices (0-based), 
        and float array of two uv vectors in all triangles
    """
    nholes = mask.shape[0]
    tlist = []
    for i in range(nholes):
        for j in range(nholes):
            for k in range(nholes):
                if i < j and j < k:
                    tlist.append((i, j, k))
    tarray = np.array(tlist).astype(np.int)
    if verbose:
        print("tarray", tarray.shape, "\n", tarray)

    tname = []
    uvlist = []
    # foreach row of 3 elts...
    for triple in tarray:
        tname.append("{0:d}_{1:d}_{2:d}".format(
            triple[0], triple[1], triple[2]))
        if verbose:
            print('triple:', triple, tname[-1])
        uvlist.append((mask[triple[0]] - mask[triple[1]],
                       mask[triple[1]] - mask[triple[2]]))
    # print(len(uvlist), "uvlist", uvlist)
    if verbose:
        print(tarray.shape, np.array(uvlist).shape)
    return tarray, np.array(uvlist)

def makebaselines(mask):
    """
    ctrs_eqt (nh,2) in m
    returns np arrays of eg 21 baselinenames ('0_1',...), eg (21,2) baselinevectors (2-floats)
    in the same numbering as implaneia
    """
    nholes = mask.shape[0]
    blist = []
    for i in range(nholes):
        for j in range(nholes):
            if i < j:
                blist.append((i, j))
    barray = np.array(blist).astype(np.int)
    # blname = []
    bllist = []
    for basepair in blist:
        # blname.append("{0:d}_{1:d}".format(basepair[0],basepair[1]))
        baseline = mask[basepair[0]] - mask[basepair[1]]
        bllist.append(baseline)
    return barray, np.array(bllist)
