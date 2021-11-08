import numpy as np
import pandas as pd
import scanpy as scp
import igraph

def local_cutoff(adata,tolerance_factor=1,key_neighbours="neighbors",key_times=None,copy=False):
#
# def local_cutoff(adata,tolerance_factor=1,key_neighbours="neighbors",key_times=None,copy=False):
#
#       Function that removes all the connections for each cell whose connection is over mean(knn distances)*tolerance_factor. 
#       This helps to remove possible spurious connections with distant clusters.
#       
#       Parameters:
#       adata: h5ad format single cell dataset
#       tolerance_factor: factor to scale the cutoff distances above the mean (1 by default)
#       key_neigbours: key that is look for the neighbours graph in .uns ("neighbours" by default)
#       key_times: categorical variable to use for subclustering the distances before prunning for each cell (None y default)
#       copy: if to return the modified adata or to return a copy of it (False by default)
#
    
    #Make copy of the neighbours matrix
    if copy:
        adataC = adata.copy()
        mm = adataC.uns[key_neighbours]["distances"]
        cc = adataC.uns[key_neighbours]["connectivities"]
    else:
        mm = adata.uns[key_neighbours]["distances"]
        cc = adata.uns[key_neighbours]["connectivities"]

    ##Classical neighbours
    #Go over all the neighbours
    if key_times == None:
        xx,yy = mm[:,:].nonzero() #Nonzero-positions
        for i in range(mm.shape[0]):
            #Extract the position of the non-zero neighbours
            y = yy[xx==i]
            x = xx[xx==i]

            #Compute the mean
            mean = mm[x,y].mean()
            #Check the p value of each neighbour according to a gamma distribution and set to zero for removal is is above the p-value
            for p in zip(x,y):
                v = mm[p]
                if v > mean*tolerance_factor:
                    mm[p] = 0
    else:
        if key_times not in adata.obs.columns.values:
            raise ValueError("key_times = "+str(key_times)+" is not a key that exists in adata.obs.")
        else:
            xx,yy = mm[:,:].nonzero() #Nonzero-positions
            kk = np.array(adata.obs.loc[:,key_times].values[yy])
            for i in range(mm.shape[0]):
                for j in np.unique(kk):
                    #Extract the position of the non-zero neighbours of a range
                    y = yy[xx==i]
                    x = xx[xx==i]
                    k = kk[xx==i]
                    #Extract those that are from the time bach to make the statistics
                    y = y[k==j]
                    x = x[k==j]

                    #Compute the mean
                    mean = mm[x,y].mean()
                    #Check the p value of each neighbour according to a gamma distribution and set to zero for removal is is above the p-value
                    for p in zip(x,y):
                        v = mm[p]
                        if v > mean*tolerance_factor:
                            mm[p] = 0           
    #Eliminate neighbours set to zero
    mm.eliminate_zeros()
    
    cc = cc[mm>0]
    
    #Return copy if necessary
    if copy:
        return adataC
    else:
        return