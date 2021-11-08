import numpy as np
import pandas as pd
import scanpy as scp
import igraph

def make_reconstruction(adata,key_times,key_groups,
                        key_neighbors="neighbors",
                        flavor="votes",key_added="time_reconstruction",
                        copy=False):
#
#    make_reconstruction(adata,key_times,key_groups,
#                            key_neighbors="neighbors",
#                            flavor="votes",key_added="time_reconstruction",
#                            copy=False):
#
#       Function to make cluster the cells into clusters defined by time and cluster for the time reconstruction. 
#       
#       Parameters:
#       adata: h5ad format single cell dataset
#       key_times: key to the .obs column that contains the information of time
#       key_groups: key to the .obs column that contains the information of clusters in the data
#       key_neigbours: key that is look for the neighbours graph in .uns ("neighbours" by default)
#       flavor: style of the clustered matrix by time and clusters (votes is the number of connection between clusters, distances is average distance between clusters, paga is uses the paga algorithm to weight the connections) ("votes" by default)
#       key_added: key added to the .uns containing the reconstructed matrix ("time_reconstruction" by default)
#       copy: if to return the modified adata or to return a copy of it (False by default)
#
    
    #Extract names
    tuples = [(str(i),str(j)) for i,j in adata.obs.loc[:,[key_times,key_groups]].values]
    #Extract neighbor distances
    mm = adata.uns[key_neighbors]["distances"]
    #Make Dataframe
    data = pd.DataFrame(columns=pd.MultiIndex.from_tuples(pd.unique(tuples)),index=pd.MultiIndex.from_tuples(pd.unique(tuples)))
    #Add counts to the distances
    if flavor=="paga":
        aux = "__"
        while aux in adata.obs.columns.values:
            aux = aux + "_"
            
        adata.obs[aux] = [str(i)+aux+str(j) for i,j in adata.obs.loc[:,[key_times,key_groups]].values] 
        scp.tl.paga(adata,groups=aux)
        mm = adata.uns["paga"]["connectivities"]
        for pos_in,group_in in enumerate(adata.obs[aux].cat.categories):
            group_in = (group_in.split(aux)[0],group_in.split(aux)[1])
            for pos_out,group_out in enumerate(adata.obs[aux].cat.categories):
                group_out = (group_out.split(aux)[0],group_out.split(aux)[1])
                data.loc[group_in,group_out] = mm[pos_in,pos_out]
        adata.obs.drop(labels=aux,axis=1,inplace=True)
    else:
        for group_in in data.index:
            t = [i==group_in for i in tuples]
            for group_out in data.columns:
                t_out = [i==group_out for i in tuples]
                #votes
                if flavor=="votes":
                    data.loc[group_out,group_in] = (mm[:,t][t_out,:]>0).sum() #For some reason, the slicing work inversely as expected
                elif flavor=="distances":
                    d = mm[t,:][:,t_out].mean()
                    if d > 0:
                        data.loc[group_out,group_in] = -d #For some reason, the slicing work inversely as expected
                    else:
                        data.loc[group_out,group_in] = 0 #For some reason, the slicing work inversely as expected
                else:
                    raise ValueError("Flavor has to be one of the following: ['votes','distances','paga']")
                
    if flavor=="distances":
        data[data!=0] = (data[data!=0]-data.min())/-data.min()
    #Sort dataframe
    data.sort_index(axis=0,inplace=True)
    data.sort_index(axis=1,inplace=True)
    
    if copy:
        adataC = adata.copy()
        adataC.uns[key_added] = {"key_times":key_times,"key_groups":key_groups,"key_neighbors":key_neighbors,"flavor":flavor,"connections_matrix":data}
        return adataC
    else:
        adata.uns[key_added] = {"key_times":key_times,"key_groups":key_groups,"key_neighbors":key_neighbors,"flavor":flavor,"connections_matrix":data}
        return

def extract_matrix(adata,key_times,key_groups,
                   flavor="backward",
                   use_key="time_reconstruction",
                   time_order=None,
                   retain_n_max=None,retain_above_p=None,retain_time_sep=np.Inf,
                   normalize=None,
                   copy=False):
#
#    extract_matrix(adata,key_times,key_groups,
#                    flavor="backward",
#                    use_key="time_reconstruction",
#                    time_order=None,
#                    retain_n_max=None,retain_above_p=None,retain_time_sep=np.Inf,
#                    normalize=None,
#                    copy=False):
#
#       Function to make cluster the cells into clusters defined by time and cluster for the time reconstruction. 
#       
#       Parameters:
#       adata: h5ad format single cell dataset
#       key_times: key to the .obs column that contains the information of time
#       key_groups: key to the .obs column that contains the information of clusters in the data
#       flavor: direction of the causality matrix between "backward", "forward" and "same" ("backward" by default)
#       use_key: key hwere the reconstruction matrix is stored
#       time_order: order of the categories of time, if None, they are inferred (None by default)
#       retain_n_max: Retain only the n most weighted connection of each node. If None, keep all. (None by default)
#       retain_above_p: Retain only the weights that are above certain score. If none, keep all. (None by default)
#       retain_time_sep: Retain only the connections that are between timepoints separated less that the specified value (np.Inf by default),
#       normalize: If True, normalie the data such that the sum of the outgoing edges add to one. If None, ic normalizes the just if "votes" is the flavor (None by default)
#       copy: if to return the modified adata or to return a copy of it (False by default)
#

    if use_key not in adata.uns.keys():
        raise ValueError(use_key," not found in the .uns property.")
    else:
        data = adata.uns[use_key]["connections_matrix"]
    
    #Infer order if not defined
    if time_order == None:
        time_order = np.sort(np.unique([i[0] for i in data.index]))
        print("Causal order inferred: ",time_order)
    
    #Remove undesired counts for the flavour
    for group_in in data.index:
        for group_out in data.columns:
            incoming_pos = np.where(group_in[0] == time_order)[0][0]
            outgoing_pos = np.where(group_out[0] == time_order)[0][0]
            #Forward
            if flavor == "forward":
                if not (incoming_pos < outgoing_pos):
                    data.loc[group_in,group_out] = 0
            #Same
            if flavor == "same":
                if incoming_pos != outgoing_pos:
                    data.loc[group_in,group_out] = 0
            #Backward
            if flavor == "backward":
                if not (incoming_pos > outgoing_pos):
                    data.loc[group_in,group_out] = 0
                
            #Time separation
            if abs(incoming_pos-outgoing_pos) > retain_time_sep:
                data.loc[group_in,group_out] = 0
                                    
    #Normalize
    if normalize == None:
        if adata.uns[use_key]["flavor"] == "votes":
            normalize = True
        elif adata.uns[use_key]["flavor"] == "distances":
            normalize = False
        elif adata.uns[use_key]["flavor"] == "paga":
            normalize = False
            
    if normalize:
        #Forward
        if flavor == "forward" and normalize:
            data = data.transpose()
            total = data.sum(axis=0)
            total[total==0] += 1
            data /= total
            data = data.transpose()    
        #Backward
        elif flavor == "backward" and normalize:
            data = data.transpose()
            total = data.sum(axis=0)
            total[total==0] += 1
            data /= total
            data = data.transpose()
        #Same
        elif flavor == "backward":
            None
        else:
            raise ValueError("flavor_weights has to be one of the following: ['forward','same','backward']")
                
    if retain_n_max != None:
        for i in data.index:
            v = -1*np.sort(-1*data.loc[i,:])[retain_n_max-1]
            data.loc[i,data.loc[i,:]<v] = 0

    if retain_above_p != None:
        for i in data.index:
            data.loc[i,data.loc[i,:]<retain_above_p] = 0
            
    #Create graph
    g = igraph.Graph(directed = True)
    g.add_vertices(data.index.values)
    for i in g.vs:
        i["time"] = i["name"][0]
        i["cluster"] = i["name"][1]
    for i in g.vs:
        for j in g.vs:
            v = data.loc[i["name"],j["name"]]
            if v != 0:
                g.add_edge(i.index,j.index,weight=v)
            
    if copy:
        adataC = adata.copy()
        adataC.uns[use_key]["weighted_matrix"] = {"time_order":time_order,"flavor":flavor,"matrix":data,"graph":g}        
        return adataC
    else:
        adata.uns[use_key]["weighted_matrix"] = {"time_order":time_order,"flavor":flavor,"matrix":data,"graph":g}
        return