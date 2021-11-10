import numpy as np
import pandas as pd
import scanpy as scp
import igraph
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

def plot_causality(adata,
                   key_hue=None,hue_function="normalize",title_hue="Color",
                   key_size=None,size_function="normalize",title_size="Size",
                   origin_node=None,target_node=None,n_paths_ploted=np.Inf,
                   line_width=4,marker_size=4000,marker_sizes=(0,4000),
                   key_neighbors="neighbors",
                   use_key="time_reconstruction",
                   time_order=None,ax=None):
#
#    def plot_causality(adata,
#                    key_hue=None,hue_function="normalize",title_hue="Color",
#                    key_size=None,size_function="normalize",title_size="Size",
#                    origin_node=None,target_node=None,n_paths_ploted=np.Inf,
#                    line_width=4,marker_size=4000,marker_sizes=(0,4000),
#                    key_neighbors="neighbors",
#                    use_key="time_reconstruction",
#                    time_order=None,ax=None):
#
#       Plot the causality reconstruction of the different clusters in time.
#
#       adata: h5ad format single cell dataset
#       key_hue: key to use for coloring the nodes from a parametere in .obs or .var.index (None by default)
#       hue_function: method to sintetize the information in the key_hue "normalize" or "mean" ("normalize" by default)
#       title_hue: name added to the legend ("Color" by default)
#       key_size: key to use for sizing the nodes from a parametere in .obs or .var.index (None by default)
#       size_function: method to sintetize the information in the key_size "normalize" or "mean" ("normalize" by default)
#       title_size: name added to the legend ("Size" by default),
#       origin_node: tuple specifying the origine of a reconstructed trajectory as (time,cluster_name) (None by default)
#       target_node: tuple specifying the origine of a reconstructed trajectory as (time,cluster_name). If not specified bu origin_node is specified, it reconstructs all trajectories from origin (None by default)
#       n_paths_ploted: top number of reconstructed paths to show over the plot
#       line_width: width factor of the causality lines ploted (4 by default)
#       marker_size: size factor of the nodes ploted (4000 by default)
#       marker_sizes: sizes range of the nodes size ((0,4000) by default),
#       key_neighbors: key of the neigbours matrix used for the reconstruction ("neighbors" by default),
#       use_key: key of the time reconstruction object in .uns ("time_reconstruction" by default)
#       ax: axis object where to plot the reconstruction, if None, a new plot is created and fig,axis are returned.
#     
    if use_key not in adata.uns.keys():
        raise ValueError(use_key," not found in .uns. Please run make_reconstruction and extract_matrix first.")
    else:
        matrix = adata.uns[use_key]["weighted_matrix"]["matrix"]
    
    key_times = adata.uns[use_key]["key_times"]
    key_groups = adata.uns[use_key]["key_groups"]
    time_order = adata.uns[use_key]["weighted_matrix"]["time_order"]
    time_map = {j:i for i,j in enumerate(np.unique([i[0] for i in matrix.index]))}
    cluster_map = {j:i for i,j in enumerate(np.sort(np.unique([i[1] for i in matrix.index])))}
    
    #Make plot if does not given the axis
    if ax == None:
        fig,axis = plt.subplots(figsize=[20,15])
    else:
        axis = ax
    
    #Print lines
    for i,origin in enumerate(matrix.index):
        for j,target in enumerate(matrix.columns):
            x1 = time_map[origin[0]]
            y1 = cluster_map[origin[1]]
            x2 = time_map[target[0]]
            y2 = cluster_map[target[1]]

            axis.plot([x1,x2],[y1,y2],linewidth=line_width*matrix.loc[origin,target],color="black",zorder=-2)
            
    #Print trajectory
    if origin_node != None:
        path_values = []
        g = adata.uns[use_key]["weighted_matrix"]["graph"]
        if target_node == None:
            source = g.vs[np.where(np.all(np.array(g.vs["name"]) == origin_node,axis=1))[0][0]].index
            path = g.get_all_simple_paths(source)
            m_length = np.max([len(i) for i in path])
            path = [i for i in path if len(i)==m_length]
            vv = []
            for i in path:
                v = 1
                for j in range(len(i)-1):
                    origin = g.vs[i[j]]["name"]
                    target = g.vs[i[j+1]]["name"]
                    e=np.where(np.all(np.array(g.get_edgelist())==(g.vs[i[j]].index,g.vs[i[j+1]].index),axis=1))[0][0]
                    v *= g.es[e]["weight"]
                vv.append(v)
                    
            v_total = np.sum(vv)
            if n_paths_ploted >= len(vv):
                order = np.argsort(vv)
            else:
                order = np.argsort(vv)[-n_paths_ploted:]
            path = [path[i] for i in order]
            for i in path:
                lx = [0]
                ly = [0]
                for j in range(len(i)-1):
                    lx = lx[:-1]
                    ly = ly[:-1]
                    origin = g.vs[i[j]]["name"]
                    target = g.vs[i[j+1]]["name"]
                    x1 = time_map[origin[0]]
                    y1 = cluster_map[origin[1]]
                    x2 = time_map[target[0]]
                    y2 = cluster_map[target[1]]
                    lx.append(x1)
                    lx.append(x2)
                    ly.append(y1)
                    ly.append(y2)

                path_values.append(v)
                axis.plot(lx,ly,linewidth=line_width*matrix.loc[origin,target],zorder=-2)                   
        elif target != None:
            source = g.vs[np.where(np.all(np.array(g.vs["name"]) == origin_node,axis=1))[0][0]].index
            target = g.vs[np.where(np.all(np.array(g.vs["name"]) == target_node,axis=1))[0][0]].index

            path = g.get_all_simple_paths(source,target)
            m_length = np.max([len(i) for i in path])
            path = [i for i in path if len(i)==m_length]
            vv = []
            for i in path:
                v = 1
                for j in range(len(i)-1):
                    origin = g.vs[i[j]]["name"]
                    target = g.vs[i[j+1]]["name"]
                    e=np.where(np.all(np.array(g.get_edgelist())==(g.vs[i[j]].index,g.vs[i[j+1]].index),axis=1))[0][0]
                    v *= g.es[e]["weight"]
                vv.append(v)
                    
            v_total = np.sum(vv)
            if n_paths_ploted >= len(vv):
                order = np.argsort(vv)
            else:
                order = np.argsort(vv)[-n_paths_ploted:]
            
            path = [path[i] for i in order]
            for i in path:
                lx = [0]
                ly = [0]
                for j in range(len(i)-1):
                    lx = lx[:-1]
                    ly = ly[:-1]
                    origin = g.vs[i[j]]["name"]
                    target = g.vs[i[j+1]]["name"]
                    x1 = time_map[origin[0]]
                    y1 = cluster_map[origin[1]]
                    x2 = time_map[target[0]]
                    y2 = cluster_map[target[1]]
                    lx.append(x1)
                    lx.append(x2)
                    ly.append(y1)
                    ly.append(y2)

                path_values.append(v)
                axis.plot(lx,ly,linewidth=line_width*matrix.loc[origin,target],zorder=-2) 

    #Checking non-zero elements
    pos = adata.obs.loc[:,[key_times,key_groups]]
    pos.loc[:,key_size] = 1             
    pos = pos.groupby([key_times,key_groups])
    pos = pos.count().loc[:,key_size]
    pos = pos.reset_index()
    exist = pos.iloc[:,2].values!=0

    #Size
    if key_size in adata.obs.columns:
        pos = adata.obs.loc[:,[key_times,key_groups,key_size]]
    elif key_size in adata.var.index:
        pos = adata.obs.loc[:,[key_times,key_groups]]
        pos[key_size] = np.array(adata[:,key_size].X.todense())
    else:
        pos = adata.obs.loc[:,[key_times,key_groups]]
        pos.loc[:,key_size] = 1
        pos2 = pos.groupby([key_times,key_groups]).count()
        size = np.zeros(pos2.shape[0])*marker_size 
    
    pos = pos.groupby([key_times,key_groups])
    if key_size != None:
        if size_function == "normalize":
            pos = pos.sum().loc[:,key_size].unstack()
            pos = (pos.transpose()/pos.transpose().sum(axis=0)).transpose()
            pos[np.isnan(pos.values)]=0
            pos = pos.stack().reset_index()
            #pos = pos.loc[pos.loc[:,0]!=0,:]
        elif size_function == "mean":
            pos = pos.mean().loc[:,key_size]
            pos = pos.reset_index()
            pos.loc[np.isnan(pos.loc[:,key_size]),key_hue]=0
            #pos = pos.loc[pos.loc[:,0]!=0,:]
        else:
            raise ValueError("size_function can only be one of the following: [normalize,mean]")
            
        size = pos.iloc[:,2].values*marker_size
    else:
        pos = pos.count().loc[:,key_size]
        pos = pos.reset_index()
        
    #Color
    if key_hue in adata.obs.columns:
        pos = adata.obs.loc[:,[key_times,key_groups,key_hue]]
    elif key_hue in adata.var.index.values:
        pos = adata.obs.loc[:,[key_times,key_groups]]
        pos[key_hue] = np.array(adata[:,[key_hue]].X.todense())
    else:
        pos = adata.obs.loc[:,[key_times,key_groups]]
        pos.loc[:,key_hue] = 1
        pos2 = pos.groupby([key_times,key_groups]).count()
        hue = np.zeros(pos2.shape[0])*marker_size 
           
    pos = pos.groupby([key_times,key_groups])
    if key_hue != None:
        if hue_function == "normalize":
            pos = pos.sum().loc[:,key_hue].unstack()
            pos = (pos.transpose()/pos.transpose().sum(axis=1)).transpose()
            pos[np.isnan(pos.values)]=0
            pos = pos.stack().reset_index()
            pos.reset_index()
            #pos = pos[pos!=0,:]
        elif hue_function == "mean":
            pos = pos.mean().loc[:,key_hue]
            pos = pos.reset_index()
            pos.loc[np.isnan(pos.loc[:,key_hue].values),key_hue]=0
            #pos = pos.loc[pos.loc[:,0]!=0,:]
        else:
            raise ValueError("size_function can only be one of the following: [normalize,mean]")
    
        hue = pos.iloc[:,2].values
    else:
        pos = pos.count().loc[:,key_hue]
        pos = pos.reset_index()
    
    x = np.array([time_map[i] for i in pos.loc[:,key_times]])
    y = np.array([cluster_map[i] for i in pos.loc[:,key_groups]])
    sns.scatterplot(x=x[exist],y=y[exist],hue=hue[exist],size=size[exist],sizes=marker_sizes,alpha=1,ax=axis)
    
    #Plotting labels, titles, etc...
    axis.tick_params(labelsize=20)
    axis.set_xticks([i for i in time_map.values()])
    axis.set_xticklabels([i for i in time_map.keys()])
    axis.set_yticks([i for i in cluster_map.values()])
    axis.set_yticklabels([i for i in cluster_map.keys()])
    axis.set_xlabel(key_times,fontsize=20)
    axis.set_ylabel(key_groups,fontsize=20)
    axis.set_title(adata.uns[use_key]["flavor"]+" "+adata.uns[use_key]["weighted_matrix"]["flavor"],fontsize=20)
        
    s = []
    s.append(mpatches.Patch(color=[1,1,1],label=title_hue))
    s.append(plt.scatter([0],[0],s=100,label=np.round(np.min(hue),2),color="#EDD1CB"))
    s.append(plt.scatter([0],[0],s=100,label=np.round(np.max(hue),2),color="#2D1E3E"))
    s.append(mpatches.Patch(color=[1,1,1],label=title_size))
    s.append(plt.scatter([0],[0],s=100,label=np.round(np.min(size)/marker_size,2),color="#EDD1CB"))
    s.append(plt.scatter([0],[0],s=1000,label=np.round(np.max(size)/marker_size,2),color="#EDD1CB"))
    s.append(mpatches.Patch(color=[1,1,1],label="Path score"))
    if origin_node != None:
        if n_paths_ploted >= len(vv):
            path_values = [i/v_total for i in np.sort(vv)]
        else:
            path_values = [i/v_total for i in np.sort(vv)[-n_paths_ploted:]]   
        for j,i in enumerate(path_values):
            color = "C"+str(np.mod(j,10))
            s.append(mlines.Line2D([],[],color=color,label=i))
            
    thirdLegend = plt.legend(bbox_to_anchor=(1,0.5), handles=s, loc="upper left",fontsize=20,title="",title_fontsize=20)
    plt.gca().add_artist(thirdLegend)  
    
    if ax == None:
        return axis
    else:
        return

def plot_reconstruction_over_map(adata,
                                 use_key="time_reconstruction",key_obsm="X_umap",
                                 line_width=1,
                                 key_annotations=None,
                                 ax=None):
#
#    def plot_reconstruction_over_map(adata,
#                                    use_key="time_reconstruction",key_obsm="X_umap",
#                                    line_width=1,
#                                    key_annotations=None,
#                                    ax=None):
#
#       Plot the time reconstruction over the dimensionality reduction map.
#
#       adata: h5ad format single cell dataset
#       use_key: key of the time reconstructed object in .uns ("time_reconstruction" by default)
#       key_obsm: key in .obsm of the representation over which to plot the data ("X_umap" by default)
#       line_width: width factor of the causality lines (1 by default),
#       key_annotations: key in .obs that annotated the clusters (None by default)
#       ax: axis object where to plot the reconstruction, if None, a new plot is created and fig,axis are returned.
#

    if use_key not in adata.uns.keys():
        raise ValueError(use_key," not found in .uns. Please run make_reconstruction and extract_matrix first.")
    else:
        matrix = adata.uns[use_key]["weighted_matrix"]["matrix"]

    if key_obsm not in adata.obsm.keys():
        raise ValueError(key_obsm," not found in .obsm. Please use a valid representation.")
    
    key_times = adata.uns[use_key]["key_times"]
    key_groups = adata.uns[use_key]["key_groups"]
    time_order = adata.uns[use_key]["weighted_matrix"]["time_order"]
    time_map = {j:i for i,j in enumerate(np.unique([i[0] for i in matrix.index]))}
    cluster_map = {j:i for i,j in enumerate(np.sort(np.unique([i[1] for i in matrix.index])))}
    
    xspace = (adata.obsm[key_obsm][:,0].max()-adata.obsm[key_obsm][:,0].min())
    yspace = (adata.obsm[key_obsm][:,1].max()-adata.obsm[key_obsm][:,1].min())
    for i,t in enumerate(time_order[1:]):
        t_old = time_order[i]
        plt.gca().set_prop_cycle(None)
        sns.scatterplot(x=adata[adata.obs[key_times]==t,:].obsm[key_obsm][:,0]+1.1*xspace*i,
                        y=adata[adata.obs[key_times]==t,:].obsm[key_obsm][:,1],
                        hue=adata[adata.obs[key_times]==t,:].obs[key_groups],
                        ax=ax)
        plt.gca().set_prop_cycle(None)
        sns.scatterplot(x=adata[adata.obs[key_times]==t_old,:].obsm[key_obsm][:,0]+1.1*xspace*i,
                        y=adata[adata.obs[key_times]==t_old,:].obsm[key_obsm][:,1],
                        hue=adata[adata.obs[key_times]==t_old,:].obs[key_groups],
                        alpha=0.5,
                        zorder=0,
                        ax=ax)            
        
        
    #Plot horizontal lines separating the time maps
    for i,t in enumerate(time_order[1:]):
        plt.vlines(1.1*xspace*max(0,i-1)+1.05*xspace+adata.obsm[key_obsm][:,0].min(),
                  ax.get_ylim()[0],
                  ax.get_ylim()[1],
                  color="black")
    
    #Plot labels
    for cluster in adata.obs[key_groups].unique():
        for i,t in enumerate(time_order[1:]):
            
            x = adata[(adata.obs[key_times]==t)*(adata.obs[key_groups]==cluster),:].obsm[key_obsm][:,0].mean()+1.1*xspace*i
            y = adata[(adata.obs[key_times]==t)*(adata.obs[key_groups]==cluster),:].obsm[key_obsm][:,1].mean()
            if not np.isnan(x):
                ax.text(x,y,cluster,backgroundcolor="lightgrey",zorder=1)

            t_old = time_order[i]
            x = adata[(adata.obs[key_times]==t_old)*(adata.obs[key_groups]==cluster),:].obsm[key_obsm][:,0].mean()+1.1*xspace*i
            y = adata[(adata.obs[key_times]==t_old)*(adata.obs[key_groups]==cluster),:].obsm[key_obsm][:,1].mean()
            if not np.isnan(x):
                ax.text(x,y,str(cluster)+"*",backgroundcolor="lightgrey",zorder=1)

                
    #Plot causality lines
    for i,origin in enumerate(matrix.index.values):
        t_origin,c_origin=origin
        for j,target in enumerate(matrix.columns.values):
            t_target,c_target=target
            x1,y1=adata[(adata.obs[key_times]==t_origin)*(adata.obs[key_groups]==c_origin),:].obsm[key_obsm].mean(axis=0)[:2]
            x2,y2=adata[(adata.obs[key_times]==t_target)*(adata.obs[key_groups]==c_target),:].obsm[key_obsm].mean(axis=0)[:2]
            add = time_map[t_origin]+(time_map[t_target]-time_map[t_origin])
            x1 += 1.1*xspace*add
            x2 += 1.1*xspace*add

            if matrix.loc[origin,target]>0:
                ax.plot([x1,x2],[y1,y2],linewidth=line_width*matrix.loc[origin,target],color="black",zorder=0)
                
    #Plot legend
    plt.gca().set_prop_cycle(None)
    l=[]
    for i in adata.obs.loc[:,key_groups].cat.categories:
        aux = plt.scatter([0],[0],label=i)
        l.append(aux)
                 
    legend = plt.legend(bbox_to_anchor=(1,0.5), handles=l, loc="upper left",fontsize=20,title="",title_fontsize=20)
    plt.gca().add_artist(legend)  

    return