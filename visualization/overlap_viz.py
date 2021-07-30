import numpy as np
import pandas as pd
from nilearn import plotting, image

def read_groups(group_file, roi_file):
    """Read SVInet result and the correct ROI label associated with the index in the groups.txt file.
    Returns with pandas data frame.
    group_file: path to groups.txt (string)
    roi_file: path to a text file with the index ROI pairing
    """
    groups = np.loadtxt(group_file, delimiter='\t')
    ROIlabels=np.loadtxt(roi_file, delimiter='\t')
    
    #create a binarized version of the table
    group_bin = np.where(groups[:,2:]>0.0, 1, 0)
    
    # creating a list of column names
    column_names = []
    n_networks = groups[:,2:].shape[1]
    #create columns for the probability values
    for i in range(n_networks):
        name = 'N'+str(i)
        column_names.append(name)

    
    #create columns for the binarized values    
    for i in range(n_networks):
        name = 'bin_N'+str(i)
        column_names.append(name)

    # creating a list of inde names
    index_values = groups[:,1]

    # creating the dataframe
    df = pd.DataFrame(data = np.hstack([groups[:,2:], group_bin]), 
                      index = index_values, 
                      columns = column_names)
    df.sort_index(inplace=True)
    df.insert(loc = 0,
          column = 'Node',
          value = ROIlabels[:,1])
    
    #add a sum of each binarized value to the dataframe
    df['NumConnections'] = group_bin.sum(axis=1)
    
    return df

def setup_prob_networks(network_df, atlas, n_networks):
    """Create 3D matrices for each network based on ROIs from a given atlas,
    with the probability of each ROI belonging to that network.
    
    network_df: pandas dataframe, output of read_groups (the function expects consistent naming, e.g. column Node, N0..)
    atlas: parcellation in a 3D image format
    n_networks: the number of networks you want to convert into 3D arrays
    """

    networks = []
    
    for network in network_df.iloc[:,1:n_networks+1]:
        print('Creating network with probabiliy values: ',network)
        rois = network_df['Node'][network_df[network]>0.0].to_numpy()
        prob = network_df[network][network_df[network]>0.0].to_numpy()
        mat = np.zeros(atlas.dataobj.shape)
        for roi, roi_val in zip(rois, prob):
            mat += np.where(atlas.get_fdata().astype(int)==roi, roi_val, 0)
        networks.append(mat)
    
    return networks

def setup_bin_networks(network_df, atlas, n_networks):
    """Create 3D matrices for each network based on ROIs from a given atlas,
    with binary value of each ROI.
    
    network_df: pandas dataframe, output of read_groups (the function expects consistent naming, e.g. column Node, bin_N0..)
    atlas: parcellation in a 3D image format
    n_networks: the number of networks you want to convert into 3D arrays
    """
    networks = []
    
    for network in network_df.iloc[:,1:n_networks+1]:
        print('Creating binary network: ',network)
        rois = network_df['Node'][network_df[network]>0.0].to_numpy()
        mat = np.zeros(atlas.dataobj.shape)
        for roi in rois:
            mat += np.where(atlas.get_fdata().astype(int)==roi, 1, 0)
        networks.append(mat)
    
    return networks

def visualize_network_overlap(bin_network1, bin_network2, brain_temp, cmap='gnuplot', surf=False):
    """Visualize overlap between 2 networks with binary values.
    bin_network1: 3D array, network with binary values
    bin_network2: 3D array, network with binary values
    brain_temp: a template brain to create new images for visualization
    cmap: colormap
    surf: whether the netword should be displayed on a surface mesh
    """
    network1 = bin_network1.copy()
    network2 = bin_network2.copy()
    networks = [network1, network2]
    for i, network in enumerate(networks):
        network += np.where(network.astype(int)==1, i, 0)
    
    networks = np.array(networks)
    overlap_mat = networks.sum(axis=0)
    
    if overlap_mat.max()> 2:
        print('There is overlap between the networks.')
    else:
        print('No overlap.')

    overlap_brain = image.new_img_like(brain_temp, overlap_mat)
    
    if surf==True:
        view = plotting.view_img_on_surf(overlap_brain, symmetric_cmap=False, cmap=cmap, threshold=0.5)
    else:
        view = plotting.plot_roi(overlap_brain, cmap=cmap)
    
    return view

def setup_nconnections_map(network_df, atlas):
    """Create a 3D array where each ROI's value is how many networks it is connected to.
    network_df: pandas dataframe, output of read_groups()
    atlas: parcellation in a 3D image format
    """
    rois = network_df['Node'].to_numpy()
    n_connections = network_df['NConnections'].to_numpy()
    mat = np.zeros(atlas.dataobj.shape)
    for roi, roi_val in zip(rois, n_connections):
        mat += np.where(atlas.get_fdata().astype(int)==roi, roi_val, 0)
    return mat