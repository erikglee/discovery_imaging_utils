import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

def imagesc_schaeffer_17(connectivity_matrix, parcel_labels, minmax, border_width=5, add_colorbar=True, dpi=200,
                         x_tick_labels=True, y_tick_labels=True, matplotlib_color_scheme='jet',
                         x_tick_font_size='xx-small',y_tick_font_size='xx-small', title='', linewidth = 1, linecolor='black'):

    """This function can make a connectomic plot for the 17 network schaeffer parcellation
    at any resolution. Needs to take a nxn numpy matrix, a length n list of parcel names (taken
    directly from the Schaeffer/Yeo parcellation), and a two element list specifying the
    minimum and maximum for the color scale (i.e. minmax = [0, 1]). You can choose how
    wide you want the border coloring to be, whether or not to use a colorbar, figure resolution,
    etc. (see kwargs above). tick font size and coloring schemes accept values that work with
    matplotlib.


    example usage:
    imagesc_schaeffer_17(nxn_conn_mat_as_np_array, len_n_list_of_label_names, [-1, 1])"""


    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    #The names of the different networks for the visualization
    network_names = ['Vis. A', 'Vis. B', 'SomMot. A', 'SomMot. B', 'Temp. Par.', 'Dors. Attn. A',
                    'Dors. Attn. B', 'Sal. A', 'Sal. B', 'Cont. A', 'Cont. B', 'Cont. C', 'DMN A',
                    'DMN B', 'DMN C', 'Limbic A', 'Limbic B']

    #The name of different networks to pull out of the labels
    network_identifiers = ['VisCent','VisPeri','SomMotA','SomMotB','TempPar', 'DorsAttnA',
                           'DorsAttnB','SalVentAttnA','SalVentAttnB','ContA','ContB','ContC','DefaultA',
                           'DefaultB','DefaultC','Limbic_TempPole', 'Limbic_OFC']

    alt_network_identifiers = ['VisCent','VisPeri','SomMotA','SomMotB','TempPar', 'DorsAttnA',
                       'DorsAttnB','SalVentAttnA','SalVentAttnB','ContA','ContB','ContC','DefaultA',
                       'DefaultB','DefaultC','LimbicA','LimbicB']

    network_colors = [
        [97/255, 38/255, 107/255, 1], #vis. a
        [195/255, 40/255, 39/255, 1], #vis. b
        [79/255, 130/255, 165/255, 1], #sommot a
        [82/255, 181/255, 140/255, 1], #sommat b
        [53/255, 75/255, 159/255, 1], #temp par
        [75/255, 147/255, 72/255, 1], #dors attn a
        [50/255, 116/255, 62/255, 1], #dors attn b
        [149/255, 77/255, 158/255, 1], #sal A
        [222/255, 130/255, 177/255, 1], #sal B
        [210/255, 135/255, 48/255, 1], #cont a
        [132/255, 48/255, 73/255, 1], #cont b
        [92/255, 107/255, 130/255, 1], #cont c
        [217/255, 221/255, 72/255, 1], #dmn a
        [176/255, 49/255, 69/255, 1], #dmn b
        [41/255, 37/255, 99/255, 1], #dmn c
        [75/255, 87/255, 61/255, 1], #limbic a
        [149/255, 166/255, 110/255, 1] #limbic b
    ]


    #Array to store network IDs (0-N, corresponding to order of network names)
    network_ids = np.zeros(len(parcel_labels))
    all_networks_present = np.zeros(len(network_identifiers))

    #Find which network each parcel belongs to
    for i in range(0,len(parcel_labels)):

        network_id_found = 0

        for j in range(0,len(network_identifiers)):

            if network_identifiers[j] in parcel_labels[i]:
                network_ids[i] = j
                network_id_found = 1
                all_networks_present[j] = 1

        if network_id_found == 0:

            for j in range(0,len(network_identifiers)):

                if alt_network_identifiers[j] in parcel_labels[i]:
                    network_ids[i] = j
                    network_id_found = 1
                    all_networks_present[j] = 1

        #Check that the parcel belongs to a network
        if network_id_found == 0:

            raise NameError('Error - network not found for parcel: ' + parcel_labels[i])

    #Check the each network has a parcel
    if np.sum(all_networks_present) != all_networks_present.shape[0]:

        inds = np.where(all_networks_present == 0)
        network_string = ''
        for temp_ind in inds[0]:
            print(temp_ind)
            network_string = network_string + network_identifiers[temp_ind] + ' (Primary ID)/' + alt_network_identifiers[temp_ind] + ' (Possible Secondary ID), '

        raise NameError('Error - no parcels found for networks: ' + network_string[:-2])



    #Create arrays for the sorted network ids and also store the inds to
    #obtain the sorted matrix
    sorted_ids = np.sort(network_ids, kind = 'mergesort')
    sorted_id_inds = np.argsort(network_ids, kind = 'mergesort')

    #Calculate where the center and edge of each network is for labeling
    #different networks on netmat figures
    network_edges = np.zeros(len(network_names))
    for i in range(len(network_names)):
        for j in range(len(sorted_ids)):

            if sorted_ids[j] == i:

                network_edges[i] = j

    network_centers = np.zeros(len(network_names))
    network_centers[0] = network_edges[0]/2.0 - 0.5
    for i in range(1,len(network_edges)):
        network_centers[i] = (network_edges[i] + network_edges[i-1])/2.0


    #Sort the connectivity matrix to be aligned with networks
    sorted_conn_matrix = np.zeros(connectivity_matrix.shape)
    sorted_conn_matrix = np.reshape(connectivity_matrix[sorted_id_inds,:], connectivity_matrix.shape)
    sorted_conn_matrix = np.reshape(sorted_conn_matrix[:,sorted_id_inds], connectivity_matrix.shape)

    #Apply colormap to data
    cmap = getattr(matplotlib.cm, matplotlib_color_scheme)
    norm = matplotlib.colors.Normalize(vmin=minmax[0], vmax=minmax[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    jet_conn_matrix =  m.to_rgba(sorted_conn_matrix, norm=True)

    #Add border to matrix
    jet_conn_with_borders = np.zeros((jet_conn_matrix.shape[0] + border_width, jet_conn_matrix.shape[1] + border_width, \
                                      jet_conn_matrix.shape[2]))
    jet_conn_with_borders[0:(-1*border_width),border_width:,:] = jet_conn_matrix
    for i in range(0,sorted_ids.shape[0]):
        jet_conn_with_borders[i,0:border_width,:] = network_colors[int(sorted_ids[i])]
        jet_conn_with_borders[(-1*border_width):,i+border_width,:] = network_colors[int(sorted_ids[i])]

    ######################################################################################
    ######################################################################################

    #Make and plot figure
    fig = plt.figure(dpi=dpi)
    plot_obj = plt.imshow(jet_conn_with_borders)
    if len(title) > 0:
        plt.title(title)

    lw = linewidth
    #Add lines to identify network borders
    for i in network_edges:
        plt.axvline(x=i + 0.5 + border_width,color=linecolor, lw=lw)
    plt.axvline(x=border_width - 0.5, color=linecolor, lw=lw)

    for i in network_edges:
        plt.axhline(y=i + 0.5,color=linecolor, lw=lw)

    ######################################################################################
    ######################################################################################

    #optionally add x tick labels
    if x_tick_labels:
        plt.xticks(ticks=network_centers + border_width + 0.5,labels=network_names, rotation=90, fontsize=x_tick_font_size)

    #optionally add y tick labels
    if y_tick_labels:
        plt.yticks(ticks=network_centers + 0.5,labels=network_names, fontsize=y_tick_font_size)

    #optionally add colorbar
    if add_colorbar:
        ax = plt.gca()
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        plt.colorbar(mappable = mappable, cax = cax)




    return #fig


def imagesc_schaeffer_7(connectivity_matrix, parcel_labels, minmax, border_width=5, add_colorbar=True, dpi=200,
                         x_tick_labels=True, y_tick_labels=True, matplotlib_color_scheme='jet',
                         x_tick_font_size='xx-small',y_tick_font_size='xx-small', title='', linewidth = 1, linecolor='black'):

    """This function can make a connectomic plot for the 17 network schaeffer parcellation
    at any resolution. Needs to take a nxn numpy matrix, a length n list of parcel names (taken
    directly from the Schaeffer/Yeo parcellation), and a two element list specifying the
    minimum and maximum for the color scale (i.e. minmax = [0, 1]). You can choose how
    wide you want the border coloring to be, whether or not to use a colorbar, figure resolution,
    etc. (see kwargs above). tick font size and coloring schemes accept values that work with
    matplotlib.


    example usage:
    imagesc_schaeffer_17(nxn_conn_mat_as_np_array, len_n_list_of_label_names, [-1, 1])"""


    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    network_names = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    network_identifiers = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    alt_network_identifiers = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    network_colors = [[121/255,3/255,136/255,1],[67/255,129/255,182/255,1],[0/255,150/255,0/255,1], \
                      [198/255,41/255,254/255,1],[219/255,249/255,160/255,1], \
                      [232/255,149/255,0/255,1], [207/255,60/255,74/255,1]]

    network_colors = [
        [97/255, 38/255, 107/255, 1], #vis. a
        [195/255, 40/255, 39/255, 1], #vis. b
        [79/255, 130/255, 165/255, 1], #sommot a
        [82/255, 181/255, 140/255, 1], #sommat b
        [53/255, 75/255, 159/255, 1], #temp par
        [75/255, 147/255, 72/255, 1], #dors attn a
        [50/255, 116/255, 62/255, 1], #dors attn b
        [149/255, 77/255, 158/255, 1], #sal A
        [222/255, 130/255, 177/255, 1], #sal B
        [210/255, 135/255, 48/255, 1], #cont a
        [132/255, 48/255, 73/255, 1], #cont b
        [92/255, 107/255, 130/255, 1], #cont c
        [217/255, 221/255, 72/255, 1], #dmn a
        [176/255, 49/255, 69/255, 1], #dmn b
        [41/255, 37/255, 99/255, 1], #dmn c
        [75/255, 87/255, 61/255, 1], #limbic a
        [149/255, 166/255, 110/255, 1] #limbic b
    ]


    #Array to store network IDs (0-N, corresponding to order of network names)
    network_ids = np.zeros(len(parcel_labels))
    all_networks_present = np.zeros(len(network_identifiers))

    #Find which network each parcel belongs to
    for i in range(0,len(parcel_labels)):

        network_id_found = 0

        for j in range(0,len(network_identifiers)):

            if network_identifiers[j] in parcel_labels[i]:
                network_ids[i] = j
                network_id_found = 1
                all_networks_present[j] = 1

        if network_id_found == 0:

            for j in range(0,len(network_identifiers)):

                if alt_network_identifiers[j] in parcel_labels[i]:
                    network_ids[i] = j
                    network_id_found = 1
                    all_networks_present[j] = 1

        #Check that the parcel belongs to a network
        if network_id_found == 0:

            raise NameError('Error - network not found for parcel: ' + parcel_labels[i])

    #Check the each network has a parcel
    if np.sum(all_networks_present) != all_networks_present.shape[0]:

        inds = np.where(all_networks_present == 0)
        network_string = ''
        for temp_ind in inds[0]:
            print(temp_ind)
            network_string = network_string + network_identifiers[temp_ind] + ' (Primary ID)/' + alt_network_identifiers[temp_ind] + ' (Possible Secondary ID), '

        raise NameError('Error - no parcels found for networks: ' + network_string[:-2])



    #Create arrays for the sorted network ids and also store the inds to
    #obtain the sorted matrix
    sorted_ids = np.sort(network_ids, kind = 'mergesort')
    sorted_id_inds = np.argsort(network_ids, kind = 'mergesort')

    #Calculate where the center and edge of each network is for labeling
    #different networks on netmat figures
    network_edges = np.zeros(len(network_names))
    for i in range(len(network_names)):
        for j in range(len(sorted_ids)):

            if sorted_ids[j] == i:

                network_edges[i] = j

    network_centers = np.zeros(len(network_names))
    network_centers[0] = network_edges[0]/2.0 - 0.5
    for i in range(1,len(network_edges)):
        network_centers[i] = (network_edges[i] + network_edges[i-1])/2.0


    #Sort the connectivity matrix to be aligned with networks
    sorted_conn_matrix = np.zeros(connectivity_matrix.shape)
    sorted_conn_matrix = np.reshape(connectivity_matrix[sorted_id_inds,:], connectivity_matrix.shape)
    sorted_conn_matrix = np.reshape(sorted_conn_matrix[:,sorted_id_inds], connectivity_matrix.shape)

    #Apply colormap to data
    cmap = getattr(matplotlib.cm, matplotlib_color_scheme)
    norm = matplotlib.colors.Normalize(vmin=minmax[0], vmax=minmax[1])
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    jet_conn_matrix =  m.to_rgba(sorted_conn_matrix, norm=True)

    #Add border to matrix
    jet_conn_with_borders = np.zeros((jet_conn_matrix.shape[0] + border_width, jet_conn_matrix.shape[1] + border_width, \
                                      jet_conn_matrix.shape[2]))
    jet_conn_with_borders[0:(-1*border_width),border_width:,:] = jet_conn_matrix
    for i in range(0,sorted_ids.shape[0]):
        jet_conn_with_borders[i,0:border_width,:] = network_colors[int(sorted_ids[i])]
        jet_conn_with_borders[(-1*border_width):,i+border_width,:] = network_colors[int(sorted_ids[i])]

    ######################################################################################
    ######################################################################################

    #Make and plot figure
    fig = plt.figure(dpi=dpi)
    plot_obj = plt.imshow(jet_conn_with_borders)
    if len(title) > 0:
        plt.title(title)

    lw = linewidth
    #Add lines to identify network borders
    for i in network_edges:
        plt.axvline(x=i + 0.5 + border_width,color=linecolor, lw=lw)
    plt.axvline(x=border_width - 0.5, color=linecolor, lw=lw)

    for i in network_edges:
        plt.axhline(y=i + 0.5,color=linecolor, lw=lw)

    ######################################################################################
    ######################################################################################

    #optionally add x tick labels
    if x_tick_labels:
        plt.xticks(ticks=network_centers + border_width + 0.5,labels=network_names, rotation=90, fontsize=x_tick_font_size)

    #optionally add y tick labels
    if y_tick_labels:
        plt.yticks(ticks=network_centers + 0.5,labels=network_names, fontsize=y_tick_font_size)

    #optionally add colorbar
    if add_colorbar:
        ax = plt.gca()
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        plt.colorbar(mappable = mappable, cax = cax)




    return #fig

def quantile_quantile_dist_plot(observed_pvals, null_dists = None, percentiles = [1, 5, 10], title = '', dpi = 150):


    #p-values below 50 so we can represent them symmetrically
    if np.max(percentiles) > 50:
        raise NameError('Error - all percentiles should be less than 50')

    percentiles = np.sort(percentiles)
    legend_str = []

    line_types = ['dashed','dashdot','dotted','solid']

    sorted_negtrans_pvals = np.sort(-1*np.log(observed_pvals))

    if type(null_dists) == type(None):
        raise NameError('Error: not implemented yet, need to provide null dists')
    else:


        negtrans_nulls = np.sort(-1*np.log(null_dists), axis = 1)
        median_null = np.median(negtrans_nulls, axis = 0)

        plt.figure(dpi = dpi)
        for i, temp_perc_upper in enumerate(percentiles):

            upper_dist = np.percentile(negtrans_nulls, temp_perc_upper, axis=0)
            lower_dist = np.percentile(negtrans_nulls, 100 - temp_perc_upper, axis=0)
            plt.fill_between(median_null, upper_dist, lower_dist, alpha = 0.3, color = 'grey')

            temp_line_type = line_types[np.mod(i, len(line_types))]
            plt.plot(median_null, upper_dist, linestyle = temp_line_type, color = 'black', linewidth = 1)
            plt.plot(median_null, lower_dist, linestyle = temp_line_type, color = 'black', linewidth = 1, label = '_nolegend_')
            legend_str.append('> ' + str(temp_perc_upper) + ' perc.')

        plt.plot(median_null, median_null, color = 'black', linewidth = 1)
        legend_str.append('Null Median')
        plt.plot(median_null, sorted_negtrans_pvals, color = 'blue')
        legend_str.append('Observed P-vals')
        plt.legend(legend_str)
        plt.xlabel('Expected p-values (-log10(p))')
        plt.ylabel('Observed p-values (-log10(p))')
        plt.title(title)
