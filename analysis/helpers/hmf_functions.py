import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from skimage.feature import peak_local_max
from skimage.morphology import label, watershed

def corrected_size(sizeN):
    Rs = (3*sizeN/(4*np.pi))**(1.0/3.0)
    Rs1 = Rs + 2.0
    return (Rs/Rs1)**3 * sizeN

def plot_regions(distance):
    """
    Function for DEBUGGING and optimal parameter SEARCH wrt PEAK finding
    :param:     initial mask generation threshold (0.3 at the moment8
    :param:     sigma of gaussian filter <- determine by requiring that no. of identified peaks == number of gt halos
    :param:     min_distance of peak_local_max <- determine by " " "
    """

    # initial mask generation; has to occur before smoothing of distance
    image = (distance > 0.03) # seems good...

    # initial smoothing for eradicating edge effects
    distance = nd.gaussian_filter(distance, sigma=4.0, mode='wrap')

    # local peak finder
    local_maxi = peak_local_max(distance, indices=False, labels=image, exclude_border=False, min_distance=3)
    inds_maxi  = peak_local_max(distance, indices=True, labels=image, exclude_border=False, min_distance=3)

    # marker-based watershed region fidner
    markers = label(local_maxi)
    labels_ws = watershed(-distance, markers=markers, mask=image, watershed_line=True)

    # plotting
    plt.figure()
    plt.imshow(distance, cmap='cubehelix')
    for ids in inds_maxi:
        plt.scatter(ids[1], ids[0], c='red', s=1.0)
    plt.xlim((0, 512))
    plt.ylim((512, 0))

    copy = labels_ws.copy()
    for k in np.unique(labels_ws)[1:]:
        mask = labels_ws == k
        copy[mask] = np.random.uniform(10, 100, (1,))[0]
    plt.figure()
    plt.imshow(copy)
    for ids in inds_maxi:
        plt.scatter(ids[1], ids[0], c='red', s=1.0)
    plt.xlim((0, 512))
    plt.ylim((512, 0))
    plt.show()
    return inds_maxi

def label_regions(distance, sim, save_maxima=False, mode=None, peak_to_threshold_mapping=True):
    # initial cut
    distance[distance <= 0.04] = 0.0
    old_dist = distance.copy()

    # initial mask generation; has to occur before smoothing of distance
    image = (distance.copy() > 0.04)  # old: 0.005


    # Adaptive thresholding, smoothing and peak-retrieving
    levels = [0.2, 0.07]  # , 0.04]
    sigmas = [4, 1.5]  # , 1]

    new_dist = distance.copy()
    for k in range(len(levels)):
        d = new_dist.copy()  # new_dist.copy()
        dshape = distance.shape
        mask = distance > levels[k]
        d = nd.gaussian_filter(d, sigma=sigmas[k], mode='wrap')
        mean = np.mean(d)
        new_dist -= mean
        new_dist = new_dist.flatten()
        new_dist[mask.flatten()] = d.flatten()[mask.flatten()]
        new_dist = np.reshape(new_dist, dshape)
    distance = new_dist
    distance += abs(distance.min())


    # THE THRESHOLDS HAVE TO BE TAKEN FROM OLD Dist
    d_low = np.where(old_dist > 0.04, distance, 0.0)
    d_int = np.where(old_dist > 0.07, distance, 0.0)
    d_hig = np.where(old_dist > 0.2, distance, 0.0)
    sugg_low = peak_local_max(d_low, labels=d_low > 0.0, indices=False, exclude_border=False, min_distance=3)
    sugg_int = peak_local_max(d_int, labels=d_int > 0.0, indices=False, exclude_border=False, min_distance=5)
    sugg_hig = peak_local_max(d_hig, labels=d_hig > 0.0, indices=False, exclude_border=False, min_distance=8)

    sugg_low = np.where(np.logical_and(0.07 >= old_dist, old_dist > 0.04), sugg_low, 0.0)
    sugg_int = np.where(np.logical_and(0.2 >= old_dist, old_dist > 0.07), sugg_int, 0.0)
    sugg_hig = np.where(np.logical_and(np.inf >= old_dist, old_dist > 0.2), sugg_hig, 0.0)

    final_maxima = np.logical_or(sugg_low, sugg_int)
    final_maxima = np.logical_or(sugg_hig, final_maxima)
    print('no of local maxi:', final_maxima.sum())
    if save_maxima:
        np.save('saved_sizes' + sim + '/final_maxima.npy', final_maxima)
        np.save('saved_sizes' + sim + '/dist_p.npy', distance)
        print('Saved final_maxima and dist_p.')

    #x_low, y_low, z_low = np.nonzero(sugg_low)
    #x_int, y_int, z_int = np.nonzero(sugg_int)
    #x_hig, y_hig, z_hig = np.nonzero(sugg_hig)

    #inds_maxi_low = np.stack((x_low, y_low, z_low)).T
    #inds_maxi_int = np.stack((x_int, y_int, z_int)).T
    #inds_maxi_hig = np.stack((x_hig, y_hig, z_hig)).T
    #print len(inds_maxi_low)
    #print len(inds_maxi_int)
    #print len(inds_maxi_hig)

    # marker-based watershed region finder
    markers = label(final_maxima)
    labels_ws = watershed(-distance, markers=markers, mask=image, watershed_line=True)
    u, co = np.unique(labels_ws, return_counts=True)
    print('no of incorrected proto halos >=8*285 right from ws, no pbc yet:', len(co[co>=8*285]))


    # In order to identify the peak_val to contour threshold relation
    if mode=='peak_to_thresh_mode':
        isolated_regions = nd.find_objects(labels_ws)

        Fs = np.linspace(0.04, 0.4, 100)
        peak_vals = []
        masses = []
        f_threshs = []

        for k, slice2d in enumerate(tqdm(isolated_regions)):
            k = k + 1
            # isolate the only region of interest, protect rest
            cleaned_mask = (labels_ws[slice2d] == k)
            # good assertion check
            # assert np.unique(labels_ws[slice2d][cleaned_mask]) == k

            # the following line is correct and works if we only want the max,
            # it would not work if we wanted the argmax(!), since cleaned_mask
            # further reduces the dimensions of distance[slice2d]!
            peak_value = np.max(distance[slice2d][cleaned_mask])
            mh = []
            fh = []
            for f in Fs:
                adjusted_mask = (old_dist[slice2d]) >= f
                mh.append((np.logical_and(adjusted_mask, cleaned_mask)).sum())
                fh.append(f)
            peak_vals.append(peak_value * np.ones_like(Fs))
            masses.append(mh)
            f_threshs.append(fh)
        return np.asarray(masses), np.asarray(peak_vals), np.asarray(f_threshs)

    # Adaptive correction according to peak val and contour
    # MAYBE WE SHOULD DO THIS AFTER THE TRIPLE THING FOR PERIODIC!!!!
    if peak_to_threshold_mapping==True:
        collect_corrected = np.zeros_like(labels_ws)
        isolated_regions = nd.find_objects(labels_ws)

        def f(peak_val):
            if peak_val <= 0.07:
                return 0.04
            elif peak_val <= 0.15:
                return 0.05
            elif peak_value <= 0.2:
                return 0.055
            elif peak_val <= 0.3:
                return 0.08
            elif peak_val <= 0.4:
                return 0.07

            elif peak_val <= 0.5:
                return 0.1

            else:
                return 0.04
            """
            anchor1 = 0.08  # 0.1 old
            anchor2 = 0.32
            if peak_val < anchor1:
                return 0.040
            elif peak_val < anchor2:
                m = (0.075 - 0.040) / (anchor2 - anchor1)
                q = 0.040 - m * anchor1
                return m * peak_val + q
            elif peak_val >= anchor2:
                return 0.040"""

        for k, slice2d in enumerate(tqdm(isolated_regions)):
            k = k + 1
            # isolate the only region of interest, protect rest
            cleaned_mask = (labels_ws[slice2d] == k)
            # good assertion check
            # assert np.unique(labels_ws[slice2d][cleaned_mask]) == k
            peak_value = np.max(distance[slice2d][cleaned_mask])
            f_trhesh = f(peak_value)
            adjusted_mask = (old_dist[slice2d]) >= f_trhesh

            collect_corrected[slice2d] += k * np.logical_and(adjusted_mask, cleaned_mask)

        labels_ws = collect_corrected
        print(labels_ws.shape)
        return labels_ws

    else:
        return labels_ws

def label_regions_periodic(distance, sim, save_maxima=False):
    # maybe there is a better way of accounting for the pbc
    d1 = distance
    d2 = np.roll(distance, shift=64, axis=(0, 1, 2))
    d3 = np.roll(distance, shift=128, axis=(0, 1, 2))

    b1 = label_regions(d1, sim=sim, save_maxima=save_maxima)
    b2 = np.roll(label_regions(d2, sim=sim, save_maxima=save_maxima), shift=-64, axis=(0, 1, 2))
    b3 = np.roll(label_regions(d3, sim=sim, save_maxima=save_maxima), shift=-128, axis=(0, 1, 2))

    h, s = [], []
    for bi, start in tqdm(zip([b1, b2, b3], [1000, 10000, 20000])):
        bi = bi.flatten()
        unq, inv, counts = np.unique(bi, return_inverse=True, return_counts=True)

        # now construct the new array with the labels being the sizes directly through counts[inv]
        hi = counts[inv]

        # also construct a new array holding new unique labels for each region; later we simply do np.unique to get hmf
        si = np.arange(start, start + len(bi))[inv]    # 0 is reserved for BG

        # have to set the BG pixels to 0 again; use original bi
        hi[bi==0] = 0.0
        si[bi==0] = 0.0
        h.append(hi)
        s.append(si)

    where_max = np.argmax(np.stack(h, axis=-1), axis=-1)
    d = np.stack(s, axis=-1)
    F = []
    for di, m in tqdm(zip(d, where_max)):
        F.append(di[m])
    F = np.asarray(F)
    return F
