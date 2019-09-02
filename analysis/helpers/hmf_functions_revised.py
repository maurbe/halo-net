import numpy as np
import scipy.ndimage as nd
from skimage.morphology import label, watershed, extrema

"""
def f(p):
    #m = -0.23 / 80000
    #q = 0.23
    #return 0.1#m * peak_val + q
    anchor1 = 2e4
    anchor2 = 6e4
    if p < anchor1:
        return 0.18
    elif p <= anchor2:
        m = -(0.2 - 0.05) / (anchor2 - anchor1)
        q = 0.2 - m * anchor1
        return m * p + q
    else:
        return 0.03
"""


def f(p):
    if p <= 1e3:
        return 0.05
    elif p <= 1e4:
        return 0.18
    elif p <= 2e4:
        return 0.15
    elif p <= 6e4:
        return 0.12 # 0.1 makes the validation one almost perfect...

    elif p <= 1e5:
        return 0.07
    elif p <= 2e5:
        return 0.06

    else:
        return 0.03

def mainfunction(distance, pw=64):

    # step 1: expand the domain periodically
    distance    = np.pad(distance, pad_width=pw, mode='wrap')
    image       = distance > 0


    # step 2: identify the maxima and the maxima inside original domain
    distance    = nd.gaussian_filter(distance, sigma=1, mode='wrap')
    h_maxima    = extrema.h_maxima(distance, h=0.05)

    domain_in = np.s_[pw:-pw, pw:-pw, pw:-pw]
    h_maxima_in = np.zeros_like(h_maxima)
    h_maxima_in[domain_in] = h_maxima[domain_in]

    inds_max    = np.transpose(np.nonzero(h_maxima))
    inds_max_in = np.transpose(np.nonzero(h_maxima_in))


    # step 3: watershed segmentation; unfortunately have to do it for all maxima (not just 'in') and then filter
    labels_ws   = watershed(-distance, markers=label(h_maxima), mask=image, watershed_line=False)
    labels_wsF  = labels_ws.copy()
    un, inv, counts = np.unique(labels_ws, return_inverse=True, return_counts=True)
    counts[0] = 0
    mass_labels = counts[inv].reshape(distance.shape)  # "trick" to get mass label of every pixel


    # step 4: filter out (set to 0) those regions, whose maximum lies outside the domain
    regions = nd.find_objects(labels_ws)
    for i, region in enumerate(regions):
        id = i+1

        subvol_lab  = labels_ws[region]
        subvol_max  = h_maxima_in[region]
        mask        = subvol_lab==id

        # now check if there is an inside_max present inside, if not -> set region to 0
        if (subvol_max[mask]).sum() == 0:
            distance[region][mask]      = 0
            mass_labels[region][mask]   = 0
            labels_wsF[region][mask]    = 0
    # this is necessary, since (intermediate) labels are now missing in label_wsN (were filtered) -> relabel
    labels_wsF = label(labels_wsF)
    return labels_wsF, distance


def retrieve_corrected_regions(distance, sim, preload=True):

    # step 1: call the mainfucntion to return the filtered labels
    if preload:
        labels_wsF = np.load('hmf' + sim + '/src/uncorrected_labels' + sim + '.npy')
        distance = np.pad(distance, pad_width=64, mode='wrap')
        distance = nd.gaussian_filter(distance, sigma=2, mode='wrap')
    else:
        labels_wsF, distance = mainfunction(distance)
        np.save('hmf' + sim + '/src/uncorrected_labels' + sim + '.npy', labels_wsF )


    # step 2: peak-to-threshold correction
    corrected_regions   = np.zeros_like(labels_wsF)
    filtered_regions    = nd.find_objects(labels_wsF)
    peak_vals           = []
    corrected_sizes     = []
    for k, region in enumerate(filtered_regions):
        id = k+1

        mask            = labels_wsF[region]==id
        uncorrected_mass= mask.sum()
        peak_value      = np.max(distance[region][mask])
        threshold       = f(1.0/uncorrected_mass*np.max(distance[region][mask]))
        #threshold = f(uncorrected_mass)
        #threshold = f(np.median(distance[region][mask]))
        adjusted_mask   = distance[region] >= threshold

        corrected_regions[region] += id * np.logical_and(adjusted_mask, mask)
        peak_vals.append(peak_value)
        corrected_sizes.append((np.logical_and(adjusted_mask, mask)).sum())


    # step 3: return 3 things:
    # 1) corrected_regions (holding unique labels for each unique region)
    # 2) peak_values (float) of each region
    # 3) corrected size (int) of each region
    return corrected_regions, np.asarray(peak_vals), np.asarray(corrected_sizes)


def find_peak_to_thresh_relation(distance, sim, homedir, preload=True):

    # step 1: call the mainfucntion to return the filtered labels
    if preload:
        labels_wsF = np.load(homedir + 'hmf' + sim + '/src/uncorrected_labels' + sim + '.npy')
        distance = np.pad(distance, pad_width=64, mode='wrap')
        distance = nd.gaussian_filter(distance, sigma=2, mode='wrap')
    else:
        labels_wsF, distance = mainfunction(distance)
        np.save(homedir + 'hmf' + sim + '/src/uncorrected_labels' + sim + '.npy', labels_wsF)

    print(labels_wsF.shape, distance.shape)

    # step 2: for each region collect the masses varying with threshold
    filtered_regions    = nd.find_objects(labels_wsF)
    thresholds          = np.linspace(0, 1, 100)
    peak_vals           = []
    masses              = []

    from tqdm import tqdm
    for k, region in enumerate(tqdm(filtered_regions)):
        id = k+1

        mask        = labels_wsF[region]==id
        peak_value  = np.max(distance[region][mask])

        # now change the threshold and collect the masses
        object_mass = []
        object_quan = []
        for t in thresholds:
            adjusted_mask = distance[region] >= t
            combined_mask = np.logical_and(adjusted_mask, mask)

            object_mass.append(combined_mask.sum())
            #object_quan.append(combined_mask.sum() *
            #                   np.mean(distance[region][combined_mask]))
        peak_vals.append(1.0/object_mass[0] *
                         np.mean(distance[region][mask])
                                   * np.ones_like(thresholds))
        #peak_vals.append(object_mass[0]
        #                * np.ones_like(thresholds))
        masses.append(object_mass)


    # step 3: return 3 things
    # 1) the "mass-matrix"
    # 2) the peak_values
    # 3) the "threshold-matrix"
    f_threshs = [thresholds for m in masses]
    return np.asarray(masses), np.asarray(peak_vals), np.asarray(f_threshs)

