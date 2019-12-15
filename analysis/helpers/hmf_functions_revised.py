import numpy as np
from tqdm import tqdm
import scipy.ndimage as nd
from skimage.morphology import label, watershed, extrema


def fit(p):
    X = 3.75
    x0 = 3.9
    x1 = 4.95
    x2 = 5.65

    Y = 1.3
    y0 = 1.8
    y1 = 3.5
    y2 = 5.6

    if p < X:
        s = (y0 - Y) / (x0 - X)
        q = y0 - s * x0
        return s * p + q
    elif p < x1:
        s = (y1 - y0) / (x1 - x0)
        q = y1 - s * x1
        return s * p + q
    else:
        s = (y2 - y1) / (x2 - x1)
        q = y2 - s * x2
        return s * p + q

def best_index(mean_distances, raw_masses):

    x = np.linspace(3.0, 5.75, 1e2)
    y = np.asarray([fit(m) for m in x])

    raw_masses = np.log10(raw_masses)
    raw_masses[np.isnan(raw_masses)] = 0
    raw_masses = np.where(raw_masses==-np.inf, np.inf, raw_masses)
    #import matplotlib.pyplot as plt
    #plt.plot(x,y, 'k-')
    #plt.plot(raw_masses, mean_distances, 'g-')

    Fm = np.asarray([fit(m) for m in raw_masses])
    Fm = np.where(Fm==-np.inf, np.inf, Fm)
    diffs = abs(Fm - mean_distances)
    index = min(np.argsort(diffs)[:2]) # yes, min() is correct!
    #plt.plot(raw_masses[index], mean_distances[index], 'ro')
    #print(abs(Fm - mean_distances))
    #print(index, raw_masses[index], mean_distances[index])
    #plt.show()
    return index


def overplot_helper(distance, pw=64):

    # step 1: expand the domain periodically
    distance    = np.pad(distance, pad_width=pw, mode='wrap')
    image       = distance > 0


    # step 2: identify the maxima and the maxima inside original domain
    distance    = nd.gaussian_filter(distance, sigma=1, mode='wrap')
    h_maxima    = extrema.h_maxima(distance, h=0.05)

    # step 3: watershed segmentation; unfortunately have to do it for all maxima (not just 'in') and then filter
    labels_ws   = watershed(-distance, markers=label(h_maxima), mask=image, watershed_line=False)
    return labels_ws
    """
    # step 2: peak-to-threshold correction
    corrected_regions = np.zeros_like(labels_ws)
    filtered_regions = nd.find_objects(labels_ws)

    from tqdm import tqdm
    for k, region in enumerate(tqdm(filtered_regions)):
        id = k + 1

        mask = labels_ws[region] == id
        uncorrected_mass = []
        peak_value = []

        thresholds = np.linspace(0.0, 1.0, 50)

        for t in thresholds:
            adjusted_mask = distance[region] >= t
            combined_mask = np.logical_and(adjusted_mask, mask)

            uncorrected_mass.append(combined_mask.sum())
            peak_value.append(np.mean(uncorrected_mass[-1] ** (1. / 3.) / 4.0 * distance[region][combined_mask]))
        index = f(peak_value, uncorrected_mass)
        threshold = thresholds[index]
        adjusted_mask = distance[region] >= threshold

        corrected_regions[region] += id * np.logical_and(adjusted_mask, mask)

    # step 3: return 3 things:
    # 1) corrected_regions (holding unique labels for each unique region)
    # 2) peak_values (float) of each region
    # 3) corrected size (int) of each region
    return corrected_regions
    """


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

    # relabelling is necessary, since (intermediate) labels are now missing in label_wsF (were filtered out)
    labels_wsF = label(labels_wsF)
    return labels_wsF, distance


def retrieve_corrected_regions(distance, sim, preload=True):
    """
    :return:    1) corrected_regions (holding unique labels for each unique region)
                2) dmean_vals (float) of each region
                3) corrected size (int) of each region
    """

    # step 1: call the mainfunction to return the filtered labels or preload
    if preload:
        labels_wsF = np.load('src/uncorrected_labels' + sim + '.npy')
    else:
        labels_wsF, _ = mainfunction(distance, sim=sim)
        np.save('src/uncorrected_labels' + sim + '.npy', labels_wsF)

    distance = np.pad(distance, pad_width=64, mode='wrap')
    distance = nd.gaussian_filter(distance, sigma=2, mode='wrap')


    # step 2: correction
    corrected_regions   = np.zeros_like(labels_wsF)
    filtered_regions    = nd.find_objects(labels_wsF)
    dmean_vals          = []
    corrected_sizes     = []

    for k, region in enumerate(tqdm(filtered_regions)):
        id = k+1

        mask            = labels_wsF[region]==id
        uncorrected_mass= []
        dmean_value     = []

        thresholds = np.linspace(0, 1, 100)

        for t in thresholds:
            adjusted_mask = distance[region] >= t
            combined_mask = np.logical_and(adjusted_mask, mask)

            uncorrected_mass.append(combined_mask.sum())
            dmean_value.append(np.mean( uncorrected_mass[-1]**(1./3.)/4.0 *
                                        distance[region][combined_mask]))
        dmean_value = np.asarray(dmean_value)
        dmean_value[np.isnan(dmean_value)] = 0.0

        index           = best_index(dmean_value, uncorrected_mass)
        threshold       = thresholds[index]
        adjusted_mask   = distance[region] >= threshold

        corrected_regions[region] += id * np.logical_and(adjusted_mask, mask)
        dmean_vals.append(dmean_value)
        corrected_sizes.append(uncorrected_mass[index])

    # step 3: return 3 fields
    return corrected_regions, np.asarray(dmean_vals), np.asarray(corrected_sizes)


def find_correction(distance, sim, homedir, preload=True):

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
    thresholds          = np.linspace(0, 1, 15)
    dmean_vals           = []
    volumes              = []

    from tqdm import tqdm
    for k, region in enumerate(tqdm(filtered_regions)):
        id = k+1

        mask        = labels_wsF[region]==id

        # now change the threshold and collect the volumes (sizes) and means of un-normalized distances
        object_vol  = []
        object_quan = []
        for t in thresholds:
            adjusted_mask = distance[region] >= t
            combined_mask = np.logical_and(adjusted_mask, mask)

            object_vol.append(combined_mask.sum())
            object_quan.append(np.mean( object_vol[-1]**(1./3.)/4.0 *
                                        distance[region][combined_mask]))

        dmean_vals.append(object_quan)
        volumes.append(object_vol)

    dmean_vals = np.asarray(dmean_vals)
    dmean_vals[np.isnan(dmean_vals)] = 0.0

    # step 3: return 2 things
    # 1) the "volume-matrix"
    # 2) the dmean_values "matrix"
    return np.asarray(volumes), dmean_vals

