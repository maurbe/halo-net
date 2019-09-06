import numpy as np
import scipy.ndimage as nd
from skimage.morphology import label, watershed, extrema


def f(mean_distances, raw_masses):

    def fit(p):
        X = 3.75
        x0 = 3.9
        x1 = 4.95
        x2 = 5.65

        Y = 1.3
        y0 = 1.8
        y1 = 3.5
        y2 = 5.6

        if p < x0:
            s = (y0 - Y) / (x0 - X)
            q = y0 - s * x0
            return s * p + q
        elif p < x1:
            s = (y1 - y0) / (x1 - x0)
            q = y1 - s * x1
            return s * p + q
        else:# p < x2:
            s = (y2 - y1) / (x2 - x1)
            q = y2 - s * x2
            return s * p + q
        #else:
        #    return 5.6

    x = np.linspace(0.5, 5.75, 1e3)
    y = np.asarray([fit(m) for m in x])

    raw_masses = np.log10(raw_masses)
    raw_masses[np.isnan(raw_masses)] = 0

    #plt.plot(x,y, 'k-')
    #plt.plot(raw_masses, mean_distances, 'g-')

    diffs = abs(np.asarray([fit(m) for m in raw_masses]) - mean_distances)
    index = np.argmin(diffs)
    #plt.plot(raw_masses[index], mean_distances[index], 'ro')
    #print(abs(np.asarray([fit(m) for m in raw_masses]) - mean_distances))
    #print(index, raw_masses[index], mean_distances[index])
    #plt.show()
    return index


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

    from tqdm import tqdm
    for k, region in enumerate(tqdm(filtered_regions)):
        id = k+1

        mask            = labels_wsF[region]==id
        uncorrected_mass= []
        peak_value      = []

        thresholds = np.linspace(0.0, 1.0, 50)

        for t in thresholds:
            adjusted_mask = distance[region] >= t
            combined_mask = np.logical_and(adjusted_mask, mask)

            uncorrected_mass.append(combined_mask.sum())
            peak_value.append(np.mean(
                                        uncorrected_mass[-1]**(1./3.)/4.0 *
                                        distance[region][combined_mask]))
        peak_value = np.asarray(peak_value)
        peak_value[np.isnan(peak_value)] = 0.0

        index       = f(peak_value, uncorrected_mass)
        threshold =   thresholds[index]
        adjusted_mask   = distance[region] >= threshold

        corrected_regions[region] += id * np.logical_and(adjusted_mask, mask)
        peak_vals.append(peak_value)
        corrected_sizes.append(uncorrected_mass[index])

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
    thresholds          = np.linspace(0, 1, 30)
    peak_vals           = []
    masses              = []

    from tqdm import tqdm
    for k, region in enumerate(tqdm(filtered_regions)):
        id = k+1

        mask        = labels_wsF[region]==id
        #peak_value  = np.max(distance[region][mask])

        # now change the threshold and collect the masses
        object_mass = []
        object_quan = []
        for t in thresholds:
            adjusted_mask = distance[region] >= t
            combined_mask = np.logical_and(adjusted_mask, mask)

            object_mass.append(combined_mask.sum())
            object_quan.append(np.mean(
                                object_mass[-1]**(1./3.)/4.0 *
                                distance[region][combined_mask]))

        peak_vals.append( object_quan )
        #peak_vals.append(object_mass[0]
        #                #/ np.max(distance[region][mask])
        #                * np.ones_like(thresholds))
        masses.append(object_mass)

    peak_vals = np.asarray(peak_vals)
    peak_vals[np.isnan(peak_vals)] = 0.0

    # step 3: return 3 things
    # 1) the "mass-matrix"
    # 2) the peak_values
    # 3) the "threshold-matrix"
    f_threshs = [thresholds for m in masses]
    return np.asarray(masses), peak_vals, np.asarray(f_threshs)

