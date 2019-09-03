import numpy as np
import scipy.ndimage as nd
from skimage.morphology import label, watershed, extrema

import matplotlib.pyplot as plt
plt.figure()
def f(mean_distances, raw_masses):
    # Todo: MAKE sure that the thresholds DON'T go up all the way to 1
    # Todo: but only up to what is indicated in the plot (0.4?)
    N0 = 16 * 285
    def fit(m):

        P = [5.254736517726764, 7.112314761129568, 5.07115401589218, 3.417892033619879,
             6.220494922942824, 2.7256100546997932, 4.80787659198392, 1.9859665557554766,
             4.455193866685188, 3.9066683894320633, 4.748783114403779, 2.4089349140457097,
             2.6767924581297087, 2.5035738817908717, 1.7326016355676177, 2.103432783736956,
             5.384313221319651, 3.285739934193872, 2.2493481384640344, 0.0, 6.158211415609253,
             2.0114952998027906, 2.239712033777085, 1.3441221034471849, 2.284868318070065, 0.0,
             2.3567581453406525, 1.7783011898791676, 2.5751971352811536, 2.207983352604284][::-1] -0.3

        M =     [3.65963101, 3.725155,  3.79067898, 3.85620296, 3.92172695, 3.98725093,
                4.05277491, 4.1182989,  4.18382288, 4.24934687, 4.31487085, 4.38039483,
                4.44591882, 4.5114428,  4.57696679, 4.64249077, 4.70801475, 4.77353874,
                4.83906272, 4.9045867,  4.97011069, 5.03563467, 5.10115866, 5.16668264,
                5.23220662, 5.29773061, 5.36325459, 5.42877858, 5.49430256, 5.55982654]
                #5.62535053
        index = np.argmin(abs(M-m))
        return P[index]

    """
    def fit(m):
        anchor0 = 3.75
        anchor1 = 3.8
        anchor2 = 4.35
        if m < anchor1:
            s = 2.0 / (anchor1 - anchor0)
            q = anchor1 - s * anchor1
            return s*m+q
        elif m < anchor2:
            s = 0.52 / (anchor2 - anchor1)
            q = anchor2 - s * anchor2
            return 3.24#s*m+q
        else:
            return anchor2
    """
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

"""
def f(p):
    #m = -0.23 / 80000
    #q = 0.23
    #return 0.1#m * peak_val + q
    anchor1 = 1e4
    anchor2 = 10**(4.3)
    anchor3 = 1e5
    if p < anchor1:
        m = +0.3 / (anchor2 -anchor1)
        q = 0.3 - m * anchor2
        return m * p + q
    elif p <= anchor2:
        m = -(0.3 - 0.05) / (anchor3 - anchor2)
        q = 0.3 - m * anchor2
        return m * p + q
    else:
        return 0.05



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
                                        uncorrected_mass[0]**(1./3.)/4.0 *
                                        distance[region][combined_mask]))
        peak_value = np.asarray(peak_value)
        peak_value[np.isnan(peak_value)] = 0.0

        index       = f(peak_value, uncorrected_mass)
        threshold = thresholds[index]
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
    thresholds          = np.linspace(0, 1, 100)
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
                                object_mass[0]**(1./3.)/4.0 *
                                distance[region][combined_mask]))

        peak_vals.append( #-1 * np.mean( np.gradient(object_mass) )
                           object_quan)
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

