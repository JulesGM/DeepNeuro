import numpy as np
from scipy.interpolate import griddata
from sklearn.preprocessing import scale

def gen_images(locs, features, nGridPoints, normalize=True,
               augment=False, pca=False, stdMult=0.1, n_components=2, edgeless=False):
    """

    Taken from https://github.com/pbashivan/EEGLearn/blob/master/eeg_cnn_lib.py

    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode
    :param loc: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param nGridPoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each feature over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param stdMult:     Standard deviation of noise for augmentation
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """

    feat_array_temp = []
    nElectrodes = locs.shape[0]  # Number of electrodes

    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0, "%s, %s" % (str(features.shape[1]), nElectrodes)
    n_colors = np.floor(features.shape[1] / nElectrodes).astype(np.int64)

    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes: nElectrodes * (c + 1)])

    """
    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], stdMult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], stdMult, pca=False, n_components=n_components)
    """

    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):nGridPoints * 1j,
                     min(locs[:, 1]):max(locs[:, 1]):nGridPoints * 1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, nGridPoints, nGridPoints]))

    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)

    for i in xrange(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(
                locs, feat_array_temp[c][i, :],
                (grid_x, grid_y), method='cubic')

        print('Interpolating {0}/{1}\r'.format(i + 1, nSamples))

    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])

    return np.swapaxes(np.asarray(temp_interp), 0, 1)  # swap axes to have [samples, colors, W, H]
