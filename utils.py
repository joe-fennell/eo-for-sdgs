import numpy as np


def image_histogram_equalization(image):
    """ Performs an image histogram equalisation
    
    Parameters
    ----------
    image : array-like
        image must have 3 dimensions with band in the 3rd dimension
    
    Returns
    -------
    image_corrected : array
        equalised image of same shape as input
    
    """
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), 256, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (255-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    
    if len(image.shape) == 3:
        return image_equalized.reshape(image.shape).astype('uint8')[:,:,::-1]
    else:
        return image_equalized.reshape(image.shape).astype('uint8')
