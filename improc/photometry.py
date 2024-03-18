
import numpy as np

from improc.tools import make_gaussian


# caching the soft-edge circles for faster calculations
CACHED_CIRCLES = []
CACHED_RADIUS_RESOLUTION = 0.01


def get_circle(radius, imsize=15, oversampling=100):
    """Get a soft-edge circle.

    This function will return a 2D array with a soft-edge circle of the given radius.

    Parameters
    ----------
    radius: float
        The radius of the circle.
    imsize: int
        The size of the 2D array to return. Must be square. Default is 15.
    oversampling: int
        The oversampling factor for the circle.
        Default is 100.

    Returns
    -------
    circle: np.ndarray
        A 2D array with the soft-edge circle.

    """
    # Check if the circle is already cached
    for circ in CACHED_CIRCLES:
        if np.abs(circ.radius - radius) < CACHED_RADIUS_RESOLUTION:
            return circ

    # Create the circle
    circ = Circle(radius, imsize=imsize, oversampling=oversampling)

    # Cache the circle
    CACHED_CIRCLES.append(circ)

    return circ


class Circle:
    def __init__(self, radius, imsize=15, oversampling=100):
        self.radius = radius
        self.imsize = imsize
        self.oversampling = oversampling

        # these include the circle, after being moved by sub-pixel shifts for all possible positions in x and y
        self.datacube = np.zeros((oversampling ** 2, imsize, imsize))

        for i in range(oversampling):
            for j in range(oversampling):
                x = i / oversampling
                y = j / oversampling
                self.datacube[i * oversampling + j] = self._make_circle(x, y)

    def _make_circle(self, x, y):
        """Generate the circles for a given sub-pixel shift in x and y. """

        if x < 0 or x > 1 or y < 0 or y > 1:
            raise ValueError("x and y must be between 0 and 1")

        # Create the circle
        xgrid, ygrid = np.meshgrid(np.arange(self.imsize), np.arange(self.imsize))
        xgrid = xgrid - self.imsize // 2 - x
        ygrid = ygrid - self.imsize // 2 - y
        r = np.sqrt(xgrid ** 2 + ygrid ** 2)
        im = 1 + self.radius - r
        im[r <= self.radius] = 1
        im[r > self.radius + 1] = 0
        # TODO: improve this with a better soft-edge function

        return im

    def get_image(self, dx, dy):
        """Get the circle with the given pixel shifts, dx and dy.

        Parameters
        ----------
        dx: float
            The shift in the x direction. Can be a fraction of a pixel.
        dy: float
            The shift in the y direction. Can be a fraction of a pixel.

        Returns
        -------
        im: np.ndarray
            The circle with the given shifts.
        """
        # Get the integer part of the shifts
        ix = int(np.floor(dx))
        iy = int(np.floor(dy))

        # Get the fractional part of the shifts
        fx = dx - ix
        fx = int(fx * self.oversampling)  # convert to oversampled pixels
        fy = dy - iy
        fy = int(fy * self.oversampling)  # convert to oversampled pixels

        # Get the circle
        im = self.datacube[fx * self.oversampling + fy, :, :]

        # roll and crop the circle to the correct position
        im = np.roll(im, -ix, axis=1)
        if ix > 0:
            im[:, -ix:] = 0
        else:
            im[:, :-ix] = 0
        im = np.roll(im, -iy, axis=0)
        if iy > 0:
            im[-iy:, :] = 0
        else:
            im[:-iy, :] = 0

        return im


def iterative_photometry(
        image, weight, flags, psf, radii=[3.0, 5.0, 7.0], annulus=[7.5, 10.0], iterations=3, verbose=False
):
    """Perform aperture and PSF photometry on an image, at positions, using a list of apertures.

    The "iterative" part means that it will use the starting positions but move the aperture centers
    around based on the centroid found using the PSF. The centroid will be used as the new position
    for the aperture and PSF photometry, and the new centroid will be updated.

    Parameters
    ----------
    image: np.ndarray
        The image to perform photometry on.
    weight: np.ndarray
        The weight map for the image.
    flags: np.ndarray
        The flags for the image.
    psf: np.ndarray or float scalar
        The PSF to use for photometry.
        If given as a float, will interpret that as a Gaussian
        with that FWHM, in units of pixels.
    radii: list or 1D array
        The apertures to use for photometry.
        Must be a list of positive numbers.
        In units of pixels!
        Default is [3, 5, 7].
    iterations: int
        The number of iterations to perform.
        Each iteration will refine the position of the aperture.
        Default is 3.
    verbose: bool
        If True, print out information about the progress.
        Default is False.

    Returns
    -------
    photometry: dict
        A dictionary with the output of the photometry.

    """
    # Make sure the image is a 2D array
    if len(image.shape) != 2:
        raise ValueError("Image must be a 2D array")

    # Make sure the weight is a 2D array
    if len(weight.shape) != 2:
        raise ValueError("Weight must be a 2D array")

    # Make sure the flags is a 2D array
    if len(flags.shape) != 2:
        raise ValueError("Flags must be a 2D array")

    # Make sure the PSF is a 2D array
    if np.isscalar(psf):
        psf = make_gaussian(psf, imsize=image.shape)
    else:
        if len(psf.shape) != 2:
            raise ValueError("PSF must be a 2D array")
    # TODO: still need to figure out how to actually use the PSF for photometry!

    # Make sure the apertures are a list or 1D array
    radii = np.atleast_1d(radii)
    if not np.all(radii > 0):
        raise ValueError("Apertures must be positive numbers")

    xgrid, ygrid = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    xgrid -= image.shape[1] // 2
    ygrid -= image.shape[0] // 2

    nandata = np.where(flags > 0, np.nan, image)

    # find a rough estimate of the centroid using non-tapered cutout
    cx = np.nansum(xgrid * nandata) / np.nansum(nandata)
    cy = np.nansum(ygrid * nandata) / np.nansum(nandata)
    cxx = np.nansum((xgrid - cx) ** 2 * nandata) / np.nansum(nandata)
    cyy = np.nansum((ygrid - cy) ** 2 * nandata) / np.nansum(nandata)
    cxy = np.nansum((xgrid - cx) * (ygrid - cy) * nandata) / np.nansum(nandata)

    # get some very rough estimates just so we have something in case of immediate failure of the loop
    fluxes = [np.nansum(nandata)] * len(radii)
    areas = [float(np.nansum(~np.isnan(nandata)))] * len(radii)
    background = 0.0
    variance = np.nanvar(nandata)

    photometry = dict(
        psf_flux=0.0,  # TODO: update this!
        psf_err=0.0,  # TODO: update this!
        psf_area=0.0,  # TODO: update this!
        radii=radii,
        fluxes=fluxes,
        areas=areas,
        background=background,
        variance=variance,
        offset_x=cx,
        offset_y=cy,
        moment_xx=cxx,
        moment_yy=cyy,
        moment_xy=cxy,
    )

    # Loop over the iterations
    for i in range(iterations):
        fluxes = np.zeros(len(radii))
        areas = np.zeros(len(radii))
        need_break = False

        # reposition based on the last centroids
        reposition_cx = cx
        reposition_cy = cy
        for j, r in enumerate(radii):
            # make a circle-mask based on the centroid position
            try:
                mask = get_circle(radius=r, imsize=nandata.shape[0]).get_image(reposition_cx, reposition_cy)
            except Exception as e:
                print(e)
                raise

            # mask the data and get the flux
            masked_data = nandata * mask
            fluxes[j] = np.nansum(masked_data)  # total flux, not per pixel!
            areas[j] = np.nansum(mask)  # save the number of pixels in the aperture

            # get an offset annulus to get a local background estimate
            inner = get_circle(radius=annulus[0], imsize=nandata.shape[0]).get_image(reposition_cx, reposition_cy)
            outer = get_circle(radius=annulus[1], imsize=nandata.shape[0]).get_image(reposition_cx, reposition_cy)
            annulus_map = outer - inner

            # background and variance only need to be calculated once (they are the same for all apertures)
            # but moments/centroids can be calculated for each aperture, but we will only want to save one
            # so how about we use the smallest one?
            if j == 0:  # smallest aperture only
                # TODO: consider replacing this with a hard-edge annulus and do median or sigma clipping on the pixels
                background = np.nansum(nandata * annulus_map) / np.nansum(annulus_map)  # b/g per pixel
                variance = np.nansum((nandata - background) * annulus_map) ** 2 / np.nansum(annulus_map)  # noise per pixel

                normalization = (fluxes[j] - background * areas[j])
                masked_data_bg = (nandata - background) * mask

                # update the centroids
                cx = np.nansum(xgrid * masked_data_bg) / normalization
                cy = np.nansum(ygrid * masked_data_bg) / normalization

                # update the second moments
                cxx = np.nansum((xgrid - cx) ** 2 * masked_data_bg) / normalization
                cyy = np.nansum((ygrid - cy) ** 2 * masked_data_bg) / normalization
                cxy = np.nansum((xgrid - cx) * (ygrid - cy) * masked_data_bg) / normalization

                # TODO: how to do PSF photometry with offsets and a given PSF? and get the error, too!

                # check that we got reasonable values! If not, break and keep the current values
                if np.isnan(cx) or cx > nandata.shape[1] or cx < 0:
                    need_break = True
                    break  # there's no point doing more radii if we are not going to save the results!
                if np.isnan(cy) or cy > nandata.shape[0] or cy < 0:
                    need_break = True
                    break  # there's no point doing more radii if we are not going to save the results!
                if np.nansum(mask) == 0 or np.nansum(annulus_map) == 0:
                    need_break = True
                    break  # there's no point doing more radii if we are not going to save the results!

        if need_break:
            break

        photometry['psf_flux'] = 0.0  # TODO: update this!
        photometry['psf_err'] = 0.0  # TODO: update this!
        photometry['psf_area'] = 0.0  # TODO: update this!
        photometry['radii'] = radii
        photometry['fluxes'] = fluxes
        photometry['areas'] = areas
        photometry['background'] = background
        photometry['variance'] = variance
        photometry['offset_x'] = cx
        photometry['offset_y'] = cy
        photometry['moment_xx'] = cxx
        photometry['moment_yy'] = cyy
        photometry['moment_xy'] = cxy

    # calculate from 2nd moments the width, ratio and angle of the source
    # ref: https://en.wikipedia.org/wiki/Image_moment
    major = np.sqrt(2 * (cxx + cyy + np.sqrt((cxx - cyy) ** 2 + 4 * cxy ** 2)))
    minor = np.sqrt(2 * (cxx + cyy - np.sqrt((cxx - cyy) ** 2 + 4 * cxy ** 2)))
    angle = np.arctan2(2 * cxy, cxx - cyy) / 2
    elongation = major / minor

    photometry['major'] = major
    photometry['minor'] = minor
    photometry['angle'] = angle
    photometry['elongation'] = elongation

    return photometry


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    c = get_circle(radius=3.0)
    plt.imshow(c.get_image(0.0, 0.0))
    plt.show()
    print('done')