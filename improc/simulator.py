import time
import numpy as np
import scipy
from collections import defaultdict

from pipeline.parameters import Parameters


class SimPars(Parameters):

    def __init__(self, **kwargs):
        super().__init__()  # initialize base Parameters without passing arguments

        # sensor parameters
        self.image_size_x = self.add_par('image_size_x', 512, int, 'Image size in x')
        self.image_size_y = self.add_par('image_size_y', None, (int, None), 'Image size in y (assume square if None)')

        self.bias_mean = self.add_par('bias_mean', 100, (int, float), 'Mean bias level')
        self.bias_std = self.add_par(
            'bias_std', 1, (int, float),
            'variation in the actual mean bias value of each pixel. '
        )
        self.dark_current = self.add_par(
            'dark_current', 0.1, (int, float),
            'Dark current electrons per second per pixel'
        )
        self.read_noise = self.add_par('read_noise', 1.0, (int, float), 'Read noise rms per pixel')
        self.saturation_limit = self.add_par('saturation_limit', 5e4, (int, float), 'Saturation limit')
        self.bleed_fraction_x = self.add_par(
            'bleed_fraction_x', 0.0, float,
            'Fraction of electrons that bleed in the x direction if saturation is reached'
        )
        self.bleed_fraction_y = self.add_par(
            'bleed_fraction_y', 0.0, float,
            'Fraction of electrons that bleed in the y direction if saturation is reached'
        )
        self.pixel_qe_std = self.add_par(
            'pixel_qe_std', 0.01, float,
            'Standard deviation of the pixel quantum efficiency (around value of 1.0)'
        )
        self.gain_mean = self.add_par('gain_mean', 1.0, (int, float), 'Mean gain')
        self.gain_std = self.add_par('gain_std', 0.0, (int, float), 'Gain variation between pixels')

        # camera parameters
        self.vignette_amplitude = self.add_par('vignette_amplitude', 0.01, float, 'Vignette amplitude')
        self.vignette_radius = self.add_par(
            'vignette_radius', 280, float,
            'Inside this radius the vignette is ignored'
        )
        self.vignette_offset_x = self.add_par('vignette_offset_x', 10.0, float, 'Vignette offset in x')
        self.vignette_offset_y = self.add_par('vignette_offset_y', -5.0, float, 'Vignette offset in y')

        self.optic_psf_mode = self.add_par('optic_psf_mode', 'gaussian', str, 'Optical PSF mode')
        self.optic_psf_pars = self.add_par('optic_psf_pars', {'sigma': 1.0}, dict, 'Optical PSF parameters')

        # sky parameters
        self.background_mean = self.add_par('background_mean', 10.0, (int, float), 'Mean background level')
        self.background_std = self.add_par(
            'background_std', 1.0, (int, float),
            'Variation of background level between different sky instances'
        )
        self.background_minimum = self.add_par(
            'background_minimum', 6.0, (int, float),
            'Minimal value of the background (to avoid negative values). Will regenerate if below this value.'
        )
        self.transmission_mean = self.add_par('transmission_mean', 1.0, (int, float), 'Mean transmission (zero point)')
        self.transmission_std = self.add_par(
            'transmission_std', 0.01, (int, float),
            'Variation of transmission (zero point) between different sky instances'
        )
        self.transmission_minimum = self.add_par(
            'transmission_minimum', 0.5, (int, float),
            'Minimal value of the transmission (to avoid negative values). Will regenerate if below this value.'
        )

        self.seeing_mean = self.add_par('seeing_mean', 1.0, (int, float), 'Mean seeing')
        self.seeing_std = self.add_par('seeing_std', 0.0, (int, float), 'Seeing variation between images')
        self.seeing_minimum = self.add_par(
            'seeing_minimum', 0.5, (int, float),
            'Minimal value of the seeing (to avoid negative values). Will regenerate if below this value.'
        )

        self.atmos_psf_mode = self.add_par('atmos_psf_mode', 'gaussian', str, 'Atmospheric PSF mode')
        self.atmos_psf_pars = self.add_par('atmos_psf_pars', {'sigma': 1.0}, dict, 'Atmospheric PSF parameters')
        self.oversampling = self.add_par('oversampling', 5, int, 'Oversampling of the PSF')

        self.star_number = self.add_par('star_number', 1000, int, 'Number of stars (on average) to simulate')

        self.exact_star_number = self.add_par(
            'exact_star_number', True, bool,
            'If True, will use star_number as the exact number of stars'
            'in each image. If False, will choose from a Poisson distribution '
            'using star_number as the mean, every time make_stars is called, '
            'or when using make_image(new_stars=True).'
        )

        self.star_min_flux = self.add_par('star_min_flux', 100, (int, float), 'Minimum flux for the star power law')
        self.star_flux_power_law = self.add_par(
            'star_flux_power_law', -2.0, float,
            'Power law index for the flux distribution of stars'
        )
        self.star_position_std = self.add_par(
            'star_position_std', 0.0, float,
            'Standard deviation of the position of stars between images (in both x and y)'
        )

        self.galaxy_number = self.add_par(
            'galaxy_number', 0, int,
            'Number of galaxies (on average) to simulate'
        )

        self.exact_galaxy_number = self.add_par(
            'exact_galaxy_number', True, bool,
            'If True, will use galaxy_number as the exact number of galaxies'
            'in each image. If False, will choose from a Poisson distribution '
            'using galaxy_number as the mean, every time make_galaxies is called, '
            'or when using make_image(new_galaxies=True).'
        )

        self.galaxy_min_flux = self.add_par(
            'galaxy_min_flux', 100, (int, float),
            'Minimum flux for the galaxy power law'
        )

        self.galaxy_flux_power_law = self.add_par(
            'galaxy_flux_power_law', -2.0, float,
            'Power law index for the flux distribution of galaxies'
        )

        # use the star position std for the galaxy position (same scintillation motion)
        # self.galaxy_position_std = self.add_par(
        #     'galaxy_position_std', 0.0, float,
        #     'Standard deviation of the position of galaxies between images (in both x and y)'
        # )

        self.galaxy_min_width = self.add_par(
            'galaxy_min_width', 1.0, float,
            'Minimum width for the galaxy power law'
        )

        self.galaxy_width_power_law = self.add_par(
            'galaxy_width_power_law', -2.0, float,
            'Power law index for the width distribution of galaxies'
        )

        self.cosmic_ray_number = self.add_par('cosmic_ray_number', 0, int, 'Average number of cosmic rays per image')
        self.satellite_number = self.add_par('satellite_number', 0, int, 'Average number of satellites per image')

        self.exposure_time = self.add_par('exposure_time', 1, (int, float), 'Exposure time in seconds')

        self.show_runtimes = self.add_par('show_runtimes', False, bool, 'Show runtimes for each step of the simulation')

        # lock this object, so it can't be accidentally given the wrong name
        self._enforce_no_new_attrs = True

        self.override(kwargs)

    @property
    def imsize(self):
        """
        Return the image size as a tuple (x, y).
        """
        if self.image_size_y is None:
            return self.image_size_x, self.image_size_x
        else:
            return self.image_size_y, self.image_size_x


class SimTruth:
    """
    Contains the truth values for a simulated image.
    This object should be generated for each image,
    and saved along with it, to compare the analysis
    results to the ground truth values.
    """
    def __init__(self):
        # things involving the image sensor
        self.imsize = None  # the size of the image (y, x)
        self.bias_mean = None  # the mean counts for each pixel (e.g., 100)
        self.pixel_bias_std = None  # variations between pixels (the square of this is the var of a Poisson process)
        self.pixel_bias_map = None  # final result of the bias for each pixel

        self.gain_mean = None  # mean gain across image (used for finding the source noise)
        self.pixel_gain_std = None  # variation of gain between pixels
        self.pixel_gain_map = None  # final result of the gain of each pixel

        self.qe_mean = None  # total quantum efficiency of the sensor
        self.pixel_qe_std = None  # variation of the pixel quantum efficiency (around value of qe_mean)
        self.pixel_qe_map = None  # final map of the pixel QE values

        self.dark_current = None  # counts per second per pixel (mean and variance)
        self.read_noise = None  # read noise per pixel, added to the background_std
        self.saturation_limit = None  # pixels above this value will be clipped to this number

        # things involving the camera/telescope
        self.vignette_amplitude = None  # intensity of the vignette
        self.vignette_radius = None  # inside this radius the vignette is ignored
        self.vignette_offset_x = None  # vignette offset in x
        self.vignette_offset_y = None  # vignette offset in y
        self.vignette_map = None  # e.g., vignette

        self.oversampling = None  # how much do we need the PSF to be oversampled?
        self.optic_psf_mode = None  # e.g., 'gaussian'
        self.optic_psf_pars = None  # a dict with the parameters used to make this PSF
        self.optic_psf_image = None  # the final shape of the PSF for this image
        self.optic_psf_fwhm = None  # e.g., the total effect of optical aberrations on the width

        # things involving the sky
        self.background_mean = None  # mean sky+dark current across image
        self.background_std = None  # variation between images
        self.background_minimum = None  # minimal value of the background (to avoid negative values)
        self.background_instance = None  # the background for this specific image's sky

        self.transmission_mean = None  # average sky transmission
        self.transmission_std = None  # variation in transmission between images
        self.transmission_minimum = None  # minimal value of the transmission (to avoid negative values)
        self.transmission_instance = None  # the transmission for this specific image's sky

        self.seeing_mean = None  # the average seeing in this survey
        self.seeing_std = None  # the variation in seeing between images
        self.seeing_minimum = None  # minimal value of the seeing (to avoid negative values)
        self.seeing_instance = None  # the seeing for this specific image's sky

        self.atmos_psf_mode = None  # e.g., 'gaussian'
        self.atmos_psf_pars = None  # a dict with the parameters used to make this PSF
        self.atmos_psf_image = None  # the final shape of the PSF for this image
        self.atmos_psf_fwhm = None  # e.g., the seeing

        # things involving the specific set of objects in the sky
        self.star_number = None  # average number of stars in each field
        self.star_min_flux = None  # minimal flux for the star power law
        self.star_flux_power_law = None  # power law index of the flux of stars
        self.star_mean_fluxes = None  # for each star, the mean flux (in photons per total exposure time)
        self.star_mean_x_pos = None  # for each star, the mean x position
        self.star_mean_y_pos = None  # for each star, the mean y position
        self.star_position_std = None  # for each star, the variation in position (in both x and y)
        self.star_real_x_pos = None  # for each star, the real position on the image plane
        self.star_real_y_pos = None  # for each star, the real position on the image plane
        self.star_real_flux = None  # for each star, the real flux measured on the sky (before vignette, etc.)

        # additional random things that are unique to each image
        self.cosmic_ray_x_pos = None  # where each cosmic ray was
        self.cosmic_ray_y_pos = None  # where each cosmic ray was

        # TODO: add satellite trails

        # the noise and PSF info used to make the image
        self.psf = None  # the PSF used to make this image
        self.psf_downsampled = None  # the PSF, correctly downsampled as to retain the symmetric single peak pixel
        self.average_counts = None  # the final counts, not including noise
        self.noise_var_map = None  # the total variance from read, dark, sky b/g, and source noise
        self.total_bkg_var = None  # the total variance from read, dark, and sky b/g (not including source noise)


class SimSensor:
    """
    Container for the properties of a simulated sensor.
    """
    def __init__(self):
        self.bias_mean = None  # the mean counts for each pixel (e.g., 100)
        self.pixel_bias_std = None  # variations between pixels (the square of this is the var of a Poisson process)
        self.pixel_bias_map = None  # final result of the bias for each pixel

        self.gain_mean = None  # mean gain across image (used for finding the source noise)
        self.pixel_gain_std = None  # variation of gain between pixels
        self.pixel_gain_map = None  # final result of the gain of each pixel

        self.qe_mean = None  # total quantum efficiency of the sensor
        self.pixel_qe_std = None  # variation of the pixel quantum efficiency (around value of qe_mean)
        self.pixel_qe_map = None  # final map of the pixel QE values

        self.dark_current = None  # counts per second per pixel (mean and variance)
        self.read_noise = None  # read noise per pixel, added to the background_std
        self.saturation_limit = None  # pixels above this value will be clipped to this number
        self.bleed_fraction_x = None  # fraction of electrons that bleed in the x direction if saturation is reached
        self.bleed_fraction_y = None  # fraction of electrons that bleed in the y direction if saturation is reached

    def show_bias(self):
        """
        Show the bias map.
        """
        pass

    def show_pixel_qe(self):
        """
        Show the pixel quantum efficiency map.
        """
        pass

    def show_gain(self):
        """
        Show the gain map.
        """
        pass

    def show_saturated_stars(self):
        """
        Produce an image with some stars that
        have x1, x2, x4 and so on times the
        saturation limit and see how their shape
        looks after applying bleeds and saturation clipping.

        """
        pass


class SimCamera:
    """
    Container for the properties of a simulated camera.
    """
    def __init__(self):
        self.vignette_amplitude = None  # intensity of the vignette
        self.vignette_radius = None  # inside this radius the vignette is ignored
        self.vignette_offset_x = None  # vignette offset in x
        self.vignette_offset_y = None  # vignette offset in y
        self.vignette_map = None  # e.g., vignette

        self.oversampling = None  # how much do we need the PSF to be oversampled?
        self.optic_psf_mode = None  # e.g., 'gaussian'
        self.optic_psf_pars = None  # a dict with the parameters used to make this PSF
        self.optic_psf_image = None  # the final shape of the PSF for this image
        self.optic_psf_fwhm = None  # e.g., the total effect of optical aberrations on the width

    def make_optic_psf(self):
        """
        Make the optical PSF.
        Uses the optic_psf_mode to generate the PSF map
        and calculate the width (FWHM).

        Saves the results into optic_psf_image and optic_psf_fwhm.

        """
        if self.optic_psf_mode.lower().startswith('gauss'):
            self.optic_psf_image = make_gaussian(self.optic_psf_pars['sigma'] * self.oversampling)
            self.optic_psf_fwhm = self.optic_psf_pars['sigma'] * 2.355
        else:
            raise ValueError(f'PSF mode not recognized: {self.optic_psf_mode}')

    def make_vignette(self, imsize):
        """
        Make an image of the vignette part of the flat field.
        Input the imsize as a tuple of (imsize_x, imsize_y).
        """

        v = np.ones((imsize[0], imsize[1]))
        [xx, yy] = np.meshgrid(np.arange(-imsize[1] / 2, imsize[1] / 2), np.arange(-imsize[0] / 2, imsize[0] / 2))
        xx += self.vignette_offset_x
        yy += self.vignette_offset_y
        rr = np.sqrt(xx ** 2 + yy ** 2)
        v += (self.vignette_amplitude * (rr - self.vignette_radius)) ** 2
        v[rr < self.vignette_radius] = 1.0
        self.vignette_map = 1/v


class SimSky:
    """
    Container for the properties of a simulated sky.
    """
    def __init__(self):
        self.background_mean = None  # mean sky+dark current across image
        self.background_std = None  # variation between images
        self.background_minimum = None  # minimal value of the background (to avoid negative values)
        self.background_instance = None  # the background for this specific image's sky

        self.transmission_mean = None  # average sky transmission
        self.transmission_std = None  # variation in transmission between images
        self.transmission_minimum = None  # minimal value of the transmission (to avoid negative values)
        self.transmission_instance = None  # the transmission for this specific image's sky

        self.seeing_mean = None  # the average seeing in this survey
        self.seeing_std = None  # the variation in seeing between images
        self.seeing_minimum = None  # minimal value of the seeing (to avoid negative values)
        self.seeing_instance = None  # the seeing for this specific image's sky

        self.oversampling = None  # how much do we need the PSF to be oversampled?
        self.atmos_psf_mode = None  # e.g., 'gaussian'
        self.atmos_psf_pars = None  # a dict with the parameters used to make this PSF
        self.atmos_psf_image = None  # the final shape of the PSF for this image
        self.atmos_psf_fwhm = None  # e.g., the seeing

    def make_atmos_psf(self):
        """
        Use the psf mode, pars and the seeing_instance
        to produce an atmospheric PSF.
        This PSF is used to convolve the optical PSF to get the total PSF.

        """
        if self.atmos_psf_mode.lower().startswith('gauss'):
            self.atmos_psf_image = make_gaussian(self.atmos_psf_pars['sigma'] * self.oversampling)
            self.atmos_psf_fwhm = self.atmos_psf_pars['sigma'] * 2.355
        else:
            raise ValueError(f'PSF mode not recognized: {self.atmos_psf_mode}')


class SimStars:
    """
    Container for the properties of a simulated star field.
    """
    def __init__(self):
        self.star_number = None  # average number of stars in each field
        self.star_min_flux = None  # minimal flux for the star power law
        self.star_flux_power_law = None  # power law index of the flux of stars
        self.star_mean_fluxes = None  # for each star, the mean flux (in photons per total exposure time)
        self.star_mean_x_pos = None  # for each star, the mean x position
        self.star_mean_y_pos = None  # for each star, the mean y position
        self.star_position_std = None  # for each star, the variation in position (in both x and y)

    def make_star_list(self, imsize):
        """
        Make a field of stars.
        Uses the power law to draw random mean fluxes,
        and uniform positions for each star on the sensor.
        The input, imsize, is a tuple of (imsize_x, imsize_y).
        """
        rng = np.random.default_rng()
        alpha = abs(self.star_flux_power_law) - 1
        self.star_mean_fluxes = self.star_min_flux / rng.power(alpha, self.star_number)
        self.star_mean_x_pos = rng.uniform(-0.01, 1.01, self.star_number) * imsize[1]
        self.star_mean_y_pos = rng.uniform(-0.01, 1.01, self.star_number) * imsize[0]

    def add_extra_stars(self, imsize, flux=None, x=None, y=None, number=1):
        """Add one more star to the star field. This is explicitly added by the user on top of the star_number.

        Parameters
        ----------
        imsize: tuple
            The size of the image (y, x)
        flux: float (optional)
            The flux of the new star. If None (default), will randomly choose a flux from the power law.
        x: float (optional)
            The x position of the new star. If None (default), will randomly choose a position.
        y: float (optional)
            The y position of the new star. If None (default), will randomly choose a position.
        number: int
            The number of stars to add. Default is 1.
            If any of (flux, x, y) are given as an array, then
            the number must match the length of that array.
            If all are given, number is ignored.

        """
        rng = np.random.default_rng()
        alpha = abs(self.star_flux_power_law) - 1
        flux = self.star_min_flux / rng.power(alpha, number) if flux is None else flux
        x = rng.uniform(-0.01, 1.01, number) * imsize[1] if x is None else x
        y = rng.uniform(-0.01, 1.01, number) * imsize[0] if y is None else y

        if not isinstance(x, np.ndarray):
            x = np.array([x])
        if not isinstance(y, np.ndarray):
            y = np.array([y])
        if not isinstance(flux, np.ndarray):
            flux = np.array([flux])

        if len(x) != len(y) or len(flux) != len(x):
            raise ValueError(f'Size mismatch between flux ({len(flux)}), x ({len(x)}) and y ({len(y)}).')

        self.star_mean_fluxes = np.append(self.star_mean_fluxes, np.array(flux))
        self.star_mean_x_pos = np.append(self.star_mean_x_pos, np.array(x))
        self.star_mean_y_pos = np.append(self.star_mean_y_pos, np.array(y))

    def remove_stars(self, number=1):
        """Remove the latest few stars (default is only one) from the star field. """
        if number > 0:
            self.star_mean_fluxes = self.star_mean_fluxes[:-number]
            self.star_mean_x_pos = self.star_mean_x_pos[:-number]
            self.star_mean_y_pos = self.star_mean_y_pos[:-number]

    def get_star_x_values(self):
        """
        Return the positions of the stars (in pixel coordinates)
        after possibly applying small astrometric shifts (e.g., scintillations)
        to the mean star positions.

        """
        x = self.star_mean_x_pos.copy()
        if self.star_position_std is not None:
            x += np.random.normal(0, self.star_position_std, len(x))

        return x

    def get_star_y_values(self):
        """
        Return the positions of the stars (in pixel coordinates)
        after possibly applying small astrometric shifts (e.g., scintillations)
        to the mean star positions.

        """
        y = self.star_mean_y_pos.copy()
        if self.star_position_std is not None:
            y += np.random.normal(0, self.star_position_std, len(y))

        return y

    def get_star_flux_values(self):
        """
        Return the fluxes of the stars (in photons per total exposure time)
        after possibly applying a flux change due to e.g., occultations/flares,
        or due to scintillation noise.
        (TODO: this is not yet implemented!)

        """
        return self.star_mean_fluxes


class SimGalaxies:
    """
    Container for the properties of a set of simulated galaxies.

    Galaxies here are a very simple object,
    a bulge with a Sersic profile, and a disk with an exponential profile.
    The bulge is spherically symmetric, while the disk is very thin
    and has a uniformly chosen cos(i) inclination.
    The ratio of total flux in the bulge vs disk is chosen randomly
    (from???)
    as is the ratio of the bulge length scale vs. the disk length scale.
    (we need to figure out if the ranges for these ratios are fixed or
    a user-defined parameter).

    """

    runtime = defaultdict(float)

    def __init__(self):
        self.galaxy_number = None  # average number of galaxies in each field

        self.galaxy_min_flux = None  # minimal flux for the galaxy power law
        self.galaxy_flux_power_law = None  # power law index of the flux of galaxies
        self.galaxy_mean_fluxes = None  # for each galaxy, the mean flux (in photons per total exposure time)

        self.galaxy_mean_x_pos = None  # for each galaxy, the mean x position
        self.galaxy_mean_y_pos = None  # for each galaxy, the mean y position
        self.galaxy_position_std = None  # for each galaxy, the variation in position (in both x and y)

        self.galaxy_min_width = None  # minimal width for the galaxy power law
        self.galaxy_width_power_law = None  # power law index of the width of galaxies
        self.galaxy_width_values = None  # for each galaxy, the width (in pixels)

        self.galaxy_cos_is = None  # for each galaxy, the cos(inclination), 1 means face on, 0 means edge on
        self.galaxy_rotations = None  # for each galaxy, the position angle (rotation) from North, towards East, in deg

        self.galaxy_sersic_flux_ratios = None  # the ratio of sersic flux to exponential flux for each galaxy
        self.galaxy_sersic_scale_ratios = None  # the length scale of the sersic profile relative to width, per galaxy
        self.galaxy_exp_scale_ratios = None  # the length scale of the exponential profile relative to width, per galaxy

    def make_galaxy_list(self, imsize):
        """
        Make a field of galaxies.
        Uses the power law to draw random mean fluxes
        and widths, and uniform orientations (both in inclination
        and in position angle) and positions.
        The input, imsize, is a tuple of (imsize_x, imsize_y).
        """
        rng = np.random.default_rng()
        alpha_f = abs(self.galaxy_flux_power_law) - 1
        self.galaxy_mean_fluxes = self.galaxy_min_flux / rng.power(alpha_f, self.galaxy_number)

        self.galaxy_mean_x_pos = rng.uniform(-0.01, 1.01, self.galaxy_number) * imsize[1]
        self.galaxy_mean_y_pos = rng.uniform(-0.01, 1.01, self.galaxy_number) * imsize[0]

        alpha_w = abs(self.galaxy_flux_power_law) - 1
        self.galaxy_width_values = self.galaxy_min_width / rng.power(alpha_w, self.galaxy_number)

        # use a minimal cos_i>0.1 to prevent very edge-on galaxies (as galaxies also have some thickness)
        self.galaxy_cos_is = rng.uniform(0.1, 1, self.galaxy_number)
        self.galaxy_rotations = rng.uniform(0, 360, self.galaxy_number)
        self.galaxy_sersic_flux_ratios = 10 ** rng.uniform(-2, -1, self.galaxy_number)
        self.galaxy_sersic_scale_ratios = rng.uniform(0.5, 2, self.galaxy_number) * self.galaxy_width_values
        self.galaxy_exp_scale_ratios = rng.uniform(0.5, 2, self.galaxy_number)

    def add_extra_galaxies(
            self,
            imsize=(100, 100),
            mean_x_pos=None,
            mean_y_pos=None,
            fluxes=None,
            widths=None,
            exp_scale_ratios=None,
            sersic_scale_ratios=None,
            sersic_flux_ratios=None,
            cos_is=None,
            rotations=None,
            number=1,
    ):
        """Add a galaxy (or multiple galaxies) to the list of galaxies

        Parameters
        ----------
        imsize: tuple, default (100, 100)
            The size of the image (y, x).
        mean_x_pos: float, list or ndarray of floats (optional)
            The mean x position of the galaxy. If None (default), will randomly choose a position.
        mean_y_pos: float, list or ndarray of floats (optional)
            The mean y position of the galaxy. If None (default), will randomly choose a position.
        fluxes: float, list or ndarray of floats (optional)
            The flux of the exponential profile. If None (default), will randomly choose a flux.
        widths: float, list or ndarray of floats (optional)
            The width of the galaxy. If None (default), will randomly choose a width.
        exp_scale_ratios: float, list or ndarray of floats (optional)
            The ratio of the exponential scale length to the width. If None (default), will randomly choose a ratio.
        sersic_scale_ratios: float, list or ndarray of floats (optional)
            The ratio of the sersic scale length to the width. If None (default), will randomly choose a ratio.
        sersic_flux_ratios: float, list or ndarray of floats (optional)
            The flux of the sersic profile relative to the main flux. If None (default), will randomly choose a flux.
        cos_is: float (optional)
            The inclination angle cosine that determines how flat and narrow
            the disk of the galaxy will be. One means face-on, zero means edge-on.
            If not given, will choose a uniform cos_i in range [0.1,1].
        rotations: float (optional)
            The rotation angle of the galaxy, in degrees.
            Zero means the galaxy is oriented North-South, and the angle increases
            towards East. If not given, will choose a uniform rotation in range [0,360].
        number: int, (optional) default 1
            The number of galaxies to add.
            If any of the above parameters (except imsize) is given as a
            list/ndarray then this number will be determined by the length
            of that list/ndarray.
            Any scalars given will be broadcast to this number.
            If all inputs are given as scalars, or none are given,
            will assume number=1.
        """
        rng = np.random.default_rng()

        pars = [
            mean_x_pos,
            mean_y_pos,
            fluxes,
            widths,
            exp_scale_ratios,
            sersic_scale_ratios,
            sersic_flux_ratios,
            cos_is,
            rotations,
        ]
        # if any parameters are given as lists/arrays, use the length of the first parameter as number
        new_number = next((len(p) for p in pars if isinstance(p, (list, np.ndarray))), None)
        if new_number is not None:
            number = new_number

        # check all list/array parameters are of consistent length
        for p, i in enumerate(pars):
            if isinstance(p, (list, np.ndarray)) and len(p) != number:
                raise ValueError(
                    f'Size mismatch between parameters: {len(p)} vs {number} (parameter {i+2}).')

        if mean_x_pos is None:
            mean_x_pos = rng.uniform(-0.01, 1.01, self.galaxy_number) * imsize[1]
        elif isinstance(mean_x_pos, (int, float)):
            mean_x_pos = np.full(number, mean_x_pos)
        if mean_y_pos is None:
            mean_y_pos = rng.uniform(-0.01, 1.01, self.galaxy_number) * imsize[0]
        elif isinstance(mean_y_pos, (int, float)):
            mean_y_pos = np.full(number, mean_y_pos)

        self.galaxy_mean_x_pos = np.append(self.galaxy_mean_x_pos, mean_x_pos)
        self.galaxy_mean_y_pos = np.append(self.galaxy_mean_y_pos, mean_y_pos)

        if fluxes is None:
            alpha_f = abs(self.galaxy_flux_power_law) - 1
            fluxes = self.galaxy_min_flux / rng.power(alpha_f, number)
        elif isinstance(fluxes, (int, float)):
            fluxes = np.full(number, fluxes)
        self.galaxy_mean_fluxes = np.append(self.galaxy_mean_fluxes, fluxes)

        if widths is None:
            alpha_w = abs(self.galaxy_flux_power_law) - 1
            widths = self.galaxy_min_width / rng.power(alpha_w, number)
        elif isinstance(widths, (int, float)):
            widths = np.full(number, widths)
        self.galaxy_width_values = np.append(self.galaxy_width_values, widths)

        if exp_scale_ratios is None:
            exp_scale_ratios = rng.uniform(0.5, 2, number) * widths
        elif isinstance(exp_scale_ratios, (int, float)):
            exp_scale_ratios = np.full(number, exp_scale_ratios)
        self.galaxy_exp_scale_ratios = np.append(self.galaxy_exp_scale_ratios, exp_scale_ratios)

        if sersic_scale_ratios is None:
            sersic_scale_ratios = rng.uniform(0.5, 2, number) * widths
        elif isinstance(sersic_scale_ratios, (int, float)):
            sersic_scale_ratios = np.full(number, sersic_scale_ratios)
        self.galaxy_sersic_scale_ratios = np.append(self.galaxy_sersic_scale_ratios, sersic_scale_ratios)

        if sersic_flux_ratios is None:
            sersic_flux_ratios = 10 ** rng.uniform(-2, -1, number)
        elif isinstance(sersic_flux_ratios, (int, float)):
            sersic_flux_ratios = np.full(number, sersic_flux_ratios)
        self.galaxy_sersic_flux_ratios = np.append(self.galaxy_sersic_flux_ratios, sersic_flux_ratios)

        if cos_is is None:
            cos_is = rng.uniform(0.1, 1, number)
        elif isinstance(cos_is, (int, float)):
            cos_is = np.full(number, cos_is)

        if not all(0.1 <= c <= 1 for c in cos_is):
            raise ValueError(f'Invalid cos_i values (outside [0.1,1]): {cos_is}')

        self.galaxy_cos_is = np.append(self.galaxy_cos_is, cos_is)

        if rotations is None:
            rotations = rng.uniform(0, 360, number)
        elif isinstance(rotations, (int, float)):
            rotations = np.full(number, rotations)
        self.galaxy_rotations = np.append(self.galaxy_rotations, rotations)

    def remove_galaxies(self, number=1):
        """Remove the latest few galaxies (default is only one) from the galaxy field. """
        if number > 0:
            self.galaxy_mean_fluxes = self.galaxy_mean_fluxes[:-number]
            self.galaxy_mean_x_pos = self.galaxy_mean_x_pos[:-number]
            self.galaxy_mean_y_pos = self.galaxy_mean_y_pos[:-number]
            self.galaxy_width_values = self.galaxy_width_values[:-number]
            self.galaxy_cos_is = self.galaxy_cos_is[:-number]
            self.galaxy_rotations = self.galaxy_rotations[:-number]
            self.galaxy_sersic_flux_ratios = self.galaxy_sersic_flux_ratios[:-number]
            self.galaxy_sersic_scale_ratios = self.galaxy_sersic_scale_ratios[:-number]
            self.galaxy_exp_scale_ratios = self.galaxy_exp_scale_ratios[:-number]

    def get_galaxy_x_values(self):
        """
        Return the positions of the galaxies (in pixel coordinates)
        after possibly applying small astrometric shifts (e.g., scintillations)
        to the mean galaxy positions.

        """
        x = self.galaxy_mean_x_pos.copy()
        if self.galaxy_position_std is not None:
            x += np.random.normal(0, self.galaxy_position_std, len(x))

        return x

    def get_galaxy_y_values(self):
        """
        Return the positions of the galaxies (in pixel coordinates)
        after possibly applying small astrometric shifts (e.g., scintillations)
        to the mean galaxy positions.

        """
        y = self.galaxy_mean_y_pos.copy()
        if self.galaxy_position_std is not None:
            y += np.random.normal(0, self.galaxy_position_std, len(y))

        return y

    def get_galaxy_flux_values(self):
        """
        Return the fluxes of the galaxies (in photons per total exposure time).
        """
        return self.galaxy_mean_fluxes  # do we ever need to add noise to this?

    @classmethod
    def make_galaxy_image(
            cls,
            imsize=(100, 100),
            center_x=None,
            center_y=None,
            exp_scale=1.0,
            sersic_scale=1.0,
            exp_flux=10,
            sersic_flux=0.1,
            cos_i=None,
            rotation=None,
            cutoff_radius=None,
    ):
        """Make an image of a galaxy, with a bulge and disk.

        Parameters
        ----------
        imsize: tuple, default (100, 100)
            The size of the image (y, x).
        center_x: float, default None
            The x position of the galaxy center inside the image.
            If given as None, will position at the center of the image.
        center_y: float, default None
            The y position of the galaxy center inside the image.
            If given as None, will position at the center of the image.
        exp_scale: float, default 1.0
            The length scale used to make the disk.
        sersic_scale: float, default 1.0
            The length scale used to make the bulge.
        exp_flux: float, default 1.0
            The total flux of the disk.
        sersic_flux: float, default 0.01
            The total flux of the bulge.
            Note that this is a very sharp profile,
            so its contribution to the flux is usually small.
        cos_i: float, default None
            The inclination angle cosine that determines how flat and narrow
            the disk of the galaxy will be. One means face-on, zero means edge-on.
            If not given, will choose a uniform cos_i in range [0.1,1].
        rotation: float, default None
            The rotation angle of the galaxy, in degrees.
            Zero means the galaxy is oriented North-South, and the angle increases
            towards East. If not given, will choose a uniform rotation in range [0,360].
        cutoff_radius: float, default None
            If given, will add an external edge to the galaxy
            (should be at least a few times larger than the other scales)
            that adds an exponential multiplier with a short scale length
            to all pixels outside the cutoff scale.
        Returns
        -------
        galaxy_image: np.ndarray
            The image of the galaxy, normalized to total_flux.
        """
        if center_x is None:
            center_x = imsize[1] // 2
        if center_y is None:
            center_y = imsize[0] // 2
        if cos_i is None:
            cos_i = np.random.uniform(0.1, 1)
        if rotation is None:
            rotation = np.random.uniform(0, 360)

        rotation = np.deg2rad(rotation)

        # regular coordinate grid
        t0 = time.time()
        x0, y0 = np.meshgrid(np.arange(imsize[1]), np.arange(imsize[0]))
        cls.runtime['meshgrid'] += time.time() - t0

        # transform the coordinates using rotation and translation
        t0 = time.time()
        x = (x0 - center_x) * np.cos(rotation) + (y0 - center_y) * np.sin(rotation)
        y = (y0 - center_y) * np.cos(rotation) - (x0 - center_x) * np.sin(rotation)
        r = np.sqrt(x ** 2 + y ** 2)
        cls.runtime['transform'] += time.time() - t0

        # first make the bulge using a sersic profile
        t0 = time.time()
        r0 = sersic_scale
        bulge = np.exp( -7.67 * np.power(r / r0, 1 / 4) )
        bulge *= sersic_flux / np.sum(bulge)
        cls.runtime['bulge'] += time.time() - t0

        # now make the disk
        t0 = time.time()
        r0 = exp_scale
        disk = np.exp(-1.67 * np.sqrt( (x / r0 / cos_i) ** 2 + (y / r0) ** 2) )
        disk *= exp_flux / np.sum(disk)
        cls.runtime['disk'] += time.time() - t0

        # add them together
        t0 = time.time()
        galaxy_image = bulge + disk
        cls.runtime['add'] += time.time() - t0

        # add cutoff
        if cutoff_radius is not None:
            t0 = time.time()
            cutoff = np.ones(galaxy_image.shape)  # inside the radius there is no change
            # use the disk scale as the length scale for the cutoff
            cutoff[r > cutoff_radius] = np.exp(-5.0 * (r[r > cutoff_radius] - cutoff_radius) / exp_scale)
            galaxy_image *= cutoff
            cls.runtime['cutoff'] += time.time() - t0

        return galaxy_image

    @classmethod
    def add_galaxy_to_image(
            cls,
            image,
            center_x=None,
            center_y=None,
            exp_scale=1.0,
            sersic_scale=1.0,
            exp_flux=10,
            sersic_flux=0.1,
            cos_i=None,
            rotation=None,
            cutoff_radius=None,
    ):
        """Add a galaxy to an existing image, by making a small stamp image with the galaxy and placing it in the image.

        Parameters
        ----------
        image: np.ndarray
            The image to add the galaxy to.
        center_x: float, default None
            The x position of the galaxy center inside the image.
            If given as None, will position at a random place in the image.
        center_y: float, default None
            The y position of the galaxy center inside the image.
            If given as None, will position at a random place in the image.
        exp_scale: float, default 1.0
            The length scale used to make the disk.
        sersic_scale: float, default 1.0
            The length scale used to make the bulge.
        exp_flux: float, default 1.0
            The total flux of the disk.
        sersic_flux: float, default 0.01
            The total flux of the bulge.
            Note that this is a very sharp profile,
            so its contribution to the flux is usually small.
        cos_i: float, default None
            The inclination angle cosine that determines how flat and narrow
            the disk of the galaxy will be. One means face-on, zero means edge-on.
            If not given, will choose a uniform cos_i in range [0.1,1].
        rotation: float, default None
            The rotation angle of the galaxy, in degrees.
            Zero means the galaxy is oriented North-South, and the angle increases
            towards East. If not given, will choose a uniform rotation in range [0,360].
        cutoff_radius: float, default None
            If given, will add an external edge to the galaxy
            (should be at least a few times larger than the other scales)
            that adds an exponential multiplier with a short scale length
            to all pixels outside the cutoff scale.
        """
        if center_x is None:
            center_x = np.random.uniform(0, image.shape[1])
        if center_y is None:
            center_y = np.random.uniform(0, image.shape[0])

        # the sub-pixel shift of this galaxy
        offset_x = center_x - int(center_x)
        offset_y = center_y - int(center_y)

        # make sure we have all the parameters set to something
        if cos_i is None:
            cos_i = np.random.uniform(0.1, 1)
        if rotation is None:
            rotation = np.random.uniform(0, 360)

        # estimate the size of the image needed to make this galaxy
        imsize = int(np.ceil(12 * max(exp_scale, sersic_scale))) + 1
        new_center_x = imsize // 2 + offset_x
        new_center_y = imsize // 2 + offset_y

        # in some cases the required "stamp" is larger than the original image!
        if imsize > image.shape[0] or imsize > image.shape[1]:
            imsize = image.shape
            use_assignment = False
            new_center_x = center_x
            new_center_y = center_y
        else:
            imsize = (imsize, imsize)
            use_assignment = True

        # make the galaxy image
        galaxy_image = cls.make_galaxy_image(
            imsize=imsize,
            center_x=new_center_x,
            center_y=new_center_y,
            exp_scale=exp_scale,
            sersic_scale=sersic_scale,
            exp_flux=exp_flux,
            sersic_flux=sersic_flux,
            cos_i=cos_i,
            rotation=rotation,
            cutoff_radius=cutoff_radius,
        )

        # add it to the image
        if use_assignment:
            x_trim_start = 0
            x_start = int(center_x) - imsize[1] // 2
            x_trim_end = imsize[1]
            x_end = x_start + imsize[1]

            if x_end > image.shape[1]:
                x_trim_end = imsize[1] - (x_end - image.shape[1])
                x_end = image.shape[1]

            if x_start < 0:
                x_trim_start = -x_start
                x_start = 0

            y_trim_start = 0
            y_start = int(center_y) - imsize[0] // 2
            y_trim_end = imsize[0]
            y_end = y_start + imsize[0]

            if y_start < 0:
                y_trim_start = -y_start
                y_start = 0

            if y_end > image.shape[0]:
                y_trim_end = imsize[0] - (y_end - image.shape[0])
                y_end = image.shape[0]

            # galaxy_image += 1  # debug only!
            image[y_start:y_end, x_start:x_end] += galaxy_image[y_trim_start:y_trim_end, x_trim_start:x_trim_end]

        else:
            image += galaxy_image  # the required galaxy image is so big we just make it on the same size as the image


class Simulator:
    """
    Make simulated images for testing image processing techniques.
    """

    def __init__(self, **kwargs):
        self.pars = SimPars(**kwargs)

        # classes holding parts of the simulation
        self.sensor = None
        self.camera = None
        self.sky = None
        self.stars = None
        self.galaxies = None

        # holds the truth values for this image
        self.truth = None

        # intermediate variables
        # fluxes coming from the stars
        self.star_x = None
        self.star_y = None
        self.star_f = None

        self.galaxy_x = None
        self.galaxy_y = None
        self.galaxy_f = None

        self.psf = None  # we are cheating because this includes both the optical and atmospheric PSFs
        self.psf_downsampled = None # the PSF, correctly downsampled as to retain the symmetric single peak pixel
        self.flux_top = None  # this is the mean number of photons hitting the top of the atmosphere

        # adding the sky into the mix
        self.flux_with_sky = None  # this is mean photons after passing through the atmosphere, adding the background

        # adding the effect of the camera optics
        self.flux_vignette = None  # average number of photons after passing through the aperture vignette

        # now the photons are absorbed into the sensor pixels
        self.electrons = None  # average number of electrons in each pixel, considering QE

        self.average_counts = None  # the final counts, not including noise
        self.noise_var_map = None  # the total variance from read, dark, sky b/g, and source noise
        self.total_bkg_var = None  # the total variance from read, dark, and sky b/g (not including source noise)

        # outputs:
        self.image = None

    def make_sensor(self):
        """
        Generate a sensor and save it to the simulator.
        This includes all the properties of the pixels
        and amplifiers and readout electronics.

        Usually this does not change between images
        from the same survey.
        """
        self.sensor = SimSensor()

        self.sensor.bias_mean = self.pars.bias_mean
        self.sensor.pixel_bias_std = self.pars.bias_std

        self.sensor.pixel_bias_map = np.random.normal(
            self.sensor.bias_mean,
            self.sensor.pixel_bias_std,
            size=self.pars.imsize
        )

        self.sensor.gain_mean = self.pars.gain_mean
        self.sensor.pixel_gain_std = self.pars.gain_std
        self.sensor.pixel_gain_map = np.random.normal(
            self.sensor.gain_mean,
            self.sensor.pixel_gain_std,
            size=self.pars.imsize
        )
        self.sensor.pixel_gain_map[self.sensor.pixel_gain_map < 0] = 0.0

        self.sensor.pixel_qe_std = self.pars.pixel_qe_std
        self.sensor.pixel_qe_map = np.random.normal(1.0, self.sensor.pixel_qe_std, size=self.pars.imsize)
        self.sensor.pixel_qe_map[self.sensor.pixel_qe_map < 0] = 0.0

        self.sensor.dark_current = self.pars.dark_current
        self.sensor.read_noise = self.pars.read_noise
        self.sensor.saturation_limit = self.pars.saturation_limit
        self.sensor.bleed_fraction_x = self.pars.bleed_fraction_x
        self.sensor.bleed_fraction_y = self.pars.bleed_fraction_y

    def make_camera(self):
        """
        Generate a camera and save it to the simulator.
        This includes all the properties of the optics
        like the optical PSF (not including atmospheric seeing)
        and the flat field (vignetting).
        """
        self.camera = SimCamera()
        self.camera.vignette_amplitude = self.pars.vignette_amplitude
        self.camera.vignette_radius = self.pars.vignette_radius
        self.camera.vignette_offset_x = self.pars.vignette_offset_x
        self.camera.vignette_offset_y = self.pars.vignette_offset_y

        self.camera.make_vignette(self.pars.imsize)

        self.camera.optic_psf_mode = self.pars.optic_psf_mode
        self.camera.optic_psf_pars = self.pars.optic_psf_pars

    def make_sky(self):
        """
        Generate an instance of a sky. This will usually stay the same
        when taking a series of images at the same pointing.
        We will assume that if the pointing changes, the sky changes
        values like seeing and background.

        """
        self.sky = SimSky()
        self.sky.background_mean = self.pars.background_mean
        self.sky.background_std = self.pars.background_std
        self.sky.background_minimum = self.pars.background_minimum
        if self.sky.background_minimum >= self.sky.background_mean:
            raise ValueError('background_minimum must be less than background_mean')

        for i in range(100):
            self.sky.background_instance = np.random.normal(self.sky.background_mean, self.sky.background_std)
            if self.sky.background_instance >= self.sky.background_minimum:
                break
        else:
            raise RuntimeError('Could not generate a background instance above the minimum value')

        self.sky.transmission_mean = self.pars.transmission_mean
        self.sky.transmission_std = self.pars.transmission_std
        self.sky.transmission_minimum = self.pars.transmission_minimum
        if self.sky.transmission_minimum >= self.sky.transmission_mean:
            raise ValueError('transmission_minimum must be less than transmission_mean')

        for i in range(100):
            self.sky.transmission_instance = np.random.normal(self.sky.transmission_mean, self.sky.transmission_std)
            if self.sky.transmission_instance >= self.sky.transmission_minimum:
                break
        else:
            raise RuntimeError('Could not generate a transmission instance above the minimum value')

        self.sky.seeing_mean = self.pars.seeing_mean
        self.sky.seeing_std = self.pars.seeing_std
        self.sky.seeing_minimum = self.pars.seeing_minimum
        if self.sky.seeing_minimum >= self.sky.seeing_mean:
            raise ValueError('seeing_minimum must be less than seeing_mean')

        for i in range(100):
            self.sky.seeing_instance = np.random.normal(self.sky.seeing_mean, self.sky.seeing_std)
            if self.sky.seeing_instance >= self.sky.seeing_minimum:
                break
        else:
            raise RuntimeError('Could not generate a seeing instance above the minimum value')

        self.sky.atmos_psf_mode = self.pars.atmos_psf_mode
        self.sky.atmos_psf_pars = self.pars.atmos_psf_pars

        # update the PSF parameters based on the seeing
        if self.sky.atmos_psf_mode.lower().startswith('gauss'):
            self.sky.atmos_psf_pars['sigma'] = self.sky.seeing_instance / 2.355

    def make_stars(self):
        """
        Generate a star field. This will usually stay the same
        when taking a series of images at the same pointing.

        """
        self.stars = SimStars()
        self.stars.star_number = self.pars.star_number
        self.stars.star_min_flux = self.pars.star_min_flux
        self.stars.star_flux_power_law = self.pars.star_flux_power_law
        self.stars.star_position_std = self.pars.star_position_std

        self.stars.make_star_list(self.pars.imsize)

    def make_galaxies(self):
        """
        Generate a galaxy field. This will usually stay the same
        when taking a series of images at the same pointing.

        """
        self.galaxies = SimGalaxies()
        self.galaxies.galaxy_number = self.pars.galaxy_number
        self.galaxies.galaxy_min_flux = self.pars.galaxy_min_flux
        self.galaxies.galaxy_flux_power_law = self.pars.galaxy_flux_power_law
        # self.galaxies.galaxy_position_std = self.pars.galaxy_position_std
        self.galaxies.galaxy_min_width = self.pars.galaxy_min_width
        self.galaxies.galaxy_width_power_law = self.pars.galaxy_width_power_law

        self.galaxies.make_galaxy_list(self.pars.imsize)

    def add_extra_stars(self, flux=None, x=None, y=None, number=1):
        """Add one more star to the star field. This is explicitly added by the user on top of the star_number.

        Parameters
        ----------
        flux: float (optional)
            The flux of the new star. If None (default), will randomly choose a flux from the power law.
        x: float (optional)
            The x position of the new star. If None (default), will randomly choose a position.
        y: float (optional)
            The y position of the new star. If None (default), will randomly choose a position.
        number: int
            The number of stars to add. Default is 1.

        """
        if self.stars is None:
            self.make_stars()

        self.stars.add_extra_stars(self.pars.imsize, flux, x, y, number)

    def add_extra_galaxies(
            self,
            mean_x_pos=None,
            mean_y_pos=None,
            fluxes=None,
            widths=None,
            exp_scale_ratios=None,
            sersic_scale_ratios=None,
            sersic_flux_ratios=None,
            cos_is=None,
            rotations=None,
    ):
        """
        Add a galaxy (or multiple galaxies) to the list of galaxies

        Parameters
        ----------
        mean_x_pos: float, list or ndarray of floats (optional)
            The mean x position of the galaxy. If None (default), will randomly choose a position.
        mean_y_pos: float, list or ndarray of floats (optional)
            The mean y position of the galaxy. If None (default), will randomly choose a position.
        fluxes: float, list or ndarray of floats (optional)
            The flux of the exponential profile. If None (default), will randomly choose a flux.
        widths: float, list or ndarray of floats (optional)
            The width of the galaxy. If None (default), will randomly choose a width.
        exp_scale_ratios: float, list or ndarray of floats (optional)
            The ratio of the exponential scale length to the width. If None (default), will randomly choose a ratio.
        sersic_scale_ratios: float, list or ndarray of floats (optional)
            The ratio of the sersic scale length to the width. If None (default), will randomly choose a ratio.
        sersic_flux_ratios: float, list or ndarray of floats (optional)
            The flux of the sersic profile relative to the main flux. If None (default), will randomly choose a flux.
        cos_is: float , list or ndarray of floats (optional)
            The inclination angle cosine that determines how flat and narrow
            the disk of the galaxy will be. One means face-on, zero means edge-on.
            If not given, will choose a uniform cos_i in range [0.1,1].
        rotations: float , list or ndarray of floats (optional)
            The rotation angle of the galaxy, in degrees.
            Zero means the galaxy is oriented North-South, and the angle increases
            towards East. If not given, will choose a uniform rotation in range [0,360].

        """
        self.galaxies.add_extra_galaxies(
            self.pars.imsize,
            mean_x_pos,
            mean_y_pos,
            fluxes,
            widths,
            exp_scale_ratios,
            sersic_scale_ratios,
            sersic_flux_ratios,
            cos_is,
            rotations,
        )

    def make_image(self, new_sensor=False, new_camera=False, new_sky=False, new_stars=False, new_galaxies=False):
        """
        Generate a single image.
        Will add new instance of noise, and possibly shift the stars' positions
        (if given star_position_std which is non-zero).
        To simulate a new pointing in the sky (e.g., taken at a different time),
        use new_sky=True. To simulate a new pointing of a different field,
        use new_stars=True.

        In general the sensor and camera will stay the same for a given survey.

        """
        t0 = time.time()
        if new_sensor or self.sensor is None:
            self.make_sensor()
            if self.pars.show_runtimes:
                print(f'time to make sensor: {time.time() - t0:.2f}s')

        t0 = time.time()
        if new_camera or self.camera is None:
            self.make_camera()
            if self.pars.show_runtimes:
                print(f'time to make camera: {time.time() - t0:.2f}s')

        t0 = time.time()
        if new_sky or self.sky is None:
            self.make_sky()
            if self.pars.show_runtimes:
                print(f'time to make sky: {time.time() - t0:.2f}s')

        t0 = time.time()
        if new_stars or self.stars is None:
            self.make_stars()
            if self.pars.show_runtimes:
                print(f'time to make stars: {time.time() - t0:.2f}s')

        t0 = time.time()
        if new_galaxies or self.galaxies is None:
            self.make_galaxies()
            if self.pars.show_runtimes:
                print(f'time to make galaxies: {time.time() - t0:.2f}s')

        # make a PSF
        # fwhm = np.sqrt(self.camera.optic_psf_fwhm ** 2 + self.sky.seeing_instance ** 2)
        # oversample_estimate = int(np.ceil(10 / fwhm))  # allow 10 pixels across the PSF's width
        # oversample_estimate += (oversample_estimate + 1) % 2  # make sure the oversampling is an odd number
        self.camera.oversampling = self.pars.oversampling
        self.sky.oversampling = self.pars.oversampling

        # produce the atmospheric and optical PSF
        self.camera.make_optic_psf()
        self.sky.make_atmos_psf()

        self.psf = scipy.signal.convolve(self.sky.atmos_psf_image, self.camera.optic_psf_image, mode='full')
        self.psf /= np.sum(self.psf)

        # stars:

        # make sure to update this parameter before calling get_star_x_values and get_star_y_values
        self.stars.star_position_std = self.pars.star_position_std
        self.star_x = self.stars.get_star_x_values()
        self.star_y = self.stars.get_star_y_values()
        self.star_f = self.stars.get_star_flux_values()

        # galaxies:
        self.galaxy_x = self.galaxies.get_galaxy_x_values()
        self.galaxy_y = self.galaxies.get_galaxy_y_values()
        self.galaxy_f = self.galaxies.get_galaxy_flux_values()

        t0 = time.time()
        self.make_raw_star_flux_map()  # image of the flux of stars after PSF convolution (no sky, no noise)
        if self.pars.show_runtimes:
            print(f'time to make raw star flux map: {time.time() - t0:.2f}s')

        t0 = time.time()
        self.add_atmosphere()  # add the transmission and sky background to the image, with oversampling, without noise
        if self.pars.show_runtimes:
            print(f'time to add atmosphere: {time.time() - t0:.2f}s')

        t0 = time.time()
        self.add_camera()
        if self.pars.show_runtimes:
            print(f'time to add camera: {time.time() - t0:.2f}s')

        t0 = time.time()
        self.flux_to_electrons()
        if self.pars.show_runtimes:
            print(f'time to convert flux to electrons: {time.time() - t0:.2f}s')

        t0 = time.time()
        self.add_artefacts()
        if self.pars.show_runtimes:
            print(f'time to add artefacts: {time.time() - t0:.2f}s')

        t0 = time.time()
        self.electrons_to_adu()
        if self.pars.show_runtimes:
            print(f'time to convert electrons to ADU: {time.time() - t0:.2f}s')

        t0 = time.time()
        self.add_noise()
        if self.pars.show_runtimes:
            print(f'time to add noise: {time.time() - t0:.2f}s')

        # make sure to collect all the parameters used in each part
        self.save_truth()

    def make_raw_star_flux_map(self):
        """
        Take the star/galaxy positions and fluxes and place them,
        including the effects of the combined atmospheric and
        instrumental PSF, on the image plane.
        This image does not include sky transmission/background,
        noise of any kind, or the effects of the sensor.
        It will represent the raw photon number coming from the stars.

        Will calculate the oversampled instrumental, atmospheric and total PSF.
        Will calculate the flux_top image.
        """
        ovsmp = self.pars.oversampling
        imsize = self.pars.imsize
        buffer = (int(np.ceil(imsize[0] * 0.02)), int(np.ceil(imsize[1] * 0.02)))
        imsize = (imsize[0] + buffer[0] * 2, imsize[1] + buffer[1] * 2)
        imsize = (imsize[0] * ovsmp, imsize[1] * ovsmp)
        self.flux_top = np.zeros(imsize, dtype=float)

        for x, y, f in zip(self.star_x, self.star_y, self.star_f):
            x_im = round((x + buffer[1]) * ovsmp)
            y_im = round((y + buffer[0]) * ovsmp)
            if x_im < 0 or x_im >= imsize[1] or y_im < 0 or y_im >= imsize[0]:
                continue
            self.flux_top[y_im, x_im] += f

        for i, (x, y, f) in enumerate(zip(self.galaxy_x, self.galaxy_y, self.galaxy_f)):
            x_im = round((x + buffer[1]) * ovsmp)
            y_im = round((y + buffer[0]) * ovsmp)
            # uncomment these two lines (and comment the next two) to use the old method
            # self.flux_top += self.galaxies.make_galaxy_image(
            #     imsize=imsize,
            self.galaxies.add_galaxy_to_image(
                self.flux_top,
                center_x=x_im,
                center_y=y_im,
                exp_scale=self.galaxies.galaxy_exp_scale_ratios[i] * self.galaxies.galaxy_width_values[i] * ovsmp,
                sersic_scale=self.galaxies.galaxy_sersic_scale_ratios[i] * self.galaxies.galaxy_width_values[i] * ovsmp,
                exp_flux=f,  # the galaxy flux is just the flux of the disk, without any ratio modifier
                sersic_flux=f * self.galaxies.galaxy_sersic_flux_ratios[i],  # the sersic flux is much smaller
                cos_i=self.galaxies.galaxy_cos_is[i],
                rotation=self.galaxies.galaxy_rotations[i],
                cutoff_radius=self.galaxies.galaxy_width_values[i] * ovsmp * 5,
            )

        self.flux_top = scipy.signal.convolve(self.flux_top, self.psf, mode='same')

        # downsample back to pixel resolution
        if ovsmp > 1:
            # this convolution means that each new pixel is the SUM of all the pixels in the kernel
            kernel = np.ones((ovsmp, ovsmp), dtype=float)

            # correctly downsample the PSF to retain the symmetric single peak pixel
            psf_conv = scipy.signal.convolve(self.psf, kernel, mode='same')
            peak_indices = np.unravel_index(np.argmax(psf_conv), psf_conv.shape)
            offset = tuple(p % ovsmp for p in peak_indices)

            self.psf_downsampled = self.psf[offset[0]::ovsmp, offset[1]::ovsmp].copy()
            self.psf_downsampled /= np.sum(self.psf_downsampled)

            # now downsample the flux image
            self.flux_top = scipy.signal.convolve(self.flux_top, kernel, mode='same')
            self.flux_top = self.flux_top[ovsmp // 2::ovsmp, ovsmp // 2::ovsmp].copy()
            self.flux_top = self.flux_top[buffer[0]:-buffer[0], buffer[1]:-buffer[1]].copy()

        else:
            self.psf_downsampled = self.psf.copy()

    def add_atmosphere(self):
        """
        Add the effects of the atmosphere, namely the sky background and transmission.
        """
        self.flux_with_sky = self.flux_top * self.sky.transmission_instance + self.sky.background_instance

    def add_camera(self):
        """
        Add the effects of the camera, namely the vignette.
        """
        self.flux_vignette = self.flux_with_sky * self.camera.vignette_map

    def flux_to_electrons(self):
        """
        Calculate the number of electrons in each pixel,
        accounting for the total QE and the pixel QE,
        and adding the dark current.
        """
        self.electrons = self.flux_vignette * self.sensor.pixel_qe_map
        self.electrons += self.sensor.dark_current * self.pars.exposure_time

        # add saturation and bleeding:
        # TODO: add bleeding before clipping
        self.electrons[self.electrons > self.sensor.saturation_limit] = self.sensor.saturation_limit

    def add_artefacts(self):
        """
        Add artefacts like cosmic rays, satellites, etc.
        This should produce the final image.
        """

        for i in range(np.random.poisson(self.pars.cosmic_ray_number)):
            self.add_cosmic_ray(self.average_counts)  # add in place

        # TODO: satellites should be moved to the stage above atmosphere where stars and galaxies are added!
        for i in range(np.random.poisson(self.pars.satellite_number)):
            self.add_satellite(self.average_counts)  # add in place

        # add more artefacts here...

    def add_cosmic_ray(self, image):
        """
        Add a cosmic ray to the image.

        Parameters
        ----------
        image: array
            The image to add the cosmic ray to.
            This will be modified in place.
        """
        pass

    def add_satellite(self, image):
        """
        Add a satellite trail to the image.

        Parameters
        ----------
        image: array
            The image to add the satellite trail to.
            This will be modified in place.
        """
        pass

    def electrons_to_adu(self):
        """
        Convert the number of electrons in each pixel
        to the number of ADU (analog to digital units)
        that will be read out.
        """
        self.average_counts = self.electrons * self.sensor.pixel_gain_map
        self.noise_var_map = self.electrons + self.sensor.read_noise ** 2

        # this is the background noise variance, not including source noise
        self.total_bkg_var = self.sensor.read_noise ** 2
        self.total_bkg_var += self.sensor.dark_current * self.pars.exposure_time
        self.total_bkg_var += self.sky.background_instance

    def add_noise(self):
        """
        Combine the noise variance map and the counts without noise to make
        an image of the counts, including also the bias map.
        """
        # read noise is included in the variance, but should not add to the baseline (bias)
        self.image = self.sensor.pixel_bias_map - self.sensor.read_noise ** 2 + np.random.poisson(
            self.noise_var_map, size=self.pars.imsize
        )
        self.image *= self.sensor.pixel_gain_map
        self.noise_var_map *= self.sensor.pixel_gain_map ** 2
        self.total_bkg_var *= self.sensor.pixel_gain_map ** 2
        self.image = np.round(self.image).astype(int)

    def apply_bias_correction(self, image):
        """
        Apply the bias correction to an image.
        """
        return image - self.sensor.pixel_bias_map

    def apply_dark_correction(self, image):
        """
        Apply the dark current correction to an image.
        """
        return image - self.sensor.dark_current * self.pars.exposure_time

    # TODO: apply flat

    def save_truth(self):
        """
        Save the parameters from the different steps of the simulation
        into one object that can be saved with the image.

        """
        t = SimTruth()
        t.imsize = self.pars.imsize
        t.bias_mean = self.sensor.bias_mean
        t.pixel_bias_std = self.sensor.pixel_bias_std
        t.pixel_bias_map = self.sensor.pixel_bias_map

        t.gain_mean = self.sensor.gain_mean
        t.pixel_gain_std = self.sensor.pixel_gain_std
        t.pixel_gain_map = self.sensor.pixel_gain_map

        t.qe_mean = self.sensor.qe_mean
        t.pixel_qe_std = self.sensor.pixel_qe_std
        t.pixel_qe_map = self.sensor.pixel_qe_map

        t.dark_current = self.sensor.dark_current
        t.read_noise = self.sensor.read_noise
        t.saturation_limit = self.sensor.saturation_limit
        t.bleed_fraction_x = self.sensor.bleed_fraction_x
        t.bleed_fraction_y = self.sensor.bleed_fraction_y

        t.vignette_amplitude = self.camera.vignette_amplitude
        t.vignette_radius = self.camera.vignette_radius
        t.vignette_offset_x = self.camera.vignette_offset_x
        t.vignette_offset_y = self.camera.vignette_offset_y
        t.vignette_map = self.camera.vignette_map

        t.oversampling = self.camera.oversampling
        t.optic_psf_mode = self.camera.optic_psf_mode
        t.optic_psf_pars = self.camera.optic_psf_pars
        t.optic_psf_image = self.camera.optic_psf_image
        t.optic_psf_fwhm = self.camera.optic_psf_fwhm

        t.background_mean = self.sky.background_mean
        t.background_std = self.sky.background_std
        t.background_minimum = self.sky.background_minimum
        t.background_instance = self.sky.background_instance

        t.transmission_mean = self.sky.transmission_mean
        t.transmission_std = self.sky.transmission_std
        t.transmission_minimum = self.sky.transmission_minimum
        t.transmission_instance = self.sky.transmission_instance

        t.seeing_mean = self.sky.seeing_mean
        t.seeing_std = self.sky.seeing_std
        t.seeing_minimum = self.sky.seeing_minimum
        t.seeing_instance = self.sky.seeing_instance

        t.atmos_psf_mode = self.sky.atmos_psf_mode
        t.atmos_psf_pars = self.sky.atmos_psf_pars
        t.atmos_psf_image = self.sky.atmos_psf_image
        t.atmos_psf_fwhm = self.sky.atmos_psf_fwhm

        t.star_number = self.stars.star_number
        t.star_min_flux = self.stars.star_min_flux
        t.star_flux_power_law = self.stars.star_flux_power_law
        t.star_mean_fluxes = self.stars.star_mean_fluxes
        t.star_mean_x_pos = self.stars.star_mean_x_pos
        t.star_mean_y_pos = self.stars.star_mean_y_pos
        t.star_position_std = self.stars.star_position_std
        t.star_real_x_pos = self.star_x
        t.star_real_y_pos = self.star_y
        t.star_real_flux = self.star_f

        t.galaxy_number = self.galaxies.galaxy_number
        t.galaxy_min_flux = self.galaxies.galaxy_min_flux
        t.galaxy_flux_power_law = self.galaxies.galaxy_flux_power_law
        t.galaxy_mean_fluxes = self.galaxies.galaxy_mean_fluxes
        t.galaxy_mean_x_pos = self.galaxies.galaxy_mean_x_pos
        t.galaxy_mean_y_pos = self.galaxies.galaxy_mean_y_pos
        t.galaxy_position_std = self.galaxies.galaxy_position_std
        t.galaxy_min_width = self.galaxies.galaxy_min_width
        t.galaxy_width_power_law = self.galaxies.galaxy_width_power_law
        t.galaxy_real_x_pos = self.galaxy_x
        t.galaxy_real_y_pos = self.galaxy_y
        t.galaxy_real_flux = self.galaxy_f
        t.galaxy_real_width = self.galaxies.galaxy_width_values

        t.psf = self.psf
        t.psf_downsampled = self.psf_downsampled
        t.average_counts = self.average_counts
        t.noise_var_map = self.noise_var_map
        t.total_bkg_var = self.total_bkg_var

        self.truth = t


def make_gaussian(sigma_x=2.0, sigma_y=None, rotation=0.0, norm=1, imsize=None):
    """
    Create a small image of a Gaussian centered around the middle of the image.

    Parameters
    ----------
    sigma_x: float
        The sigma width parameter.
        If sigma_x and sigma_y are specified, this will be for the x-axis.
    sigma_y: float or None
        The sigma width parameter.
        If None, will use sigma_x for both axes.
    rotation: float
        The rotation angle in degrees.
        The Gaussian will be rotated counter-clockwise by this angle.
        If sigma_y is equal to sigma_x (or None) this has no effect.
    norm: int
        Normalization of the Gaussian. Choose value:
        0- do not normalize, peak will have a value of 1.0
        1- normalize so the sum of the image is equal to 1.0
        2- normalize the squares: the sqrt of the sum of squares is equal to 1.0
    imsize: int or None
        Number of pixels on a side for the output.
        If None, will automatically choose the smallest odd integer that is larger than max(sigma_x, sigma_y) * 10.

    Returns
    -------
    output: array
        A 2D array of the Gaussian.
    """
    if sigma_y is None:
        sigma_y = sigma_x

    if imsize is None:
        imsize = int(max(sigma_x, sigma_y) * 10)
        if imsize % 2 == 0:
            imsize += 1

    if norm not in [0, 1, 2]:
        raise ValueError('norm must be 0, 1, or 2')

    x = np.arange(imsize)
    y = np.arange(imsize)
    x, y = np.meshgrid(x, y)

    x0 = imsize // 2
    y0 = imsize // 2
    # TODO: what happens if imsize is even?

    x = x - x0
    y = y - y0

    rotation = rotation * np.pi / 180.0  # TODO: add option to give rotation in different units?

    x_rot = x * np.cos(rotation) - y * np.sin(rotation)
    y_rot = x * np.sin(rotation) + y * np.cos(rotation)

    output = np.exp(-0.5 * (x_rot ** 2 / sigma_x ** 2 + y_rot ** 2 / sigma_y ** 2))

    if norm == 1:
        output /= np.sum(output)
    elif norm == 2:
        output /= np.sqrt(np.sum(output ** 2))

    return output


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    s = Simulator(star_number=0, galaxy_number=100, galaxy_min_flux=1000, galaxy_min_width=2.0)
    s.pars.image_size_y = 600
    s.make_image()

    plt.imshow(s.image)
    plt.figure()
    # plt.imshow(s.camera.vignette_map)
    plt.imshow(s.image, vmin=100, vmax=1000)

    # g = SimGalaxies()
    # im = g.make_galaxy_image()
    # plt.imshow(im)
    # plt.show()
