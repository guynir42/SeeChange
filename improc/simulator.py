
import numpy as np
import scipy

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
            'Square this and use it as Poisson variance for bias values'
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
            'pixel_qe_std', 0.0, float,
            'Standard deviation of the pixel quantum efficiency (around value of 1.0)'
        )
        self.gain_mean = self.add_par('gain_mean', 1.0, (int, float), 'Mean gain')
        self.gain_std = self.add_par('gain_std', 0.0, (int, float), 'Gain variation between pixels')

        # camera parameters
        self.vignette_amplitude = self.add_par('vignette_amplitude', 0.0, float, 'Vignette amplitude')
        self.vignette_inner_radius = self.add_par(
            'vignette_inner_radius', 0.0, float,
            'Inside this radius the vignette is ignored'
        )
        self.vignette_offset_x = self.add_par('vignette_offset_x', 0.0, float, 'Vignette offset in x')
        self.vignette_offset_y = self.add_par('vignette_offset_y', 0.0, float, 'Vignette offset in y')

        self.optic_psf_mode = self.add_par('optic_psf_mode', 'gaussian', str, 'Optical PSF mode')
        self.optic_psf_pars = self.add_par('optic_psf_pars', {'sigma': 1.0}, dict, 'Optical PSF parameters')

        # sky parameters
        self.background_mean = self.add_par('background_mean', 20.0, (int, float), 'Mean background level')
        self.background_std = self.add_par(
            'background_std', 1.0, (int, float),
            'Variation of background level between different sky instances'
        )
        self.transmission_mean = self.add_par('transmission_mean', 1.0, (int, float), 'Mean transmission (zero point)')
        self.transmission_std = self.add_par(
            'transmission_std', 0.1, (int, float),
            'Variation of transmission (zero point) between different sky instances'
        )
        self.seeing_mean = self.add_par('seeing_mean', 1.0, (int, float), 'Mean seeing')
        self.seeing_std = self.add_par('seeing_std', 0.0, (int, float), 'Seeing variation between images')
        self.atmos_psf_mode = self.add_par('atmos_psf_mode', 'gaussian', str, 'Atmospheric PSF mode')
        self.atmos_psf_pars = self.add_par('atmos_psf_pars', {'sigma': 1.0}, dict, 'Atmospheric PSF parameters')

        self.star_number = self.add_par('star_number', 100, int, 'Number of stars (on average) to simulate')
        self.star_flux_power_law = self.add_par(
            'star_flux_power_law', -1.0, float,
            'Power law index for the flux distribution of stars'
        )
        self.star_position_std = self.add_par(
            'star_position_std', 0.0, float,
            'Standard deviation of the position of stars between images (in both x and y)'
        )

        self.cosmic_ray_number = self.add_par('cosmic_ray_number', 0, int, 'Average number of cosmic rays per image')

        # lock this object, so it can't be accidentally given the wrong name
        self._enforce_no_new_attrs = True

        self.override(kwargs)


class SimTruth:
    """
    Contains the truth values for a simulated image.
    This object should be generated for each image,
    and saved along with it, to compare the analysis
    results to the ground truth values.
    """
    def __init__(self):
        # things involving the image sensor
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
        self.vignette_map = None  # how much light is lost along the edges of the image
        self.flat_field_total = None  # combination of the vignette and pixel QE variations

        self.optic_psf_mode = None  # e.g., 'gaussian'
        self.optic_psf_pars = None  # a dict with the parameters used to make this PSF

        # things involving the sky
        self.background_mean = None  # mean sky+dark current across image
        self.background_std = None  # variation between images
        self.background_instance = None  # the background for this specific image's sky

        self.transmission_mean = None  # average sky transmission
        self.transmission_std = None  # variation in transmission between images
        self.transmission_instance = None  # the transmission for this specific image's sky

        self.seeing_mean = None  # the average seeing in this survey
        self.seeing_std = None  # the variation in seeing between images
        self.seeing_instance = None  # the seeing for this specific image's sky

        self.atmos_psf_mode = None  # e.g., lorentzian
        self.atmos_psf_pars = None  # a dict with the parameters used to make this PSF
        self.total_fwhm = None  # e.g., the seeing
        self.psf_image = None  # the final shape of the PSF for this image

        # things involving the specific set of objects in the sky
        self.star_mean_fluxes = None  # for each star, the mean flux (add source noise based on gain)
        self.star_mean_x_pos = None  # for each star, the mean x position
        self.star_mean_y_pos = None  # for each star, the mean y position
        self.star_position_std = None  # for each star, the variation in position (in both x and y)

        # additional random things that are unique to each image
        self.cosmic_ray_x_pos = None  # where each cosmic ray was
        self.cosmic_ray_y_pos = None  # where each cosmic ray was

        # TODO: add satellite trails


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
        self.vignette_inner_radius = None  # inside this radius the vignette is ignored
        self.vignette_offset_x = None  # vignette offset in x
        self.vignette_offset_y = None  # vignette offset in y
        self.vignette_map = None  # e.g., vignette

        self.oversampling = None  # how much do we need the PSF to be oversampled?
        self.optic_psf_mode = None  # e.g., 'gaussian'
        self.optic_psf_pars = None  # a dict with the parameters used to make this PSF
        self.optic_psf_image = None  # the final shape of the PSF for this image
        self.optic_psf_fwhm = None  # e.g., the total effect of optical aberrations on the width

    def make_vignette(self):
        """
        Calculate the vignette across the sensor.
        Uses the amplitude, inner radius and offsets
        to calculate the flat_field_map transmission map.
        """
        pass

    def make_optic_psf(self, oversampling):
        """
        Make the optical PSF.
        Uses the optic_psf_mode to generate the PSF map
        and calculate the width (FWHM).

        Saves the results into optic_psf_image and optic_psf_fwhm.

        """
        self.oversampling = oversampling

        if self.optic_psf_mode.lower() == 'gaussian':
            self.optic_psf_image = make_gaussian(self.optic_psf_pars['sigma'] * self.oversampling)
            self.optic_psf_fwhm = self.optic_psf_pars['sigma'] * 2.355
        else:
            raise ValueError(f'PSF mode not recognized: {self.optic_psf_mode}')

    def make_vignette(self, imsize_x, imsize_y):
        """
        Make an image of the vignette part of the flat field.
        Input the imsize_x and imsize_y of the image.
        """

        self.vignette_map = np.ones((imsize_x, imsize_y))

        # TODO: continue this!


class SimSky:
    """
    Container for the properties of a simulated sky.
    """
    def __init__(self):
        self.background_mean = None  # mean sky+dark current across image
        self.background_std = None  # variation between images
        self.background_instance = None  # the background for this specific image's sky

        self.transmission_mean = None  # average sky transmission
        self.transmission_std = None  # variation in transmission between images
        self.transmission_instance = None  # the transmission for this specific image's sky

        self.seeing_mean = None  # the average seeing in this survey
        self.seeing_std = None  # the variation in seeing between images
        self.seeing_instance = None  # the seeing for this specific image's sky

        self.oversampling = None  # how much do we need the PSF to be oversampled?
        self.atmos_psf_mode = None  # e.g., 'gaussian'
        self.atmos_psf_pars = None  # a dict with the parameters used to make this PSF
        self.atmos_psf_image = None  # the final shape of the PSF for this image
        self.atmos_psf_fwhm = None  # e.g., the seeing

    def make_atmos_psf(self, oversampling):
        """
        Use the psf mode, pars and the seeing_instance
        to produce an atmospheric PSF.
        This PSF is used to convolve the optical PSF to get the total PSF.

        """
        self.oversampling = oversampling

        if self.atmos_psf_mode.lower() == 'gaussian':
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
        self.star_flux_power_law = None  # power law index of the flux of stars
        self.star_mean_fluxes = None  # for each star, the mean flux (in photons per total exposure time)
        self.star_mean_x_pos = None  # for each star, the mean x position
        self.star_mean_y_pos = None  # for each star, the mean y position
        self.star_position_std = None  # for each star, the variation in position (in both x and y)

    def make_star_field(self, imsize_x, imsize_y):
        """
        Make a field of stars.
        Uses the power law to draw random mean fluxes,
        and uniform positions for each star on the sensor.

        """
        rng = np.random.default_rng()
        max_flux = 1e6
        alpha = abs(self.star_flux_power_law) + 1
        self.star_mean_fluxes = max_flux / (1 / max_flux + rng.power(alpha, self.star_number))
        self.star_mean_x_pos = rng.uniform(0, 1, self.star_number) * imsize_x
        self.star_mean_y_pos = rng.uniform(0, 1, self.star_number) * imsize_y

    def get_star_x_values(self):
        """
        Return the positions of the stars (in pixel coordinates)
        after possibly applying small astrometric shifts (e.g., scintillations)
        to the mean star positions.

        """
        x = self.star_mean_x_pos
        if self.star_position_std is not None:
            x += np.random.normal(0, self.star_position_std, self.star_number)

        return x

    def get_star_y_values(self):
        """
        Return the positions of the stars (in pixel coordinates)
        after possibly applying small astrometric shifts (e.g., scintillations)
        to the mean star positions.

        """
        y = self.star_mean_y_pos
        if self.star_position_std is not None:
            y += np.random.normal(0, self.star_position_std, self.star_number)

        return y

    def get_star_flux_values(self):
        """
        Return the fluxes of the stars (in photons per total exposure time)
        after possibly applying a flux change due to e.g., occultations/flares,
        or due to scintillation noise.
        (TODO: this is not yet implemented!)

        """
        return self.star_mean_fluxes


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

        # intermediate variables
        # fluxes coming from the stars
        self.star_x = None
        self.star_y = None
        self.star_f = None

        self.psf = None  # we are cheating because this includes both the optical and atmospheric PSFs
        self.oversampling = None  # how much oversampling (integer) do we need to express fluxes?
        self.flux_top = None  # this is the mean number of photons hitting the top of the atmosphere

        # adding the sky into the mix
        self.flux_with_sky = None  # this is mean photons after passing through the atmosphere, adding the background

        # adding the effect of the camera optics
        self.flux_vignette = None  # average number of photons after passing through the aperture vignette

        # now the photons are absorbed into the sensor pixels
        self.electrons = None  # average number of electrons in each pixel, considering QE

        self.counts_without_noise = None  # the final counts, not including noise
        self.noise_var_map = None  # the total variance from read, dark, sky b/g, and source noise
        self.counts = None  # this is the final counts from everything except cosmic rays/satellites/artefacts

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
        self.sensor.pixel_bias_map = self.sensor.bias_mean + np.random.poisson(self.sensor.pixel_bias_std ** 2)

        self.sensor.gain_mean = self.pars.gain_mean
        self.sensor.gain_std = self.pars.gain_std
        self.sensor.gain_total = np.random.normal(self.sensor.gain_mean, self.sensor.gain_std)

        self.sensor.pixel_qe_std = self.pars.pixel_qe_std
        self.sensor.pixel_qe_total = np.random.normal(1.0, self.sensor.pixel_qe_std)

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
        self.camera.vignette_inner_radius = self.pars.vignette_inner_radius
        self.camera.vignette_offset_x = self.pars.vignette_offset_x
        self.camera.vignette_offset_y = self.pars.vignette_offset_y

        imsize_x = self.pars.image_size_x
        imsize_y = self.pars.image_size_y if self.pars.image_size_y is not None else imsize_x
        self.camera.make_vignette(imsize_x, imsize_y)

        self.camera.optic_psf_mode = self.pars.optic_psf_mode
        self.camera.optic_psf_pars = self.pars.optic_psf_pars

        oversampling = 10
        self.camera.make_optic_psf(oversampling)  # use a large oversampling just to get the FWHM

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
        self.sky.background_instance = np.random.normal(self.sky.background_mean, self.sky.background_std)

        self.sky.transmission_mean = self.pars.transmission_mean
        self.sky.transmission_std = self.pars.transmission_std
        self.sky.transmission_instance = np.random.normal(self.sky.transmission_mean, self.sky.transmission_std)

        self.sky.seeing_mean = self.pars.seeing_mean
        self.sky.seeing_std = self.pars.seeing_std
        self.sky.seeing_instance = np.random.normal(self.sky.seeing_mean, self.sky.seeing_std)

        self.sky.atmos_psf_mode = self.pars.atmos_psf_mode
        self.sky.atmos_psf_pars = self.pars.atmos_psf_pars

    def make_stars(self):
        """
        Generate a star field. This will usually stay the same
        when taking a series of images at the same pointing.

        """
        self.stars = SimStars()
        self.stars.star_number = self.pars.star_number
        self.stars.star_flux_power_law = self.pars.star_flux_power_law
        self.stars.star_position_std = self.pars.star_position_std

        imsize_x = self.pars.image_size_x
        imsize_y = self.pars.image_size_y if self.pars.image_size_y is not None else imsize_x
        self.stars.make_star_field(imsize_x, imsize_y)

    def make_image(self, new_sensor=False, new_camera=False, new_sky=False, new_stars=False):
        """
        Generate a single image.
        Will add new instance of noise, and possibly shift the stars' positions
        (if given star_position_std which is non-zero).
        To simulate a new pointing in the sky (e.g., taken at a different time),
        use new_sky=True. To simulate a new pointing of a different field,
        use new_stars=True.

        In general the sensor and camera will stay the same for a given survey.

        """
        if new_sensor or self.sensor is None:
            self.make_sensor()

        if new_camera or self.camera is None:
            self.make_camera()

        if new_sky or self.sky is None:
            self.make_sky()

        if new_stars or self.stars is None:
            self.make_stars()

        # stars:
        self.star_x = self.stars.get_star_x_values()
        self.star_y = self.stars.get_star_y_values()
        self.star_f = self.stars.get_star_flux_values()

        self.make_raw_star_flux_map()  # image of the flux of stars after PSF convolution (no sky, no noise)
        self.add_atmosphere()  # add the transmission and sky background to the image, with oversampling, without noise

        self.add_camera()

        self.flux_to_electrons()

        self.electrons_to_adu()

        self.add_noise()

        self.add_artefacts()

        # make sure to collect all the parameters used in each part
        self.save_truth()

    def make_raw_star_flux_map(self):
        """
        Take the star positions and fluxes and place a star,
        including the combined atmospheric and instrumental PSF,
        on the image plane.
        This image does not include sky transmission/background,
        noise of any kind, or the effects of the sensor.
        It will represent the raw photon number coming from the stars.

        Will calculate the oversampled instrumental, atmospheric and total PSF.
        Will calculate the flux_top image.
        """
        fwhm = np.sqrt(self.camera.optic_psf_fwhm ** 2 + self.sky.seeing_instance ** 2)
        oversample_estimate = int(np.ceil(10 / fwhm))  # allow 10 pixels across the PSF's width
        oversampling = min(1, oversample_estimate)
        self.oversampling = oversampling

        self.camera.make_optic_psf(oversampling)
        self.sky.make_atmos_psf(oversampling)

        self.psf = scipy.signal.convolve2d(self.sky.atmos_psf_image, self.camera.optic_psf_image, mode='full')

        if self.pars.image_size_y is None:
            imsize = (self.pars.image_size_x, self.pars.image_size_x)
        else:
            imsize = (self.pars.image_size_x, self.pars.image_size_y)
        imsize = (imsize[0] * self.oversampling, imsize[1] * self.oversampling)

        self.flux_top = np.zeros(imsize, dtype=float)

        for x, y, f in zip(self.star_x, self.star_y, self.star_f):
            self.flux_top[round(y * self.oversampling)][round(x * self.oversampling)] += f

        self.flux_top = scipy.signal.convolve2d(self.flux_top, self.psf, mode='same')

        # downsample back to pixel resolution
        if oversampling > 1:
            # this convolution means that each new pixel is the SUM of all the pixels in the kernel
            kernel = np.ones((oversampling, oversampling), dtype=float)
            self.flux_top = scipy.signal.convolve2d(self.flux_top, kernel, mode='same')
            self.flux_top = self.flux_top[oversampling//2::oversampling, oversampling//2::oversampling]

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

    def electrons_to_adu(self):
        """
        Convert the number of electrons in each pixel
        to the number of ADU (analog to digital units)
        that will be read out.
        """
        self.counts_without_noise = self.electrons / self.sensor.gain_map
        self.noise_var_map = (self.electrons + self.sensor.read_noise ** 2) / self.sensor.gain_map

    def add_noise(self):
        """
        Combine the noise variance map and the counts without noise to make
        an image of the counts, including also the bias map.
        """
        self.counts = self.sensor.pixel_bias_map + self.counts_without_noise + np.random.poisson(self.noise_var_map)

    def add_artefacts(self):
        """
        Add artefacts like cosmic rays, satellites, etc.
        This should produce the final image.
        """
        self.image = np.copy(self.counts)
        for i in range(np.random.poisson(self.pars.cosmic_ray_number)):
            self.add_cosmic_ray(self.image)  # add in place

        for i in range(np.random.posson(self.pars.satellite_number)):
            self.add_satellite(self.image)  # add in place

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

    def save_truth(self):
        """
        Save the parameters from the different steps of the simulation
        into one object that can be saved with the image.

        """
        pass


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
    s = Simulator()

