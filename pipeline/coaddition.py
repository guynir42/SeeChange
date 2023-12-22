
import numpy as np
from sep import Background

from models.provenance import Provenance
from models.image import Image

from pipeline.parameters import Parameters

from improc.bitmask_tools import dilate_bitmask
from improc.inpainting import Inpainter
from improc.tools import sigma_clipping


class ParsCoadd(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.method = self.add_par(
            'method',
            'zogy',
            str,
            'Coaddition method.  Currently only "naive" and "zogy" are supported. ',
            critical=True
        )

        self.alignment = self.add_par(
            'alignment',
            {},
            dict,
            'Alignment parameters. ',
            critical=True
        )

        self.inpainting = self.add_par(
            'inpainting',
            {},
            dict,
            'Inpainting parameters. ',
            critical=True
        )

        self.noise_estimator = self.add_par(
            'noise_estimator',
            'sep',
            str,
            'Method to estimate noise (sigma) in the image.  '
            'Use "sep" or "sigma_clipping" or "bkg_rms", '
            'which will rely on the Image object to have a bkg_rms_estimate. ',
            critical=True,
        )

        self.enforce_no_new_attrs = True
        self.override( kwargs )

    def get_process_name(self):
        return 'coaddition'


class Coadder:
    """Use this class to coadd (stack) images together to make a deeper image.

    Each image should have a PSF and a ZeroPoint associated with it (and loaded!) when running coaddition.

    Images are expected to be aligned (use the Aligner) and should generally be of the same sky region.
    If not already aligned, they need to have the SourceList and WorldCoordinates loaded so that
    alignment can be done on the fly.

    Input images should also have a valid Provenance associated with them. Not that for any set of images
    that share a provenance hash, their respective downstreams (e.g., PSF) should also have a single provenance
    hash for each type of downstream.  This makes it possible to identify the upstream images' associated products
    based solely on the provenance's upstream hashes.

    Areas on the edges where the images are not overlapping (or bad pixels, etc.) are coadded but will
    contribute zero weight, so the total weight of that pixel will be zero (if all input images have a bad pixel)
    or they would have lower weight if only some images had bad pixels there.

    Remember that coaddition uses convolution with the PSF, so any effects of individual pixels could affect nearby
    pixels, depending on the size of the PSF.
    """

    def __init__( self, **kwargs ):
        self.pars = ParsCoadd(**kwargs)
        self.inpainter = Inpainter(**self.pars.inpainting)
        # the aligner object is created in the image object

    def estimate_noise(self, image):
        """Get the noise RMS of the background of the given image.

        Parameters
        ----------
        image: Image
            The image for which background should be estimated.

        Returns
        -------
        sigma: float
            The RMS of the background in the image.
        """
        if image is None or image.data is None:
            raise ValueError('The image must be loaded before estimating the noise. ')

        if self.pars.noise_estimator == 'sep':
            bkg = Background(image.data)
            sigma = bkg.globalrms
        elif self.pars.noise_estimator == 'sigma_clipping':
            _, sigma = sigma_clipping(image.data)
        elif self.pars.noise_estimator == 'bkg_rms':
            if image.bkg_rms_estimate is None:
                raise ValueError('The image must have a bkg_rms_estimate before estimating the noise. ')
            sigma = image.bkg_rms_estimate
        else:
            raise ValueError(
                f'Unknown noise estimator: {self.pars.noise_estimator}.  Use "sep" or "sigma_clipping" or "bkg_rms". '
            )

        return sigma

    def _coadd_naive(self, aligned_images):
        """Simply sum the values in each image on top of each other.

        Parameters
        ----------
        aligned_images: list of Image
            Images that have been aligned to each other.
            Each image must also have a PSF object attached.

        Returns
        -------
        outim: ndarray
            The image data after coaddition.
        outwt: ndarray
            The weight image after coaddition.
        outfl: ndarray
            The bit flags array after coaddition.
        """
        imcube = np.array([image.data for image in aligned_images])
        outim = np.sum(imcube, axis=0)
        wtcube = np.array([image.weight for image in aligned_images])
        outwt = np.sum(wtcube, axis=0)

        outfl = np.zeros(outim.shape, dtype='uint16')
        for image in aligned_images:
            outfl |= image.flags

        return outim, outwt, outfl

    def _coadd_zogy(self, aligned_images):
        """Use Zackay & Ofek proper image coaddition to add the images together.

        This method uses the PSF of each image to

        Parameters
        ----------
        aligned_images: list of Image
            Images that have been aligned to each other.
            Each image must also have a PSF object attached.

        Returns
        -------
        outim: ndarray
            The image data after coaddition.
        outwt: ndarray
            The weight image after coaddition.
        outfl: ndarray
            The bit flags array after coaddition.
        psf: ndarray
            An array with the PSF of the output image.
        score: ndarray
            A matched-filtered score image of the coadded image.
        """

        images = []
        psfs = []
        fwhms = []
        flags = []
        weights = []
        flux_zps = []
        sigmas = []
        for image in aligned_images:
            images.append(image.data)
            flags.append(image.flags)
            weights.append(image.weight)
            psf_clip = image.psf.get_clip()
            padsize_x = (image.data.shape[1] - psf_clip.shape[1]) // 2
            padsize_y = (image.data.shape[0] - psf_clip.shape[0]) // 2
            psf_pad = np.pad(psf_clip, ((padsize_y, padsize_y), (padsize_x, padsize_x)))
            psfs.append(psf_pad)
            fwhms.append(image.psf.fwhm_pixels)
            flux_zps.append( 10 ** (0.4 * image.zp.zp) )
            sigmas.append(self.estimate_noise(image))

        imcube = np.array(images)
        flcube = np.array(flags)
        wtcube = np.array(weights)
        psfcube = np.array(psfs)
        sigmas = np.reshape(np.array(sigmas), (1, 1, len(sigmas)))
        flux_zps = np.reshape(np.array(flux_zps), (1, 1, len(flux_zps)))

        # make sure to inpaint missing data
        imcube = self.inpainter.run(imcube, flcube, wtcube)

        if np.sum(np.isnan(imcube)) > 0:
            raise ValueError('There are still NaNs in the image data after inpainting!')

        # This is where the magic happens
        imcube_f = np.fft.fft2(imcube)
        psfcube_f = np.fft.fft2(psfcube)
        score_f = np.sum(flux_zps / sigmas ** 2 * np.conj(psfcube_f) * imcube_f, axis=0)  # eq 7
        psf_f = np.sum(flux_zps ** 2 / sigmas ** 2 * np.abs(psfcube_f) ** 2, axis=0)  # eq 10
        outim_f = score_f / psf_f  # eq 8

        outim = np.fft.ifft2(outim_f).real
        score = np.fft.ifft2(score_f).real
        psf = np.fft.ifft2(psf_f).real
        psf = psf / np.sum(psf)

        outfl = np.zeros(outim.shape, dtype='uint16')
        for image in aligned_images:
            splash_pixels = 2  # TODO: should adjust by the PSF FWHM
            outfl |= dilate_bitmask(image.flags, iterations=splash_pixels)

        # TODO: is there a better way to get the weight image?
        outwt = 1 / np.sqrt(np.abs(outim))

        return outim, outwt, outfl, psf, score

    def run(self, images):
        """Run coaddition on the given list of images, and return the coadded image.

        The images should have at least a set of SourceList and WorldCoordinates loaded, so they can be aligned.
        The images must also have a PSF and ZeroPoint loaded for the coaddition process.

        Parameters
        ----------
        images: list of Image objects


        Returns
        -------
        output: Image object
            The coadded image.
        """
        images.sort(key=lambda image: image.mjd)
        if self.pars.alignment['to_index'] == 'last':
            index = len(images) - 1
        elif self.pars.alignment['to_index'] == 'first':
            index = 0
        else:  # TODO: consider allowing a specific index as integer?
            raise ValueError(f"Unknown alignment reference index: {self.pars.alignment['to_index']}")
        output = Image.from_images(images, index=index)
        output.provenance = Provenance(
            code_version=images[0].provenance.code_version,
            parameters=self.pars.get_critical_pars(),
            upstreams=output.get_upstream_provenances(),
            process='coaddition',
        )
        output.is_coadd = True
        output.new_image = None

        if self.pars.method == 'naive':
            self._coadd_naive(output)
        elif self.pars.method == 'zogy':
            self._coadd_zogy(output)
        else:
            raise ValueError(f'Unknown coaddition method: {self.pars.method}. Use "naive" or "zogy".')

        return output