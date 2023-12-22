
import numpy as np
from scipy.signal import correlate


from models.provenance import Provenance
from models.image import Image

from pipeline.parameters import Parameters

from improc.bitmask_tools import dilate_bitmask
from improc.inpainting import Inpainter


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

    def _coadd_naive(self, output):
        """Simply sum the values in each image on top of each other.

        Parameters
        ----------
        output: Image
            This image is the one that will be returned from the coaddition process.
            Must contain a list of upstream_images from which the aligned_images can
            be calculated.
        """
        imcube = np.array([image.data for image in output.aligned_images])
        output.data = np.sum(imcube, axis=0)
        wtcube = np.array([image.weight for image in output.aligned_images])
        output.weight = np.sum(wtcube, axis=0)

        flags = np.zeros(output.data.shape, dtype='uint16')
        for image in output.aligned_images:
            flags |= image.flags
        output.flags = flags

    def _coadd_zogy(self, output):
        """Use Zackay & Ofek proper image coaddition to add the images together.

        This method uses the PSF of each image to

        Parameters
        ----------
        output: Image
            This image is the one that will be returned from the coaddition process.
            Must contain a list of upstream_images from which the aligned_images can
            be calculated.

        """
        imcube = np.array([image.data for image in output.aligned_images])
        psfs = []
        for image in output.upstream_images:
            psf_clip = image.psf.get_clip()
            padsize_x = (imcube.shape[2] - psf_clip.shape[1]) // 2
            padsize_y = (imcube.shape[1] - psf_clip.shape[0]) // 2
            psf_pad = np.pad(psf_clip, ((padsize_y, padsize_y), (padsize_x, padsize_x)))
            psfs.append(psf_pad)
        psfcube = np.array(psfs)

        flags = np.zeros(output.data.shape, dtype='uint16')
        for image in output.aligned_images:
            splash_pixels = 2  # TODO: should adjust by the PSF FWHM
            flags |= dilate_bitmask(image.flags, iterations=splash_pixels)
        output.flags = flags

        # TODO: is there a better way to get the weight image?
        wt = 0

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