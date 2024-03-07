import numpy as np
import sqlalchemy as sa

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.utils import parse_session

from models.base import SmartSession
from models.cutouts import Cutouts
from models.measurements import Measurements
from models.enums_and_bitflags import BitFlagConverter

from improc.photometry import iterative_photometry


class ParsMeasurer(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.aperture_radii = self.add_par(
            'aperture_radii',
            [1.0, 2.0, 3.0],
            list,
            'Radius of the aperture in pixels. Can give a list of values. '
        )

        self.aperture_units = self.add_par(
            'aperture_units',
            'fwhm',
            str,
            'Units for the aperture radii. Can be pixels, arcsec, fwhm. '
            'This describes the units of the aperture_radii input, '
            'not the saved aperture values in each measurement object. '
        )

        self.annulus_radii = self.add_par(
            'annulus_radii',
            [7.5, 10.0],
            list,
            'Inner and outer radii of the annulus. '
        )

        self.annulus_units = self.add_par(
            'annulus_units',
            'pixels',
            str,
            'Units for the annulus radii. Can be pixels, arcsec, fwhm. '
            'This describes the units of the annulus_radii input. '
            'Use "pixels" to make a constant annulus, or "fwhm" to '
            'adjust the annulus size for each image based on the PSF width. '
        )

        # TODO: should we choose the "best aperture" using the config, or should each Image have its own aperture?
        self.chosen_aperture = self.add_par(
            'chosen_aperture',
            'auto',
            [str, int],
            'The aperture radius that is used for photometry. '
            'Choose either the index in the aperture_radii list, '
            'the string "psf", or the string "auto" to choose '
            'the best aperture in each image separately. '
        )

        self.analytical_cuts = self.add_par(
            'analytical_cuts',
            ['negatives', 'bad pixels', 'streak like', 'psf width'],
            [list],
            'Which kinds of analytic cuts are used to give scores to this measurement. '
        )

        self.negative_positive_threshold = self.add_par(
            'negative_positive_threshold',
            3.0,
            float,
            'How many times the local background RMS for each pixel counts '
            'as being a negative or positive pixel. '
        )

        self.bad_pixel_radius = self.add_par(
            'bad_pixel_radius',
            3.0,
            float,
            'Radius in pixels for the bad pixel cut. '
        )

        self.bad_pixel_exclude = self.add_par(
            'bad_pixel_exclude',
            [],
            list,
            'List of strings of the bad pixel types to exclude from the bad pixel cut. '
            'The same types are ignored when running photometry. '
        )

        self.streak_filter_angle_step = self.add_par(
            'streak_filter_angle_step',
            5.0,
            float,
            'Step in degrees for the streaks filter bank. '
        )

        self.width_filter_multipliers = self.add_par(
            'width_filter_multipliers',
            [0.5, 2.0, 5.0, 10.0],
            list,
            'Multipliers of the PSF width to use as matched filter templates'
            'to compare against the real width (x1.0) when running psf width filter. '
        )

        self.external_analysis_versions = self.add_par(
            'external_analysis_versions',
            {},  # TODO: add "real-bogus" once we have something for that.
            [dict],
            'A dictionary where each key is the name of an external software that is run on the cutouts, '
            'e.g., a "real-bogus" classifier, and the value is a string with the version of the software. ',
        )

        self.thresholds = self.add_par(
            'thresholds',
            {'negatives': 0.3, 'streak like': 1.0, 'bad pixels': 1.0, 'psf width': 1.0},
            dict,
            'A dictionary where each key is the name of an analytic cut or external analysis, '
            'and the value is the threshold for the score of that cut. '
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'measuring'


class Measurer:
    def __init__(self, **kwargs):
        self.pars = ParsMeasurer(**kwargs)

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

    def run(self, *args, **kwargs):
        """
        Go over the cutouts from an image and measure all sorts of things
        for each cutout: photometry (flux, centroids), real/bogus, etc.

        Returns a DataStore object with the products of the processing.
        """
        # most likely to get a Cutouts object or list of Cutouts
        if isinstance(args[0], Cutouts):
            args[0] = [args[0]]  # make it a list if we got a single Cutouts object for some reason

        if isinstance(args[0], list) and all([isinstance(c, Cutouts) for c in args[0]]):
            args, kwargs, session = parse_session(*args, **kwargs)
            ds = DataStore()
            ds.cutouts = args[0]
            ds.detections = args[0][0].sources
            ds.sub_image = ds.detections.image
            ds.image = ds.sub_image.new_image
        else:
            ds, session = DataStore.from_args(*args, **kwargs)
        self.has_recalculated = False

        # get the provenance for this step:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

        # try to find some measurements in memory or in the database:
        measurements_list = ds.get_measurements(prov, session=session)

        if measurements_list is None or len(measurements_list) == 0:  # must create a new list of Measurements
            self.has_recalculated = True
            # use the latest source list in the data store,
            # or load using the provenance given in the
            # data store's upstream_provs, or just use
            # the most recent provenance for "detection"
            detections = ds.get_detections(session=session)

            if detections is None:
                raise ValueError(f'Cannot find a source list corresponding to the datastore inputs: {ds.get_inputs()}')

            cutouts = ds.get_cutouts(session=session)

            measurements_list = []
            for c in ds.cutouts:
                m = Measurements(cutouts=c)

                # TODO: implement all sorts of analysis and photometry
                m.set_apertures(self.pars.aperture_radii, self.pars.aperture_units)

                ignore_bits = 0
                for badness in self.pars.bad_pixel_exclude:
                    ignore_bits |= BitFlagConverter.convert(badness)

                flags = c.sub_flags ^ ignore_bits  # remove the bad pixels that we want to ignore

                # TODO: consider if there are any additional parameters that photometry needs
                output = iterative_photometry(c.sub_data, c.sub_weight, flags, m.psf, radii=m.aper_radii)
                m.flux_psf = output['psf_flux']
                m.flux_psf_err = output['psf_err']
                m.area_psf = output['psf_area']
                m.flux_apertures = output['fluxes']
                m.flux_apertures_err = np.sqrt(output['variance'] * output['areas'])  # TODO: add source noise??
                m.aper_radii = output['radii']
                m.area_apertures = output['areas']
                m.background = output['background']
                m.background_rms = np.sqrt(output['variance'])
                m.offset_x = output['offset_x']
                m.offset_y = output['offset_y']
                m.width = (output['major'] + output['minor']) / 2
                m.elongation = output['elongation']
                m.position_angle = output['position_angle']

                m.calculate_magnitudes()  # use the fluxes to calculate magnitudes

                if m.provenance is None:
                    m.provenance = prov
                else:
                    if m.provenance.id != prov.id:
                        raise ValueError('Provenance mismatch for measurements and provenance!')

                measurements_list.append(m)

            # TODO: implement the actual code to do this.
            #  For each source in the SourceList make a Cutouts object.
            #  For each Cutouts calculate the photometry (flux, centroids).
            #  Apply analytic cuts to each stamp image, to rule out artefacts.
            #  Apply deep learning (real/bogus) to each stamp image, to rule out artefacts.
            #  Save the results as Measurement objects, append them to the Cutouts objects.
            #  Commit the results to the database.

            # add the resulting list to the data store
            ds.measurements = measurements_list

        # make sure this is returned to be used in the next step
        return ds

