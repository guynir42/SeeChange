
from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.utils import parse_session

from models.base import SmartSession
from models.measurements import Measurements


class ParsAssociator(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.association_radius = self.add_par(
            'association_radius',
            2.0,
            float,
            'Radius to associate different measurements with the same object (in arcsec). '
        )
        self.add_alias('association_radius', 'radius')

        self.disqualifier_thresholds = self.add_par(
            'disqualifier_thresholds',
            {
                'negatives': 0.3,
                'bad pixels': 1,
                'offsets': 5.0,
                'filter bank': 1,
            },
            dict,
            'Thresholds for disqualifying a measurement based on its disqualifier_scores. '
        )

        self.prov_list = self.add_par(
            'prov_list',
            [],
            list,
            'List of additional provenance hashes, besides the provenance upstream, ' 
            'that could be determine which Measurements object to associate. '
            'Note that hashes closer to the start of the list will have priority. '
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'associating'


class Associator:
    def __init__(self, **kwargs):
        self.pars = ParsAssociator(**kwargs)

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

    def run(self, *args, **kwargs):
        """Go over the cutouts from an image and measure all sorts of things
        for each cutout: photometry (flux, centroids), etc.

        Returns a DataStore that has the objects with the associated measurements.
        """
        # most likely to get a Measurements object or list of Measurements
        if isinstance(args[0], Measurements):
            new_args = [args[0]]  # make it a list if we got a single Measurements object for some reason
            new_args += list(args[1:])
            args = tuple(new_args)

        if isinstance(args[0], list) and all([isinstance(m, Measurements) for m in args[0]]):
            args, kwargs, session = parse_session(*args, **kwargs)
            ds = DataStore()
            ds.measurements = args[0]
            ds.cutouts = [m.cutouts for m in ds.measurements]
            ds.detections = ds.cutouts[0].sources
            ds.sub_image = ds.detections.image
            ds.image = ds.sub_image.new_image
        else:
            ds, session = DataStore.from_args(*args, **kwargs)
        self.has_recalculated = False

        # get the provenance for this step:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

        # we don't strictly need to open a session for the entire loop,
        # but I think each iteration should be fast enough to justify
        # not opening and closing it each time
        with SmartSession(session) as session:
            for m in ds.measurements:
                if self.check(m):
                    m.associate_object(prov, session=session)

    def check(self, measurements):
        """Check if the measurements pass all the quality cuts."""
        for key, value in self.pars.disqualifier_thresholds:
            if measurements.disqualifier_scores[key] >= value:  # equality is for boolean or integer cuts
                return False

        return True

