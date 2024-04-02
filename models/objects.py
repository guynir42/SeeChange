import numpy as np
from collections import defaultdict

import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base, SeeChangeBase, SmartSession, AutoIDMixin, SpatiallyIndexed
from models.provenance import Provenance
from models.measurements import Measurements, measurements_object_association_table


class Object(Base, AutoIDMixin, SpatiallyIndexed):
    __tablename__ = 'objects'

    name = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc='Name of the object (can be internal nomenclature or external designation, e.g., "SN2017abc")'
    )

    discovery_ra = sa.Column(
        sa.Float,
        nullable=False,
        index=True,
        doc='Right ascension of the object when it was first detected, in degrees'
    )

    discovery_dec = sa.Column(
        sa.Float,
        nullable=False,
        index=True,
        doc='Declination of the object when it was first detected, in degrees'
    )

    measurements = orm.relationship(
        'Measurement',
        back_populates='object',
        cascade='all, delete-orphan',
        passive_deletes=True,
        lazy='selectin',
        doc='Measurements of the object'
    )

    def __init__(self, **kwargs):
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.calculate_coordinates()

    def __setattr__(self, key, value):
        if key == 'ra' and self.discovery_ra is None:
            self.discovery_ra = value

        if key == 'dec' and self.discovery_dec is None:
            self.discovery_dec = value

        super().__setattr__(key, value)

    def associate_measurements(self, associator, new_measurement=None, session=None):
        """Associate any Measurements objects that are close enough to this object's discovery coordinates.

        Will use the associator object to determine the radius for the search,
        and also to disqualify bad measurements.

        If there is more than one measurements object within the search radius and with the same MJD,
        it will choose the best one according to the following priority:
        1) Measurements object has a provenance that matches this Object's provenance.upstreams.
        2) Measurements object's provenance is on the prov_list of this Object's provenance.parameters
           (the first items on the list have higher priority than the later ones).
        3) Measurements object with the most recently created provenance is chosen.

        Once a set of measurements is determined it will also update the object's ra/dec, which are the
        mean coordinates of all associated measurements.

        Parameters
        ----------
        associator: pipeline.associating.Associator object
            The associator object that will be used to determine the radius for the search
            and the disqualifier thresholds.
        new_measurement: models.measurements.Measurement, optional
            The measurement to add to the list of already committed measurements.
            Could be a new object that hasn't been added to the DB or an existing one.
        session: sqlalchemy.orm.session.Session, optional
            The session to use for the database queries.
            If not given will use the default session,
            and close it at the end of the method.
        """
        # this includes all measurements that are close to the discovery measurement
        stmt = sa.select(Measurements).join(Provenance).where(
            Measurements.cone_search(
                self.ra,
                self.dec,
                associator.pars.radius,
            )
        )
        stmt = stmt.where(Provenance.is_bad.is_(False))
        if new_measurement is not None and new_measurement.id is not None:
            stmt = stmt.where(Measurements.id != new_measurement.id)  # do not include new_measurement

        measurements = session.scalars(stmt).all()
        if new_measurement is not None:  # now add it to the list
            measurements = [new_measurement] + measurements

        measurements = [m for m in measurements if associator.check(m)]

        measurements.sort(key=lambda m: m.mjd)

        # group measurements into a dictionary by their MJD
        measurements_per_mjd = defaultdict(list)
        for m in measurements:
            measurements_per_mjd[m.mjd].append(m)

        for mjd, m_list in measurements_per_mjd.items():
            prov_list = self.provenance.parameters['prov_list'].copy()
            prov_list = [self.provenance.upstreams[0].id] + prov_list  # prepend the upstream provenance

            # check if a measurement matches one of the provenance hashes, ideally it matches the upstream
            for hash in prov_list:
                best_m = [m for m in m_list if m.provenance.id == hash]
                if len(best_m) > 1:
                    raise ValueError('More than one measurement with the same provenance. ')
                if len(best_m) == 1:
                    measurements_per_mjd[mjd] = best_m[0]  # replace a list with a single Measurements object
                    break  # don't need to keep checking the other hashes

        # check if there is only one measurement per mjd, if not, just take the most recent provenance
        for mjd, m_list in measurements_per_mjd.items():
            if isinstance(m_list, list):
                m_list.sort(key=lambda m: m.provenance.created_at)
                measurements_per_mjd[mjd] = m_list[-1]

        # set the measurements to the updated list
        self.measurements = list(measurements_per_mjd.values())

        # update the object's ra/dec to be the mean of all associated measurements
        # TODO: should we use a weighted sum, with the weights being the uncertainties? Issue #234
        self.ra = np.sum([m.ra for m in self.measurements]) / len(self.measurements)
        self.dec = np.sum([m.dec for m in self.measurements]) / len(self.measurements)
        self.calculate_coordinates()


Object.__table_args__ = sa.Index(
    f"objects_discovery_q3c_ang2ipix_idx",
    sa.func.q3c_ang2ipix(Object.discovery_ra, Object.discovery_dec)
)

Measurements.objects = orm.relationship(
    'Object',
    secondary=measurements_object_association_table,
    passive_deletes=True,
    cascade='save-update, merge, refresh-expire, expunge',
    lazy='selectin',
    doc="The object that this measurement is associated with. "
)
