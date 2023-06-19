
import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base, SpatiallyIndexed


class Measurements(Base, SpatiallyIndexed):

    __tablename__ = 'measurements'

    cutouts_id = sa.Column(
        sa.ForeignKey('cutouts.id'),
        nullable=False,
        index=True,
        doc="ID of the cutout this measurement is associated with. "
    )

    cutouts = orm.relationship(
        'Cutouts',
        doc="The cutout this measurement is associated with. "
    )

    # TODO: we need to decide what columns are actually saved.
    #  E.g., should we save a single flux or an array/JSONB of fluxes?
    #  Same thing for scores (e.g., R/B).
    #  Are analytical cuts saved with the "scores"?
    #  What about things like centroid positions / PSF widths?

