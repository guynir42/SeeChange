
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import JSONB


from models.base import Base, SeeChangeBase, AutoIDMixin, SmartSession
from models.enums_and_bitflags import (
    bitflag_to_string,
    string_to_bitflag,
    process_steps_dict,
    process_steps_inverse,
    pipeline_products_dict,
    pipeline_products_inverse,
)


class Report(Base, AutoIDMixin):
    """A report on the status of analysis of one section from an Exposure.

    The report's main role is to keep a database record of when we started
    and finished processing this section of the Exposure. It also keeps
    track of any errors or warnings that came up during processing.
    """
    __tablename__ = 'reports'

    exposure_id = sa.Column(
        sa.ForeignKey('exposures.id', ondelete='CASCADE', name='reports_exposure_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the exposure for which the report was made. "
        )
    )

    exposure = orm.relationship(
        'Exposure',
        cascade='save-update, merge, refresh-expire, expunge',
        doc=(
            "Exposure for which the report was made. "
        )
    )

    section_id = sa.Column(
        sa.Text,
        nullable=False,
        doc=(
            "ID of the section of the exposure for which the report was made. "
        )
    )

    started_at = sa.Column(
        sa.DateTime,
        nullable=False,
        doc=(
            "Time when processing of the section started. "
        )
    )

    finished_at = sa.Column(
        sa.DateTime,
        nullable=True,
        doc=(
            "Time when processing of the section finished. "
            "If an error occurred, this will show the time of the error. "
            "If the processing is still ongoing (or hanging) this will be NULL. "
        )
    )

    success = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc=(
            "Whether the processing of this section was successful. "
        )
    )

    worker_id = sa.Column(
        sa.Text,
        nullable=False,
        doc=(
            "ID of the worker/process that ran this section. "
        )
    )

    error_step = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "Name of the processing step where an error occurred. "
        )
    )

    error_type = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "Type of error that was raised during processing. "
        )
    )

    error_message = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "Error message that was raised during processing. "
        )
    )

    warnings = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "Comma-separated string of warnings that were raised during processing. "
            "Each warning begins with the processing step name, followed by the warning message and trace. "
        )
    )

    process_memory = sa.Column(
        JSONB,
        nullable=False,
        default={},
        doc='Memory usage of the process during processing. '
            'Each key in the dictionary is for a processing step, '
            'and the value is the memory usage in megabytes. '
    )

    process_runtime = sa.Column(
        JSONB,
        nullable=False,
        default={},
        doc='Runtime of the process during processing. '
            'Each key in the dictionary is for a processing step, '
            'and the value is the runtime in seconds. '
    )

    progress_steps_bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        default=0,
        index=True,
        doc='Bitflag recording what processing steps have already been applied to this section. '
    )

    @property
    def progress_steps(self):
        """A comma separated string of the processing steps that have already been applied to this section. """
        return bitflag_to_string(self.progress_steps_bitflag, process_steps_dict)

    @progress_steps.setter
    def progress_steps(self, value):
        """Set the progress steps for this report using a comma separated string. """
        self.progress_steps_bitflag = string_to_bitflag(value, process_steps_inverse)

    def append_progress(self, value):
        """Add some keywords (in a comma separated string)
        describing what is processing steps were done on this section.
        The keywords will be added to the list "progress_steps"
        and progress_bitflag for this report will be updated accordingly.
        """
        self.progress_steps_bitflag |= string_to_bitflag(value, process_steps_inverse)

    products_exist_bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        default=0,
        index=True,
        doc='Bitflag recording which pipeline products were not None when the pipeline finished. '
    )

    @property
    def products_exist(self):
        """A comma separated string representing which products
        have already been filled on the datastore when the pipeline finished.
        """
        return bitflag_to_string(self.products_exist_bitflag, pipeline_products_dict)

    @products_exist.setter
    def products_exist(self, value):
        """Set the products_exist for this report using a comma separated string. """
        self.products_exist_bitflag = string_to_bitflag(value, pipeline_products_inverse)

    def append_products_exist(self, value):
        """Add some keywords (in a comma separated string)
        describing which products existed (were not None) on the datastore.
        The keywords will be added to the list "products_exist"
        and products_exist_bitflag for this report will be updated accordingly.
        """
        self.products_exist_bitflag |= string_to_bitflag(value, pipeline_products_inverse)

    products_committed_bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        default=0,
        index=True,
        doc='Bitflag recording which pipeline products were not None when the pipeline finished. '
    )

    @property
    def products_committed(self):
        """A comma separated string representing which products
        have already been successfully saved using the datastore when the pipeline finished.
        """
        return bitflag_to_string(self.products_committed_bitflag, pipeline_products_dict)

    @products_committed.setter
    def products_committed(self, value):
        """Set the products_committed for this report using a comma separated string. """
        self.products_committed_bitflag = string_to_bitflag(value, pipeline_products_inverse)

    def append_products_committed(self, value):
        """Add some keywords (in a comma separated string)
        describing which products were successfully saved by the datastore.
        The keywords will be added to the list "products_committed"
        and products_committed_bitflag for this report will be updated accordingly.
        """
        self.products_committed_bitflag |= string_to_bitflag(value, pipeline_products_inverse)

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='images_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this report. "
            "The provenance has upstreams that point to the "
            "measurements and R/B score objects that themselves "
            "point back to all the other provenances that were "
            "used to produce this report. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "The provenance of this report. "
            "The provenance has upstreams that point to the "
            "measurements and R/B score objects that themselves "
            "point back to all the other provenances that were "
            "used to produce this report. "
        )
    )

    def __init__(self, **kwargs):
        SeeChangeBase.__init__(self)  # do not pass kwargs to Base.__init__, as there may be non-column attributes

        # manually set all properties (columns or not)
        self.set_attributes_from_dict(kwargs)

        if self.worker_id is None:
            self.worker_id = 'unknown'  # TODO: replace this with a real worker ID

    @orm.reconstructor
    def init_on_load(self):
        SeeChangeBase.init_on_load(self)

    def scan_datastore(self, ds, process_step, session=None):
        """Go over all the data in a datastore and update the report accordingly.
        Will commit the changes to the database.
        If there are any exceptions pending on the datastore it will re-raise them.
        """
        # parse the error, if it exists, so we can get to other data products without raising
        exception = ds.read_exception()
        
        # append the newest step to the progress bitflag
        self.append_progress(process_step)
        
        # check which objects exist on the datastore, and which have been committed
        for prod in pipeline_products_dict.values():
            if getattr(ds, prod) is not None:
                self.append_products_exist(prod)
        
        self.products_committed = ds.products_committed
        
        # store the runtime and memory usage statistics

        # parse the warnings, if they exist
        selfwarnings = ds.read_warnings()
        
        if exception is not None:
            self.error_type = exception.__class__.__name__
            self.error_message = str(exception)
            self.error_step = process_step

        with SmartSession(session) as session:
            self.commit_to_database(session=session)

        if exception is not None:
            raise exception

    def commit_to_database(self, session):
        """Commit this report to the database. """
        session.merge(self)
        session.commit()

