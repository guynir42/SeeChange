
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import JSONB


from models.base import Base, AutoIDMixin
from models.enums_and_bitflags import (
    bitflag_to_string,
    string_to_bitflag,
    report_progress_dict,
    report_progress_inverse,
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
        return bitflag_to_string(self.progress_steps_bitflag, report_progress_dict)

    @progress_steps.setter
    def progress_steps(self, value):
        """Set the progress steps for this report using a comma separated string. """
        self.progress_steps_bitflag = string_to_bitflag(value, report_progress_inverse)

    def append_progress(self, value):
        """Add some keywords (in a comma separated string)
        describing what is processing steps were done on this section.
        The keywords will be added to the list "progress_steps"
        and progress_bitflag for this report will be updated accordingly.
        """
        self.progress_steps_bitflag |= string_to_bitflag(value, report_progress_inverse)

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

    products_saved_bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        default=0,
        index=True,
        doc='Bitflag recording which pipeline products were not None when the pipeline finished. '
    )

    @property
    def products_saved(self):
        """A comma separated string representing which products
        have already been successfully saved using the datastore when the pipeline finished.
        """
        return bitflag_to_string(self.products_saved_bitflag, pipeline_products_dict)

    @products_saved.setter
    def products_saved(self, value):
        """Set the products_saved for this report using a comma separated string. """
        self.products_saved_bitflag = string_to_bitflag(value, pipeline_products_inverse)

    def append_products_saved(self, value):
        """Add some keywords (in a comma separated string)
        describing which products were successfully saved by the datastore.
        The keywords will be added to the list "products_saved"
        and products_saved_bitflag for this report will be updated accordingly.
        """
        self.products_saved_bitflag |= string_to_bitflag(value, pipeline_products_inverse)

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