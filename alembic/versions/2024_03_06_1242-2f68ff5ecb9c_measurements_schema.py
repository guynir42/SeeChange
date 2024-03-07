"""measurements schema

Revision ID: 2f68ff5ecb9c
Revises: ef05cbdd10ea
Create Date: 2024-03-06 12:42:02.319335

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '2f68ff5ecb9c'
down_revision = 'ef05cbdd10ea'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('measurements', sa.Column('mjd', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('exptime', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('filter', sa.String(), nullable=False))
    op.add_column('measurements', sa.Column('flux_psf', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('flux_psf_err', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('flux_apertures', sa.ARRAY(sa.Float()), nullable=False))
    op.add_column('measurements', sa.Column('flux_apertures_err', sa.ARRAY(sa.Float()), nullable=False))
    op.add_column('measurements', sa.Column('aper_radii', sa.ARRAY(sa.Float()), nullable=False))
    op.add_column('measurements', sa.Column('best_aperture', sa.SMALLINT(), nullable=False))
    op.add_column('measurements', sa.Column('mag_psf', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('mag_psf_err', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('mag_apertures', sa.ARRAY(sa.Float()), nullable=False))
    op.add_column('measurements', sa.Column('mag_apertures_err', sa.ARRAY(sa.Float()), nullable=False))
    op.add_column('measurements', sa.Column('magnitude', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('limmag', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('background', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('background_err', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('area_psf', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('area_apertures', sa.ARRAY(sa.Float()), nullable=False))
    op.add_column('measurements', sa.Column('offset_x', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('offset_y', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('width', sa.Integer(), nullable=False))
    op.add_column('measurements', sa.Column('elongation', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('position_angle', sa.Float(), nullable=False))
    op.add_column('measurements', sa.Column('badness_scores', JSONB, nullable=False))
    op.add_column('measurements', sa.Column('passing', sa.Boolean(), nullable=False))
    op.create_index(
        op.f('ix_measurements_badness_scores'), 'measurements', ['badness_scores'], unique=False, postgresql_using="gin"
    )
    op.create_index(op.f('ix_measurements_exptime'), 'measurements', ['exptime'], unique=False)
    op.create_index(op.f('ix_measurements_filter'), 'measurements', ['filter'], unique=False)
    op.create_index(op.f('ix_measurements_magnitude'), 'measurements', ['magnitude'], unique=False)
    op.create_index(op.f('ix_measurements_limmag'), 'measurements', ['limmag'], unique=False)
    op.create_index(op.f('ix_measurements_mag_psf'), 'measurements', ['mag_psf'], unique=False)
    op.create_index(op.f('ix_measurements_mjd'), 'measurements', ['mjd'], unique=False)
    op.create_index(op.f('ix_measurements_passing'), 'measurements', ['passing'], unique=False)
    op.create_index(op.f('ix_measurements_width'), 'measurements', ['width'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_measurements_width'), table_name='measurements')
    op.drop_index(op.f('ix_measurements_passing'), table_name='measurements')
    op.drop_index(op.f('ix_measurements_mjd'), table_name='measurements')
    op.drop_index(op.f('ix_measurements_mag_psf'), table_name='measurements')
    op.drop_index(op.f('ix_measurements_limmag'), table_name='measurements')
    op.drop_index(op.f('ix_measurements_magnitude'), table_name='measurements')
    op.drop_index(op.f('ix_measurements_filter'), table_name='measurements')
    op.drop_index(op.f('ix_measurements_exptime'), table_name='measurements')
    op.drop_index(op.f('ix_measurements_badness_scores'), table_name='measurements')
    op.drop_column('measurements', 'passing')
    op.drop_column('measurements', 'badness_scores')
    op.drop_column('measurements', 'position_angle')
    op.drop_column('measurements', 'elongation')
    op.drop_column('measurements', 'width')
    op.drop_column('measurements', 'offset_y')
    op.drop_column('measurements', 'offset_x')
    op.drop_column('measurements', 'area_apertures')
    op.drop_column('measurements', 'area_psf')
    op.drop_column('measurements', 'background_err')
    op.drop_column('measurements', 'background')
    op.drop_column('measurements', 'limmag')
    op.drop_column('measurements', 'magnitude')
    op.drop_column('measurements', 'mag_apertures_err')
    op.drop_column('measurements', 'mag_apertures')
    op.drop_column('measurements', 'mag_psf_err')
    op.drop_column('measurements', 'mag_psf')
    op.drop_column('measurements', 'best_aperture')
    op.drop_column('measurements', 'aper_radii')
    op.drop_column('measurements', 'flux_apertures_err')
    op.drop_column('measurements', 'flux_apertures')
    op.drop_column('measurements', 'flux_psf_err')
    op.drop_column('measurements', 'flux_psf')
    op.drop_column('measurements', 'filter')
    op.drop_column('measurements', 'exptime')
    op.drop_column('measurements', 'mjd')
    # ### end Alembic commands ###
