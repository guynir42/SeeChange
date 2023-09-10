"""image model

Revision ID: 4114e36a2555
Revises: f940bef6bf71
Create Date: 2023-05-31 16:39:35.909083

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '4114e36a2555'
down_revision = 'f940bef6bf71'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    image_type = postgresql.ENUM(
        "Sci",
        "ComSci",
        "Diff",
        "ComDiff",
        "Bias",
        "ComBias",
        "Dark",
        "ComDark",
        "DomeFlat",
        "ComDomeFlat",
        "SkyFlat",
        "ComSkyFlat",
        "TwiFlat",
        "ComTwiFlat",
        name='image_type'
    )
    image_type.create(op.get_bind())
    file_format = postgresql.ENUM('fits', 'hdf5', 'csv', 'npy', name='file_format')
    file_format.create(op.get_bind())
    image_format = postgresql.ENUM('fits', 'hdf5', name='image_format')
    image_format.create(op.get_bind())
    image_combine_method = postgresql.ENUM('coadd', 'subtraction', name='image_combine_method')
    image_combine_method.create(op.get_bind())

    op.create_table('image_sources',
    sa.Column('source_id', sa.Integer(), nullable=False),
    sa.Column('combined_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['combined_id'], ['images.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['source_id'], ['images.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('source_id', 'combined_id')
    )
    op.add_column('exposures', sa.Column('type', sa.Enum(
            "Sci",
            "ComSci",
            "Diff",
            "ComDiff",
            "Bias",
            "ComBias",
            "Dark",
            "ComDark",
            "DomeFlat",
            "ComDomeFlat",
            "SkyFlat",
            "ComSkyFlat",
            "TwiFlat",
            "ComTwiFlat",
            name='image_type'
        ), nullable=False)
    )
    op.add_column('exposures', sa.Column('format', sa.Enum('fits', 'hdf5', 'csv', 'npy', name='file_format'), nullable=False))
    op.drop_index('ix_exposures_section_id', table_name='exposures')
    op.create_index(op.f('ix_exposures_type'), 'exposures', ['type'], unique=False)
    op.drop_column('exposures', 'section_id')
    op.add_column('images', sa.Column('exposure_id', sa.BigInteger(), nullable=True))
    op.add_column('images', sa.Column('combine_method', sa.Enum('coadd', 'subtraction', name='image_combine_method'), nullable=True))
    op.add_column('images', sa.Column('type', sa.Enum(
            "Sci",
            "ComSci",
            "Diff",
            "ComDiff",
            "Bias",
            "ComBias",
            "Dark",
            "ComDark",
            "DomeFlat",
            "ComDomeFlat",
            "SkyFlat",
            "ComSkyFlat",
            "TwiFlat",
            "ComTwiFlat",
            name='image_type'
        ), nullable=False)
    )
    op.add_column('images', sa.Column('format', sa.Enum('fits', 'hdf5', 'csv', 'npy', name='file_format'), nullable=False))
    op.add_column('images', sa.Column('provenance_id', sa.BigInteger(), nullable=False))
    op.add_column('images', sa.Column('header', postgresql.JSONB(astext_type=sa.Text()), nullable=False))
    op.add_column('images', sa.Column('mjd', sa.Double(), nullable=False))
    op.add_column('images', sa.Column('end_mjd', sa.Double(), nullable=False))
    op.add_column('images', sa.Column('exp_time', sa.Float(), nullable=False))
    op.add_column('images', sa.Column('instrument', sa.Text(), nullable=False))
    op.add_column('images', sa.Column('telescope', sa.Text(), nullable=False))
    op.add_column('images', sa.Column('filter', sa.Text(), nullable=False))
    op.add_column('images', sa.Column('section_id', sa.Text(), nullable=False))
    op.add_column('images', sa.Column('project', sa.Text(), nullable=False))
    op.add_column('images', sa.Column('target', sa.Text(), nullable=False))
    op.add_column('images', sa.Column('filepath', sa.Text(), nullable=False))
    op.add_column('images', sa.Column('filepath_extensions', sa.ARRAY(sa.Text()), nullable=True))
    op.add_column('images', sa.Column('ra', sa.Double(), nullable=False))
    op.add_column('images', sa.Column('dec', sa.Double(), nullable=False))
    op.add_column('images', sa.Column('gallat', sa.Double(), nullable=True))
    op.add_column('images', sa.Column('gallon', sa.Double(), nullable=True))
    op.add_column('images', sa.Column('ecllat', sa.Double(), nullable=True))
    op.add_column('images', sa.Column('ecllon', sa.Double(), nullable=True))
    op.create_index('images_q3c_ang2ipix_idx', 'images', [sa.text('q3c_ang2ipix(ra, dec)')], unique=False)
    op.create_index(op.f('ix_images_combine_method'), 'images', ['combine_method'], unique=False)
    op.create_index(op.f('ix_images_ecllat'), 'images', ['ecllat'], unique=False)
    op.create_index(op.f('ix_images_end_mjd'), 'images', ['end_mjd'], unique=False)
    op.create_index(op.f('ix_images_exp_time'), 'images', ['exp_time'], unique=False)
    op.create_index(op.f('ix_images_exposure_id'), 'images', ['exposure_id'], unique=False)
    op.create_index(op.f('ix_images_filepath'), 'images', ['filepath'], unique=True)
    op.create_index(op.f('ix_images_filter'), 'images', ['filter'], unique=False)
    op.create_index(op.f('ix_images_gallat'), 'images', ['gallat'], unique=False)
    op.create_index(op.f('ix_images_instrument'), 'images', ['instrument'], unique=False)
    op.create_index(op.f('ix_images_mjd'), 'images', ['mjd'], unique=False)
    op.create_index(op.f('ix_images_provenance_id'), 'images', ['provenance_id'], unique=False)
    op.create_index(op.f('ix_images_section_id'), 'images', ['section_id'], unique=False)
    op.create_index(op.f('ix_images_project'), 'images', ['project'], unique=False)
    op.create_index(op.f('ix_images_target'), 'images', ['target'], unique=False)
    op.create_index(op.f('ix_images_telescope'), 'images', ['telescope'], unique=False)
    op.create_index(op.f('ix_images_type'), 'images', ['type'], unique=False)
    op.create_foreign_key(None, 'images', 'provenances', ['provenance_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key(None, 'images', 'exposures', ['exposure_id'], ['id'], ondelete='SET NULL')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'images', type_='foreignkey')
    op.drop_index(op.f('ix_images_type'), table_name='images')
    op.drop_index(op.f('ix_images_telescope'), table_name='images')
    op.drop_index(op.f('ix_images_project'), table_name='images')
    op.drop_index(op.f('ix_images_target'), table_name='images')
    op.drop_index(op.f('ix_images_section_id'), table_name='images')
    op.drop_index(op.f('ix_images_provenance_id'), table_name='images')
    op.drop_index(op.f('ix_images_mjd'), table_name='images')
    op.drop_index(op.f('ix_images_instrument'), table_name='images')
    op.drop_index(op.f('ix_images_gallat'), table_name='images')
    op.drop_index(op.f('ix_images_filter'), table_name='images')
    op.drop_index(op.f('ix_images_filepath'), table_name='images')
    op.drop_index(op.f('ix_images_exposure_id'), table_name='images')
    op.drop_index(op.f('ix_images_exp_time'), table_name='images')
    op.drop_index(op.f('ix_images_end_mjd'), table_name='images')
    op.drop_index(op.f('ix_images_ecllat'), table_name='images')
    op.drop_index(op.f('ix_images_combine_method'), table_name='images')
    op.drop_index('images_q3c_ang2ipix_idx', table_name='images')
    op.drop_column('images', 'ecllon')
    op.drop_column('images', 'ecllat')
    op.drop_column('images', 'gallon')
    op.drop_column('images', 'gallat')
    op.drop_column('images', 'dec')
    op.drop_column('images', 'ra')
    op.drop_column('images', 'filepath_extensions')
    op.drop_column('images', 'filepath')
    op.drop_column('images', 'target')
    op.drop_column('images', 'project')
    op.drop_column('images', 'section_id')
    op.drop_column('images', 'filter')
    op.drop_column('images', 'telescope')
    op.drop_column('images', 'instrument')
    op.drop_column('images', 'exp_time')
    op.drop_column('images', 'end_mjd')
    op.drop_column('images', 'mjd')
    op.drop_column('images', 'header')
    op.drop_column('images', 'provenance_id')
    op.drop_column('images', 'type')
    op.drop_column('images', 'format')
    op.drop_column('images', 'combine_method')
    op.drop_column('images', 'exposure_id')
    op.add_column('exposures', sa.Column('section_id', sa.TEXT(), autoincrement=False, nullable=False))
    op.drop_index(op.f('ix_exposures_type'), table_name='exposures')
    op.create_index('ix_exposures_section_id', 'exposures', ['section_id'], unique=False)
    op.drop_column('exposures', 'type')
    op.drop_column('exposures', 'format')
    op.drop_table('image_sources')
    # ### end Alembic commands ###
