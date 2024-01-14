"""unique constraint on products

Revision ID: 0232ca4409c0
Revises: 360a5ebe3848
Create Date: 2024-01-14 13:13:36.641716

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0232ca4409c0'
down_revision = '360a5ebe3848'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('cutouts', sa.Column('index_in_sources', sa.Integer(), nullable=False))
    op.create_unique_constraint('_cutouts_index_sources_provenance_uc', 'cutouts', ['index_in_sources', 'sources_id', 'provenance_id'])
    op.add_column('exposures', sa.Column('info', postgresql.JSONB(astext_type=sa.Text()), nullable=False))
    op.drop_column('exposures', 'header')
    op.add_column('images', sa.Column('ref_image_id', sa.BigInteger(), nullable=True))
    op.add_column('images', sa.Column('info', postgresql.JSONB(astext_type=sa.Text()), nullable=False))
    op.alter_column('images', 'section_id', existing_type=sa.TEXT(), nullable=True)
    op.drop_column('images', 'new_image_index')
    op.drop_column('images', 'ref_image_index')
    op.drop_column('images', 'header')
    op.create_unique_constraint('_measurements_cutouts_provenance_uc', 'measurements', ['cutouts_id', 'provenance_id'])
    op.create_unique_constraint('_source_list_image_provenance_uc', 'source_lists', ['image_id', 'provenance_id'])
    op.create_unique_constraint('_wcs_sources_provenance_uc', 'world_coordinates', ['sources_id', 'provenance_id'])
    op.create_unique_constraint('_zp_sources_provenance_uc', 'zero_points', ['sources_id', 'provenance_id'])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('_zp_sources_provenance_uc', 'zero_points', type_='unique')
    op.drop_constraint('_wcs_sources_provenance_uc', 'world_coordinates', type_='unique')
    op.drop_constraint('_source_list_image_provenance_uc', 'source_lists', type_='unique')
    op.drop_constraint('_measurements_cutouts_provenance_uc', 'measurements', type_='unique')
    op.add_column('images', sa.Column('header', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=False))
    op.add_column('images', sa.Column('ref_image_index', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('images', sa.Column('new_image_index', sa.INTEGER(), autoincrement=False, nullable=True))
    op.alter_column('images', 'section_id', existing_type=sa.TEXT(), nullable=False)
    op.drop_column('images', 'info')
    op.drop_column('images', 'ref_image_id')
    op.add_column('exposures', sa.Column('header', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=False))
    op.drop_column('exposures', 'info')
    op.drop_constraint('_cutouts_index_sources_provenance_uc', 'cutouts', type_='unique')
    op.drop_column('cutouts', 'index_in_sources')
    # ### end Alembic commands ###
