"""rework cutouts and measurements

Revision ID: 7384c6d07485
Revises: a375526c8260
Create Date: 2024-06-28 17:57:44.173607

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7384c6d07485'
down_revision = 'a375526c8260'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('_cutouts_index_sources_provenance_uc', 'cutouts', type_='unique')
    op.drop_index('ix_cutouts_ecllat', table_name='cutouts')
    op.drop_index('ix_cutouts_gallat', table_name='cutouts')
    op.drop_index('ix_cutouts_filepath', table_name='cutouts')
    op.create_index(op.f('ix_cutouts_filepath'), 'cutouts', ['filepath'], unique=True)
    op.create_unique_constraint('_cutouts_sources_provenance_uc', 'cutouts', ['sources_id', 'provenance_id'])
    op.drop_column('cutouts', 'ecllon')
    op.drop_column('cutouts', 'ra')
    op.drop_column('cutouts', 'gallat')
    op.drop_column('cutouts', 'index_in_sources')
    op.drop_column('cutouts', 'y')
    op.drop_column('cutouts', 'gallon')
    op.drop_column('cutouts', 'dec')
    op.drop_column('cutouts', 'x')
    op.drop_column('cutouts', 'ecllat')
    op.add_column('measurements', sa.Column('index_in_sources', sa.Integer(), nullable=False))
    op.add_column('measurements', sa.Column('center_x_pixel', sa.Integer(), nullable=False))
    op.add_column('measurements', sa.Column('center_y_pixel', sa.Integer(), nullable=False))
    op.drop_constraint('_measurements_cutouts_provenance_uc', 'measurements', type_='unique')
    op.create_unique_constraint('_measurements_cutouts_provenance_uc', 'measurements', ['cutouts_id', 'index_in_sources', 'provenance_id'])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('_measurements_cutouts_provenance_uc', 'measurements', type_='unique')
    op.create_unique_constraint('_measurements_cutouts_provenance_uc', 'measurements', ['cutouts_id', 'provenance_id'])
    op.drop_column('measurements', 'center_y_pixel')
    op.drop_column('measurements', 'center_x_pixel')
    op.drop_column('measurements', 'index_in_sources')
    op.add_column('cutouts', sa.Column('ecllat', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True))
    op.add_column('cutouts', sa.Column('x', sa.INTEGER(), autoincrement=False, nullable=False))
    op.add_column('cutouts', sa.Column('dec', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False))
    op.add_column('cutouts', sa.Column('gallon', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True))
    op.add_column('cutouts', sa.Column('y', sa.INTEGER(), autoincrement=False, nullable=False))
    op.add_column('cutouts', sa.Column('index_in_sources', sa.INTEGER(), autoincrement=False, nullable=False))
    op.add_column('cutouts', sa.Column('gallat', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True))
    op.add_column('cutouts', sa.Column('ra', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False))
    op.add_column('cutouts', sa.Column('ecllon', sa.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True))
    op.drop_constraint('_cutouts_sources_provenance_uc', 'cutouts', type_='unique')
    op.drop_index(op.f('ix_cutouts_filepath'), table_name='cutouts')
    op.create_index('ix_cutouts_filepath', 'cutouts', ['filepath'], unique=False)
    op.create_index('ix_cutouts_gallat', 'cutouts', ['gallat'], unique=False)
    op.create_index('ix_cutouts_ecllat', 'cutouts', ['ecllat'], unique=False)
    op.create_unique_constraint('_cutouts_index_sources_provenance_uc', 'cutouts', ['index_in_sources', 'sources_id', 'provenance_id'])
    # ### end Alembic commands ###
