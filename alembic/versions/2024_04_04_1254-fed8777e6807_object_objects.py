"""object objects

Revision ID: fed8777e6807
Revises: b2129499bfcd
Create Date: 2024-04-04 12:54:40.846173

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fed8777e6807'
down_revision = 'b2129499bfcd'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('objects',
    sa.Column('name', sa.String(), nullable=False, unique=True),
    sa.Column('is_test', sa.Boolean(), nullable=False),
    sa.Column('is_fake', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('modified', sa.DateTime(), nullable=False),
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('ra', sa.Double(), nullable=False),
    sa.Column('dec', sa.Double(), nullable=False),
    sa.Column('gallat', sa.Double(), nullable=True),
    sa.Column('gallon', sa.Double(), nullable=True),
    sa.Column('ecllat', sa.Double(), nullable=True),
    sa.Column('ecllon', sa.Double(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_objects_created_at'), 'objects', ['created_at'], unique=False)
    op.create_index(op.f('ix_objects_ecllat'), 'objects', ['ecllat'], unique=False)
    op.create_index(op.f('ix_objects_gallat'), 'objects', ['gallat'], unique=False)
    op.create_index(op.f('ix_objects_id'), 'objects', ['id'], unique=False)
    op.create_index(op.f('ix_objects_name'), 'objects', ['name'], unique=False)
    op.create_index('objects_q3c_ang2ipix_idx', 'objects', [sa.text('q3c_ang2ipix(ra, dec)')], unique=False)
    op.add_column('measurements', sa.Column('object_id', sa.BigInteger(), nullable=False))
    op.create_index(op.f('ix_measurements_object_id'), 'measurements', ['object_id'], unique=False)
    op.create_foreign_key('measurements_object_id_fkey', 'measurements', 'objects', ['object_id'], ['id'], ondelete='CASCADE')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('measurements_object_id_fkey', 'measurements', type_='foreignkey')
    op.drop_index(op.f('ix_measurements_object_id'), table_name='measurements')
    op.drop_column('measurements', 'object_id')
    op.drop_index('objects_q3c_ang2ipix_idx', table_name='objects')
    op.drop_index(op.f('ix_objects_name'), table_name='objects')
    op.drop_index(op.f('ix_objects_id'), table_name='objects')
    op.drop_index(op.f('ix_objects_gallat'), table_name='objects')
    op.drop_index(op.f('ix_objects_ecllat'), table_name='objects')
    op.drop_index(op.f('ix_objects_created_at'), table_name='objects')
    op.drop_table('objects')
    # ### end Alembic commands ###
