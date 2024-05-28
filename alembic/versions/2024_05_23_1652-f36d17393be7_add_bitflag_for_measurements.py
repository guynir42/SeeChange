"""add bitflag for measurements

Revision ID: f36d17393be7
Revises: ec64a8fd8cf3
Create Date: 2024-05-23 16:52:07.448402

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f36d17393be7'
down_revision = '2ea9f6f0b790'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('measurements', sa.Column('_bitflag', sa.BIGINT(), nullable=False))
    op.add_column('measurements', sa.Column('description', sa.Text(), nullable=True))
    op.add_column('measurements', sa.Column('_upstream_bitflag', sa.BIGINT(), nullable=False))
    op.create_index(op.f('ix_measurements__bitflag'), 'measurements', ['_bitflag'], unique=False)
    op.create_index(op.f('ix_measurements__upstream_bitflag'), 'measurements', ['_upstream_bitflag'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_measurements__upstream_bitflag'), table_name='measurements')
    op.drop_index(op.f('ix_measurements__bitflag'), table_name='measurements')
    op.drop_column('measurements', '_upstream_bitflag')
    op.drop_column('measurements', 'description')
    op.drop_column('measurements', '_bitflag')
    # ### end Alembic commands ###
