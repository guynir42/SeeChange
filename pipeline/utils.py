import git
from collections import defaultdict
import numpy as np

import sqlalchemy as sa

from models.base import SmartSession


def get_git_hash():
    """Get the commit hash of the current git repo. """
    # TODO: what if we are running on a production server
    #  that doesn't have git? Consider replacing this with
    #  an environmental variable that is automatically updated
    #  with the current git hash.

    repo = git.Repo(search_parent_directories=True)
    git_hash = repo.head.object.hexsha

    return git_hash


def get_latest_provenance(process_name, session=None):
    """
    Find the provenance object that fits the process_name
    that is the most recent.
    # TODO: we need to think about what "most recent" means.

    Parameters
    ----------
    process_name: str
        Name of the process that created this provenance object.
        Examples can include: "calibration", "subtraction", "source extraction" or just "level1".
    session: sqlalchemy.orm.session.Session or SmartSession

    Returns
    -------
    Provenance
        The most recent provenance object that matches the process_name.
        If not found, returns None.
    """
    # importing the models here to avoid circular imports
    from models.base import SmartSession
    from models.provenance import Provenance

    with SmartSession(session) as session:
        prov = session.scalars(
            sa.select(Provenance).where(
                Provenance.process == process_name
            ).order_by(Provenance.id.desc())
        ).first()

    return prov


def normalize_header_key(key):
    """
    Normalize the header key to be all uppercase and
    remove spaces and underscores.
    """
    return key.upper().replace(' ', '').replace('_', '')


def parse_session(*args, **kwargs):
    """
    Parse the arguments and keyword arguments to find a SmartSession or SQLAlchemy session.
    If one of the kwargs is called "session" that value will be returned.
    Otherwise, if any of the unnamed arguments is a session, the last one will be returned.
    If neither of those are found, None will be returned.
    Will also return the args and kwargs with any sessions removed.

    Parameters
    ----------
    args: list
        List of unnamed arguments
    kwargs: dict
        Dictionary of named arguments

    Returns
    -------
    args: list
        List of unnamed arguments with any sessions removed.
    kwargs: dict
        Dictionary of named arguments with any sessions removed.
    session: SmartSession or SQLAlchemy session or None
        The session found in the arguments or kwargs.
    """
    session = None
    sessions = [arg for arg in args if isinstance(arg, sa.orm.session.Session)]
    if len(sessions) > 0:
        session = sessions[-1]
    args = [arg for arg in args if not isinstance(arg, sa.orm.session.Session)]

    for key in kwargs.keys():
        if key in ['session']:
            if not isinstance(kwargs[key], sa.orm.session.Session):
                raise ValueError(f'Session must be a sqlalchemy.orm.session.Session, got {type(kwargs[key])}')
            session = kwargs.pop(key)

    return args, kwargs, session


