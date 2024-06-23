import pytest
import uuid

from pipeline.ref_maker import RefMaker

from models.reference import Reference


def test_finding_references(ptf_ref):
    with pytest.raises(ValueError, match='Must provide both'):
        ref = Reference.get_references(ra=188)
    with pytest.raises(ValueError, match='Must provide both'):
        ref = Reference.get_references(dec=4.5)
    with pytest.raises(ValueError, match='Must provide both'):
        ref = Reference.get_references(target='foo')
    with pytest.raises(ValueError, match='Must provide both'):
        ref = Reference.get_references(section_id='bar')
    with pytest.raises(ValueError, match='Must provide both'):
        ref = Reference.get_references(ra=188, section_id='bar')
    with pytest.raises(ValueError, match='Must provide both'):
        ref = Reference.get_references(dec=4.5, target='foo')
    with pytest.raises(ValueError, match='Must provide either ra and dec, or target and section_id'):
        ref = Reference.get_references()
    with pytest.raises(ValueError, match='Cannot provide target/section_id and also ra/dec! '):
        ref = Reference.get_references(ra=188, dec=4.5, target='foo', section_id='bar')

    ref = Reference.get_references(ra=188, dec=4.5)
    assert len(ref) == 1
    assert ref[0].id == ptf_ref.id

    ref = Reference.get_references(ra=188, dec=4.5, provenance_ids=ptf_ref.provenance_id)
    assert len(ref) == 1
    assert ref[0].id == ptf_ref.id

    ref = Reference.get_references(ra=0, dec=0)
    assert len(ref) == 0

    ref = Reference.get_references(target='foo', section_id='bar')
    assert len(ref) == 0

    ref = Reference.get_references(ra=180, dec=4.5, provenance_ids=['foo', 'bar'])
    assert len(ref) == 0


def test_making_refsets():
    name = uuid.uuid4().hex
    maker = RefMaker(maker={'name': name, 'instrument': 'PTF'})

    assert maker.im_provs is None
    assert maker.ex_provs is None
    assert maker.coadd_im_prov is None
    assert maker.coadd_ex_prov is None
    assert maker.ref_upstream_hash is None

    new_ref = maker.run(ra=0, dec=0)
    assert new_ref is None

