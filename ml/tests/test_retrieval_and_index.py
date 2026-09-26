import numpy as np
import pytest

from rifteye_ml.encoders import ColorGrid, DHash, get_encoder
from rifteye_ml.fixtures import load_fixture_image, synthetic_card, synthetic_catalog
from rifteye_ml.index import IndexModelMismatch, build_index, load_index
from rifteye_ml.retrieval import accuracy, ranked_labels, search, topk


@pytest.mark.parametrize("spec", ["colorgrid", "colorgrid:8", "dhash"])
def test_embeddings_are_unit_norm_and_deterministic(spec):
    enc = get_encoder(spec)
    imgs = [synthetic_card(i) for i in range(4)]
    a, b = enc.embed(imgs), enc.embed(imgs)
    assert a.shape == (4, enc.dim)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(a, b)


def test_clean_queries_find_themselves():
    imgs = [synthetic_card(i) for i in range(30)]
    enc = ColorGrid()
    g = enc.embed(imgs)
    idx, scores = topk(g, g, k=5)
    assert (idx[:, 0] == np.arange(30)).all()
    assert np.allclose(scores[:, 0], 1.0, atol=1e-5)


def test_rotation_invariant_search_reads_rotated_cards_and_reports_rotation():
    imgs = [synthetic_card(i) for i in range(20)]
    enc = ColorGrid()
    g = enc.embed(imgs)
    applied = [0, 90, 180, 270] * 5
    queries = [im.rotate(r, expand=True) if r else im for im, r in zip(imgs, applied)]
    idx, _, rot = search(enc, g, queries, k=3, rotation_invariant=True)
    assert (idx[:, 0] == np.arange(20)).all()
    assert all((a + int(r)) % 360 == 0 for a, r in zip(applied, rot[:, 0]))
    idx0, _, _ = search(enc, g, queries, k=3, rotation_invariant=False)
    assert (idx0[:, 0] == np.arange(20)).mean() < 1.0  # without it, sideways cards fail


def test_card_level_rollup():
    idx = np.array([[0, 1, 2], [1, 2, 0]])
    labels = ["hero", "hero", "villain"]  # printings 0 and 1 are the same card
    ranked = ranked_labels(idx, labels)
    assert ranked == [["hero", "villain"], ["hero", "villain"]]
    assert accuracy(ranked, ["hero", "villain"]) == {"top1": 0.5, "top5": 1.0, "n": 2}


def test_index_roundtrip_and_encoder_guard(tmp_path):
    rows = synthetic_catalog(12)
    enc = ColorGrid(8)
    manifest = build_index(rows, load_fixture_image, enc, tmp_path, catalog_version="test-1")
    assert manifest["rows"] == [r["printing_id"] for r in rows] and manifest["dtype"] == "float16"
    assert (tmp_path / "index.f16").stat().st_size == 12 * enc.dim * 2
    m, mat = load_index(tmp_path, enc)
    np.testing.assert_allclose(mat, enc.embed([load_fixture_image(r) for r in rows]), atol=2e-3)
    with pytest.raises(IndexModelMismatch):
        load_index(tmp_path, DHash(8))


def test_pyramid_routes_queries_to_the_nearest_scale():
    from rifteye_ml.retrieval import Pyramid, at_long_side

    imgs = [synthetic_card(i) for i in range(16)]
    small = at_long_side(imgs[0], 40)
    assert max(small.size) == 40 and small.height > small.width
    enc = ColorGrid()
    pyr = Pyramid.build(enc, imgs, [40, 120])
    # The log-space midpoint of 40 and 120 is 69 px.
    assert (pyr.level_for(38), pyr.level_for(66), pyr.level_for(72), pyr.level_for(500)) == (40, 40, 120, 120)
    queries = [at_long_side(im, 40 if i % 2 else 120) for i, im in enumerate(imgs)]
    idx, _, _ = search(enc, pyr, queries, k=3, rotation_invariant=True)
    assert idx.shape == (16, 3) and (idx[:, 0] == np.arange(16)).all()
    with pytest.raises(ValueError):
        Pyramid.build(enc, imgs, [])


def test_timm_spec_parsing():
    from rifteye_ml.encoders import parse_timm_spec

    assert parse_timm_spec("vit_small_patch14_dinov2.lvd142m") == ("vit_small_patch14_dinov2.lvd142m", 224, "")
    assert parse_timm_spec("vit_pe_core_small_patch16_384.fb@224") == ("vit_pe_core_small_patch16_384.fb", 224, "")
    assert parse_timm_spec("vit_small_patch14_dinov2.lvd142m@168/avg") == ("vit_small_patch14_dinov2.lvd142m", 168, "avg")
    with pytest.raises(ValueError):
        parse_timm_spec("@224")
