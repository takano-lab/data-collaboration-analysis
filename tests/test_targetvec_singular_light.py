import numpy as np

from src.integrated_expression.runners import (
    build_targetvec_singular_light_projectors,
    build_targetvec_singular_projectors,
)


def test_targetvec_singular_light_matches_targetvec_singular_outputs():
    rng = np.random.default_rng(7)
    anchors = [
        rng.normal(size=(12, 6)),
        rng.normal(size=(12, 5)),
        rng.normal(size=(12, 4)),
    ]
    # Make one anchor rank-deficient.
    anchors[1][:, 4] = anchors[1][:, 3]

    projs_base, z_base, s_base = build_targetvec_singular_projectors(
        anchors_inter=anchors,
        dim_integrate=4,
        zerosum=False,
    )
    projs_light, z_light, s_light = build_targetvec_singular_light_projectors(
        anchors_inter=anchors,
        dim_integrate=4,
        zerosum=False,
    )

    assert np.allclose(z_base, z_light, atol=1e-8, rtol=1e-7)
    assert np.allclose(s_base, s_light, atol=1e-8, rtol=1e-7)

    for proj_base, proj_light, anchor in zip(projs_base, projs_light, anchors):
        out_base = proj_base(anchor)
        out_light = proj_light(anchor)
        assert np.allclose(out_base, out_light, atol=1e-7, rtol=1e-6)
