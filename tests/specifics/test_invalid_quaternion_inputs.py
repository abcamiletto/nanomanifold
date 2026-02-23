import numpy as np

from nanomanifold import SO3


def test_distance_equivalent_rotations_no_error():
    q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    q_neg = -q
    dist = SO3.distance(q, q_neg)
    assert np.allclose(np.array(dist), 0.0, atol=1e-12)
