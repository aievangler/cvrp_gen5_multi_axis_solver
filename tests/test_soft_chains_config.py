from gen3.v1.ls import SoftChainsConfig


def test_prune_depths_default():
    cfg = SoftChainsConfig()
    # max_depth defaults to 8 -> expected prune depths
    assert cfg.prune_depths() == {2, 4, 5, 8}


def test_prune_depths_custom_10():
    cfg = SoftChainsConfig(max_depth=10)
    assert cfg.prune_depths() == {3, 5, 6, 10}

