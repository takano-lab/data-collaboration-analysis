import copy

from config.config import Config


def test_config_copy_preserves_dynamic_attributes():
    cfg = Config(dataset="har_subject", anchor_method="smote")

    copied = copy.copy(cfg)
    copied.anchor_method = "gaussian"

    assert copied.dataset == "har_subject"
    assert copied.anchor_method == "gaussian"
    assert cfg.anchor_method == "smote"
    assert cfg.missing_key is None
