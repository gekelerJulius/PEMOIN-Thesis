from pemoin.coordinate_systems.alignment import AlignmentSettings


def test_alignment_settings_defaults_are_piecewise_and_strict():
    settings = AlignmentSettings.from_mapping({})
    assert settings.mode == "piecewise_plane_anchor"
    assert settings.fail_on_consistency_error is True
    assert settings.min_plane_scale_inlier_ratio > 0.0


def test_alignment_settings_rejects_legacy_modes():
    try:
        AlignmentSettings.from_mapping({"mode": "legacy"})
        raised = False
    except ValueError:
        raised = True
    assert raised
