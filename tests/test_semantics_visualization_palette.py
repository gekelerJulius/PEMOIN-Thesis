import json

import numpy as np

from pemoin.data.contracts import FrameData, ResourceStore, SemanticSegment, SemanticsData
from pemoin.visualization.semantics import (
    SemanticsVisualizationSettings,
    generate_semantics_visualizations,
    render_semantics_overlay,
)
from pemoin.visualization.semantics_debug import (
    SemanticsDebugSettings,
    _colorize_labels,
    _resolve_label_ids,
    generate_semantics_debug_visualizations,
)


def _segment(*, segment_id: int, label: str, mask: np.ndarray, label_id: int | None) -> SemanticSegment:
    return SemanticSegment(
        segment_id=segment_id,
        label=label,
        score=1.0,
        mask=mask,
        label_id=label_id,
        area=int(mask.sum()),
    )


def test_render_semantics_overlay_uses_label_identity_for_colors():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    settings = SemanticsVisualizationSettings(
        overlay_alpha=1.0,
        min_segment_area=1,
        show_confidence=False,
    )
    sem_a = SemanticsData(
        frame_index=0,
        frame_id="000000",
        segments=[_segment(segment_id=101, label="Road", mask=mask, label_id=7)],
        segment_ids=np.where(mask, 101, -1).astype(np.int32),
        label_ids=np.where(mask, 7, -1).astype(np.int32),
    )
    sem_b = SemanticsData(
        frame_index=1,
        frame_id="000001",
        segments=[_segment(segment_id=202, label="Road", mask=mask, label_id=7)],
        segment_ids=np.where(mask, 202, -1).astype(np.int32),
        label_ids=np.where(mask, 7, -1).astype(np.int32),
    )

    overlay_a = render_semantics_overlay(image, sem_a, settings, include_labels=False)
    overlay_b = render_semantics_overlay(image, sem_b, settings, include_labels=False)

    assert np.array_equal(overlay_a[3, 3], overlay_b[3, 3])


def test_semantics_debug_colors_fall_back_to_label_when_only_segment_ids_exist():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:5, 1:5] = True
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    sem_a = SemanticsData(
        frame_index=0,
        frame_id="000000",
        segments=[_segment(segment_id=11, label=" Road ", mask=mask, label_id=None)],
        segment_ids=np.where(mask, 11, -1).astype(np.int32),
        label_ids=None,
    )
    sem_b = SemanticsData(
        frame_index=1,
        frame_id="000001",
        segments=[_segment(segment_id=99, label="road", mask=mask, label_id=None)],
        segment_ids=np.where(mask, 99, -1).astype(np.int32),
        label_ids=None,
    )

    ids_a, _, keys_a = _resolve_label_ids(sem_a, frame, min_segment_area=1)
    ids_b, _, keys_b = _resolve_label_ids(sem_b, frame, min_segment_area=1)

    colorized_a = _colorize_labels(ids_a, id_to_palette_key=keys_a)
    colorized_b = _colorize_labels(ids_b, id_to_palette_key=keys_b)

    assert np.array_equal(colorized_a[2, 2], colorized_b[2, 2])


def test_generate_semantics_visualizations_writes_palette_manifest(tmp_path):
    store = ResourceStore("palette_manifest", root=tmp_path)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    store.save_frame(FrameData(frame_id="000000", index=0, image=frame))
    store.save_semantics2d(
        SemanticsData(
            frame_index=0,
            frame_id="000000",
            segments=[_segment(segment_id=33, label="Road", mask=mask, label_id=7)],
            segment_ids=np.where(mask, 33, -1).astype(np.int32),
            label_ids=np.where(mask, 7, -1).astype(np.int32),
        )
    )

    generated = generate_semantics_visualizations(
        store,
        SemanticsVisualizationSettings(min_segment_area=1, show_confidence=False),
    )

    manifest_path = store.visualizations_dir() / "semantics_palette.json"
    assert manifest_path in generated
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["entries"]["label_id:7"]["display_label"] == "road"


def test_generate_semantics_debug_visualizations_writes_label_based_manifest_for_segment_ids(
    tmp_path,
):
    store = ResourceStore("palette_debug_manifest", root=tmp_path)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    store.save_frame(FrameData(frame_id="000000", index=0, image=frame))
    store.save_semantics2d(
        SemanticsData(
            frame_index=0,
            frame_id="000000",
            segments=[_segment(segment_id=44, label="Road", mask=mask, label_id=None)],
            segment_ids=np.where(mask, 44, -1).astype(np.int32),
            label_ids=None,
        )
    )

    generate_semantics_debug_visualizations(
        store,
        SemanticsDebugSettings(enabled=True, max_frames=None, min_segment_area=1),
    )

    manifest = json.loads((store.visualizations_dir() / "semantics_palette.json").read_text(encoding="utf-8"))
    assert "label:road" in manifest["entries"]
