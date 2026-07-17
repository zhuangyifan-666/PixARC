from pathlib import Path

from speca_style.image_io import resumable_batch_groups
from speca_style.manifest import build_manifest


def test_resume_uses_frozen_protocol_batch_size_for_short_tail(tmp_path: Path):
    records = build_manifest(
        samples_per_class=3,
        num_classes=2,
        base_seed=10,
        split_name="tail",
        world_size=1,
        batch_size=4,
    )
    tail = [record for record in records if record.batch_group_id == "0:1"]
    assert len(tail) == 2
    metadata = {}
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()

    pending, skipped = resumable_batch_groups(
        records,
        0,
        sample_dir,
        metadata,
        manifest_sha256="manifest",
        config_hash="config",
        speca_config_hash="speca",
        checkpoint_path="checkpoint",
        checkpoint_size=1,
        method="speca",
        interval=None,
        max_order=4,
        coordinate_mode="official_nfe_index",
        protocol_batch_size=4,
    )
    assert [len(group) for group in pending] == [4, 2]
    assert skipped == []
