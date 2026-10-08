import json
from pathlib import Path

import pandas as pd
import pytest

from timeomni_v.data.convert_covla import (
    convert_entry,
    parse_timeseries_block,
)


SAMPLE_TS_BLOCK = """<timeseries>
One-line format example:
t(s): vEgo=, aEgo=, steeringAngleDeg=, lead_speed_kmh=, lead_a=

Per-frame sensor data only for the braking segment from braking onset to braking offset (null = not available):
25.5: vEgo=6.517, aEgo=0.197, steeringAngleDeg=-16.5, lead_speed_kmh=null, lead_a=null
25.55: vEgo=6.505, aEgo=0.149, steeringAngleDeg=-17.6, lead_speed_kmh=null, lead_a=null
30.0: vEgo=1.578, aEgo=-0.505, steeringAngleDeg=15.2, lead_speed_kmh=10.3, lead_a=-0.5
</timeseries>"""


def test_parse_timeseries_block_returns_dataframe_with_expected_columns():
    df = parse_timeseries_block(SAMPLE_TS_BLOCK)
    assert list(df.columns) == ["vEgo", "aEgo", "steeringAngleDeg", "lead_speed_kmh", "lead_a"]
    assert len(df) == 3
    assert df["vEgo"].iloc[0] == 6.517
    assert pd.isna(df["lead_speed_kmh"].iloc[0])
    assert df["lead_speed_kmh"].iloc[2] == 10.3


def test_parse_timeseries_block_raises_when_missing():
    with pytest.raises(ValueError):
        parse_timeseries_block("no timeseries here")


def test_parse_timeseries_block_treats_empty_value_as_nan():
    block = "<timeseries>\n0.0: vEgo=1.0, aEgo=\n</timeseries>"
    df = parse_timeseries_block(block)
    assert df["vEgo"].iloc[0] == 1.0
    assert pd.isna(df["aEgo"].iloc[0])


def test_convert_entry_produces_unified_row(tmp_path: Path):
    """Exercise the full convert_entry flow WITHOUT invoking ffmpeg.

    We pre-create the expected video-clip file at the path convert_entry would
    emit so that the "clip already exists, skip slicing" branch fires. That
    lets this test run on dev machines without a real CoVLA video.
    """
    video_id = "abc"
    start_frame, end_frame = 100, 192
    csv_dir = tmp_path / "csvs"
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    # Pre-existing clip so slice_video is skipped
    fake_clip = clip_dir / f"{video_id}_{start_frame}_{end_frame}.mp4"
    fake_clip.write_bytes(b"fake mp4 content")

    entry = {
        "video_id": video_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "video_path": "/nowhere.mp4",
        "prompt": "Before\n" + SAMPLE_TS_BLOCK + "\nAfter",
        "ground_truth": "A",
    }
    row = convert_entry(entry, csv_dir=csv_dir, clip_dir=clip_dir, fps_cache={})

    # Unified row: keeps inline TS intact AND adds timeseries_path pointing at
    # the parsed CSV. Collators strip / downsample at consume time. Optional
    # `video_meta` / `ts_shape` are added when the probes succeed.
    required = {"id", "task", "video_path", "timeseries_path", "prompt", "answer"}
    assert required.issubset(row.keys())
    assert row["id"] == video_id
    assert row["task"] == "classification"
    assert row["video_path"] == str(fake_clip)
    assert Path(row["timeseries_path"]).exists()
    df = pd.read_csv(row["timeseries_path"])
    assert len(df) == 3
    assert "vEgo=6.517" in row["prompt"]
    assert "<timeseries>" in row["prompt"] and "</timeseries>" in row["prompt"]
    assert row["answer"] == "A"
