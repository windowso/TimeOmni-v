import json

from timeomni_v.data.dataset import TimeOmniVDataset


def test_dataset_reads_minimal_jsonl(tmp_path):
    p = tmp_path / "x.jsonl"
    row = {
        "id": "abc",
        "task": "classification",
        "video_path": "/v.mp4",
        "timeseries_path": "/t.csv",
        "prompt": "P",
        "answer": "A",
    }
    p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    ds = TimeOmniVDataset(p)
    assert len(ds) == 2
    item = ds[0]
    assert item["id"] == "abc"
    assert item["task"] == "classification"
    assert item["video_path"] == "/v.mp4"
    assert item["timeseries_path"] == "/t.csv"
    assert item["answer"] == "A"
    assert item["prompt"] == "P"
    assert item["image_path"] is None


def test_dataset_handles_missing_timeseries_path(tmp_path):
    """Baseline rows have no timeseries_path field."""
    p = tmp_path / "x.jsonl"
    row = {"id": "abc", "task": "classification",
           "video_path": "/v.mp4", "prompt": "P", "answer": "A"}
    p.write_text(json.dumps(row) + "\n")
    ds = TimeOmniVDataset(p)
    assert ds[0]["timeseries_path"] is None


def test_dataset_image_row(tmp_path):
    """Image rows surface `image_path` (list-or-string) and `task`."""
    p = tmp_path / "x.jsonl"
    row = {
        "id": "img1",
        "task": "prediction",
        "image_path": ["/a.png", "/b.png"],
        "timeseries_path": "/t.csv",
        "prompt": "P",
        "answer": "<forecast>(2024-01: 1.0)</forecast>",
    }
    p.write_text(json.dumps(row) + "\n")
    ds = TimeOmniVDataset(p)
    item = ds[0]
    assert item["image_path"] == ["/a.png", "/b.png"]
    assert item["video_path"] is None
    assert item["task"] == "prediction"
