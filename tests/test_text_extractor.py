from pathlib import Path

from tools import text_extractor


def test_extract_batch(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world", encoding="utf-8")

    count, index_path = text_extractor.extract_batch([tmp_path], tmp_path / "out")
    assert count == 1
    assert index_path.exists()
    content = index_path.read_text(encoding="utf-8")
    assert "hello world" in (tmp_path / "out" / "sample.txt").read_text(encoding="utf-8")
    assert "sample.txt" in content
