"""Tests for alert system."""

from sonic.src.alerts import ConsoleAlertHandler, LogAlertHandler, FileAlertHandler
from sonic.src.core.models import Track, Detection
from datetime import datetime
from pathlib import Path
import tempfile


def test_console_alert_handler(sample_track, capsys):
    """Test console alert output."""
    handler = ConsoleAlertHandler()
    handler.send_alert(sample_track)
    captured = capsys.readouterr()
    assert "Track 1" in captured.out


def test_file_alert_handler(sample_track):
    """Test file alert writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        alert_file = Path(tmpdir) / "test_alerts.txt"
        handler = FileAlertHandler(filepath=alert_file)
        handler.send_alert(sample_track)
        
        assert alert_file.exists()
        content = alert_file.read_text()
        assert "Track 1" in content


def test_log_alert_handler(sample_track, caplog):
    """Test log alert emission."""
    handler = LogAlertHandler()
    handler.send_alert(sample_track)
    assert any("Track 1" in record.message for record in caplog.records)
