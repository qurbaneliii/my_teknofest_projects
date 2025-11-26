"""Base interface for alert handlers."""

from abc import ABC, abstractmethod
from sonic.src.core.models import Track


class AlertHandler(ABC):
    """Abstract base class for alert dispatchers."""

    @abstractmethod
    def send_alert(self, track: Track) -> None:
        """Send alert for a track detection event.
        
        Args:
            track: Track object that triggered the alert
        """
        pass

    def __call__(self, track: Track) -> None:
        """Allow handler to be called directly."""
        self.send_alert(track)
