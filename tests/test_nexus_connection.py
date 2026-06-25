"""
tests/test_nexus_connection.py

Tests for validate_nexus_connection.

The function calls qnexus.login_with_credentials() then checks
qnexus.devices.get_all(...).df().  All network calls are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch
from tests.helpers import validate_nexus_connection, mod


class TestValidateNexusConnection:

    def test_returns_true_when_devices_found(self):
        """
        When the devices DataFrame is non-None and non-empty the function
        must return True, indicating a live Nexus connection.
        """
        mock_df = MagicMock()
        mock_df.__len__ = lambda self: 3

        with patch.object(mod.qnx, "login_with_credentials"):
            with patch.object(mod.qnx.devices, "get_all") as mock_get_all:
                mock_get_all.return_value.df.return_value = mock_df
                result = validate_nexus_connection(nexus_hosted=True)

        assert result is True

    def test_returns_false_when_empty_device_list(self):
        """
        An empty DataFrame (no devices returned) must yield False.
        This catches the case where credentials are valid but the
        account has no accessible devices.
        """
        mock_df = MagicMock()
        mock_df.__len__ = lambda self: 0

        with patch.object(mod.qnx, "login_with_credentials"):
            with patch.object(mod.qnx.devices, "get_all") as mock_get_all:
                mock_get_all.return_value.df.return_value = mock_df
                result = validate_nexus_connection(nexus_hosted=True)

        assert result is False

    def test_raises_connection_error_on_network_failure(self):
        """
        A ConnectionError from login must be re-raised with a descriptive
        message rather than swallowed silently.
        """
        with patch.object(mod.qnx, "login_with_credentials",
                          side_effect=ConnectionError("network down")):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                validate_nexus_connection()
