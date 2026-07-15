import os
import unittest
from unittest.mock import patch

from poker44.validator.forward import _finalize_provider_cycle
from poker44.validator.runtime_provider import ProviderRuntimeConfig


class _DummyProvider:
    def __init__(self) -> None:
        self.mark_calls = 0

    def mark_last_batch_evaluated(self) -> None:
        self.mark_calls += 1


class _DummyValidator:
    def __init__(self) -> None:
        self.provider = _DummyProvider()


class ProviderRuntimeGuardTests(unittest.TestCase):
    def test_defaults_to_public_eval_api_base_url(self):
        with patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            cfg = ProviderRuntimeConfig.from_env(default_validator_id="validator_hotkey")

        self.assertEqual(cfg.api_base_url, "https://api.poker44.net")
        self.assertEqual(cfg.internal_secret, "")
        self.assertEqual(cfg.validator_id, "validator_hotkey")

    def test_rejects_placeholder_internal_secret(self):
        with patch.dict(
            os.environ,
            {
                "POKER44_EVAL_API_BASE_URL": "http://127.0.0.1:3001",
                "POKER44_PROVIDER_INTERNAL_SECRET": "force-start-secret",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "placeholder"):
                ProviderRuntimeConfig.from_env(default_validator_id="validator_hotkey")

    def test_allows_missing_internal_secret_for_signed_validator_auth(self):
        with patch.dict(
            os.environ,
            {
                "POKER44_EVAL_API_BASE_URL": "http://127.0.0.1:3001",
            },
            clear=False,
        ):
            cfg = ProviderRuntimeConfig.from_env(default_validator_id="validator_hotkey")

        self.assertEqual(cfg.api_base_url, "http://127.0.0.1:3001")
        self.assertEqual(cfg.internal_secret, "")
        self.assertEqual(cfg.validator_id, "validator_hotkey")
        self.assertEqual(cfg.request_timeout_seconds, 60)

    def test_accepts_real_internal_secret(self):
        with patch.dict(
            os.environ,
            {
                "POKER44_EVAL_API_BASE_URL": "http://127.0.0.1:3001",
                "POKER44_PROVIDER_INTERNAL_SECRET": "real-secret-value",
            },
            clear=False,
        ):
            cfg = ProviderRuntimeConfig.from_env(default_validator_id="validator_hotkey")

        self.assertEqual(cfg.api_base_url, "http://127.0.0.1:3001")
        self.assertEqual(cfg.internal_secret, "real-secret-value")
        self.assertEqual(cfg.validator_id, "validator_hotkey")
        self.assertEqual(cfg.request_timeout_seconds, 60)

    def test_provider_cycle_finalization_requires_completed_evaluation(self):
        validator = _DummyValidator()

        _finalize_provider_cycle(validator, evaluation_completed=False)
        self.assertEqual(validator.provider.mark_calls, 0)

        _finalize_provider_cycle(validator, evaluation_completed=True)
        self.assertEqual(validator.provider.mark_calls, 1)


if __name__ == "__main__":
    unittest.main()
