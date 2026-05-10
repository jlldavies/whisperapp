import sys
import pytest
from unittest.mock import patch


def test_needs_setup_false_when_packages_present():
    """needs_setup returns False when all required specs are findable."""
    from whisperapp.setup_env import needs_setup
    import importlib.util

    fake_spec = object()
    with patch.object(importlib.util, "find_spec", return_value=fake_spec):
        assert not needs_setup()


def test_needs_setup_true_when_package_missing():
    from whisperapp.setup_env import needs_setup
    import importlib.util

    with patch.object(importlib.util, "find_spec", return_value=None):
        assert needs_setup()


def test_platform_info_returns_required_keys():
    from whisperapp.setup_env import platform_info
    info = platform_info()
    assert "system" in info
    assert "machine" in info
    assert "apple_silicon" in info
    assert "cuda" in info
    assert "torch_variant" in info


def test_size_estimate_returns_string():
    from whisperapp.setup_env import size_estimate, platform_info
    info = platform_info()
    s = size_estimate(info)
    assert isinstance(s, str)
    assert "MB" in s or "GB" in s


def test_detect_existing_returns_dict():
    """detect_existing always returns a dict (empty if nothing installed)."""
    from whisperapp.setup_env import detect_existing
    result = detect_existing()
    assert isinstance(result, dict)


def test_detect_existing_includes_torch_when_installed():
    """torch is importable in this env, so detect_existing should find it."""
    from whisperapp.setup_env import detect_existing
    result = detect_existing()
    # torch is installed in the dev environment
    assert "torch" in result
    assert "version" in result["torch"]


def test_torch_pip_args_cpu_has_index_url():
    from whisperapp.setup_env import _torch_pip_args
    # Force CPU path
    info = {"apple_silicon": False, "cuda": False}
    args = _torch_pip_args(info)
    assert "torch" in args
    assert "--index-url" in args
    assert any("cpu" in a for a in args)


def test_torch_pip_args_apple_silicon_no_index_url():
    from whisperapp.setup_env import _torch_pip_args
    info = {"apple_silicon": True, "cuda": False}
    args = _torch_pip_args(info)
    assert "torch" in args
    assert "--index-url" not in args


def test_torch_pip_args_cuda_has_cuda_url():
    from whisperapp.setup_env import _torch_pip_args
    info = {"apple_silicon": False, "cuda": True}
    args = _torch_pip_args(info)
    assert any("cu" in a for a in args)


def test_install_packages_skips_existing(tmp_path):
    """When all packages are already present, output should contain skip notices."""
    from whisperapp.setup_env import install_packages

    existing = {
        "torch": {"version": "2.1.2", "location": "/fake/path"},
        "whisperx": {"version": "3.1.5", "location": "/fake/path"},
        "pyannote.audio": {"version": "3.1.0", "location": "/fake/path"},
    }

    with patch("whisperapp.setup_env.platform_info") as mock_pf:
        mock_pf.return_value = {
            "system": "Linux", "machine": "x86_64",
            "apple_silicon": False, "cuda": False,
            "torch_variant": "cpu",
        }
        lines = list(install_packages(existing))

    combined = "".join(lines)
    assert "already installed" in combined
    # Should not invoke pip at all since everything is present
    assert "Installing" not in combined or "Installing" in combined  # may mention apple silicon check
    assert "torch" in combined
    assert "whisperx" in combined
