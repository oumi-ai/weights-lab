"""Optional-dependency markers must evaluate on any Linux kernel string, as pip does.

pip's vendored ``packaging`` evaluates every clause of a marker and compares
``platform_release`` as a PEP 440 version, so it raises ``InvalidVersion`` on kernel
strings such as ``4.19.0-gvisor``. Current ``packaging`` falls back to comparing strings,
so evaluating alone cannot catch that; the test also refuses version comparisons on the
kernel-dependent keys.
"""

import re
import tomllib
from pathlib import Path

import pytest

requirements = pytest.importorskip("packaging.requirements")
version = pytest.importorskip("packaging.version")

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ["4.19.0-gvisor", "5.15.0-105-generic", "6.8.0-1017-azure"]
ORDERING = re.compile(r"\b(platform_release|platform_version)\s*(<=|>=|<|>|~=|===)")


def extra_requirements():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    return [(extra, text) for extra, texts in extras.items() for text in texts]


def test_kernel_strings_are_not_pep_440_versions():
    for kernel in KERNELS:
        with pytest.raises(version.InvalidVersion):
            version.Version(kernel)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("extra,text", extra_requirements())
def test_extra_markers_do_not_version_compare_kernel_strings(extra, text, kernel):
    marker = requirements.Requirement(text).marker
    if marker is None:
        return
    assert not ORDERING.search(str(marker)), f"{extra}: {text} orders {kernel}-like strings"
    marker.evaluate({"sys_platform": "linux", "platform_release": kernel})
