# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard that the runtime image installs every workspace package it declares.

``deploy/Dockerfile`` keeps two hand-maintained lists of workspace packages — a metadata ``COPY``
block used to resolve the frozen lock, and an explicit ``uv pip install -e`` list — that duplicate
the ``runtime-tools`` dependency group in ``pyproject.toml``.

Forgetting an entry produces **no build error**. ``COPY sources/ ./sources/`` is a wildcard, so the
source tree lands in the image and only the installed distribution is missing. NAT discovers tools
through ``nat.plugins`` entry points, which exist only for *installed* distributions, so the failure
surfaces much later as a config-load error::

    ValueError: Invalid configuration: functions: Input tag 'web_page_fetch' found using
    discriminator() does not match any of the expected tags: ...

That is exactly how `web_page_fetch` shipped broken: it was added to ``runtime-tools`` and the lock,
but not to either Dockerfile list, and five eval trials failed identically before anyone looked
inside the image.
"""

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
DOCKERFILE_PATH = REPO_ROOT / "deploy" / "Dockerfile"


def _workspace_package_paths() -> dict[str, str]:
    """Map distribution name -> repo-relative path for every workspace member."""
    paths: dict[str, str] = {}
    for parent in ("sources", "frontends"):
        for pyproject in sorted((REPO_ROOT / parent).glob("*/pyproject.toml")):
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            name = (data.get("project") or {}).get("name")
            if name:
                paths[name] = f"{parent}/{pyproject.parent.name}"
    return paths


def _runtime_tools_packages() -> list[str]:
    """Return the workspace distributions in the `runtime-tools` dependency group."""
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    group = data["dependency-groups"]["runtime-tools"]
    workspace = _workspace_package_paths()
    names = []
    for entry in group:
        if not isinstance(entry, str):
            continue  # nested {include-group = ...} entries
        name = entry.split("[")[0].split(">")[0].split("=")[0].strip()
        if name in workspace:
            names.append(name)
    return names


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    return DOCKERFILE_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize("package", _runtime_tools_packages())
def test_runtime_tools_package_is_installed_in_the_image(package, dockerfile_text):
    """Every runtime-tools workspace package must be pip-installed in deploy/Dockerfile.

    Installation is what creates the `nat.plugins` entry point. Without it the package's `_type`
    is unresolvable and any config referencing it fails to load.
    """
    path = _workspace_package_paths()[package]
    # Tolerate the quoted/extras form the Dockerfile uses for knowledge_layer:
    #   uv pip install --no-deps -e "./sources/knowledge_layer[all]"
    installed = re.search(rf'uv pip install[^\n]*-e\s+"?\./{re.escape(path)}(\[[^\]]*\])?"?', dockerfile_text)
    assert installed, (
        f"{package} is in the runtime-tools group but deploy/Dockerfile never installs "
        f"./{path}. Its nat.plugins entry point will be absent from the image and any config "
        f"using its `_type` will fail to load at startup."
    )


@pytest.mark.parametrize("package", _runtime_tools_packages())
def test_runtime_tools_package_metadata_is_copied_for_the_lock(package, dockerfile_text):
    """Its pyproject.toml must reach the image before `uv sync --frozen` resolves the lock."""
    path = _workspace_package_paths()[package]
    assert f"COPY {path}/pyproject.toml ./{path}/" in dockerfile_text, (
        f"{package} is in the runtime-tools group but its pyproject.toml is not copied before "
        f"the `uv sync --frozen ... --group runtime-tools` step in deploy/Dockerfile."
    )


def test_test_only_packages_stay_out_of_the_runtime_image(dockerfile_text):
    """Packages marked test-only in pyproject must not be installed into the image.

    The counterweight to the two tests above: they push for completeness, this one keeps the
    image from growing packages the product does not ship (`pyproject.toml`, dev group).
    """
    for package in ("aiq-gsf", "you-com"):
        path = _workspace_package_paths().get(package)
        if path is None:
            continue
        assert not re.search(rf'uv pip install[^\n]*-e\s+"?\./{re.escape(path)}(\[[^\]]*\])?"?', dockerfile_text), (
            f"{package} is marked test-only in pyproject.toml but deploy/Dockerfile installs it."
        )
