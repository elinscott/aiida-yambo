# -*- coding: utf-8 -*-
"""Regression test for `YamboRestart._handle_parallelism_error`/`_handle_memory_error`.

These handlers append an `OMP_NUM_THREADS` prepend-text line built from
`new_resources['num_cores_per_mpiproc']`. `fix_parallelism`/`fix_memory` pass
the caller's `resources` dict straight through, so on a scheduler whose
resource class carries no `num_cores_per_mpiproc` field (e.g. an AiiDA
`hyperqueue` computer), the bare subscript raises `KeyError` and the handler
never reaches its actual job of adjusting the parallelism namelist and
retrying.

This module has no AiiDA profile available, so it does not go through a real
`WorkChain` instance or `BaseRestartWorkChain` engine machinery. It calls the
handler's undecorated function (`.__wrapped__`, exposed by the `wrapt`-based
`@process_handler` decorator) directly against a minimal stand-in for `self`
and the failed calculation, and stubs out `update_dict` (which needs a
backend-bound `orm.Dict`) since it sits downstream of the line under test.
That means this test exercises exactly the `OMP_NUM_THREADS` line and
nothing about `BaseRestartWorkChain`'s handler dispatch or the rest of
`update_dict`'s behaviour.

The negative control (`test_pre_fix_source_raises_keyerror`) loads the
handler as it existed at the commit this branch forked from, to show the
same input that now passes used to raise -- so the passing "after" tests are
not passing for an unrelated reason (e.g. a fake that never reaches the line
at all).
"""
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiida_yambo.workflows import yamborestart as yamborestart_module
from aiida_yambo.workflows.yamborestart import YamboRestart

REPO_ROOT = Path(__file__).resolve().parent.parent
PRE_FIX_REVISION = 'a95794c'  # tip of `new_cleanup` this branch forked from


class _FakeCalculation:
    """Stand-in for the failed `YamboCalculation` node."""

    def __init__(self, wrote_dbs=False):
        self.inputs = SimpleNamespace(
            parameters=SimpleNamespace(get_dict=lambda: {'variables': {}}),
        )
        self.outputs = SimpleNamespace(
            output_parameters=SimpleNamespace(get_dict=lambda: {'yambo_wrote_dbs': wrote_dbs}),
        )
        self.exit_status = 500


def _make_fake_self(resources, max_number_of_nodes=1):
    """Build a minimal stand-in for the `YamboRestart` instance."""
    options = SimpleNamespace(resources=dict(resources), prepend_text='')
    metadata = SimpleNamespace(options=options)
    ctx_inputs = SimpleNamespace(metadata=metadata, parameters=None, parent_folder=None, settings=None)
    ctx = SimpleNamespace(inputs=ctx_inputs, iteration=1)
    return SimpleNamespace(
        ctx=ctx,
        inputs=SimpleNamespace(max_number_of_nodes=max_number_of_nodes),
        report_error_handled=lambda calculation, message: None,
    )


@pytest.fixture(autouse=True)
def _stub_update_dict(monkeypatch):
    """Bypass `update_dict`, which needs a backend-bound `orm.Dict`.

    The line under test runs before `update_dict` is ever called, so
    stubbing it does not touch the behaviour this test exercises.
    """
    monkeypatch.setattr(yamborestart_module, 'update_dict', lambda _dict, *args, **kwargs: _dict)


def _load_pre_fix_module(tmp_path):
    """Import `yamborestart.py` as it read at `PRE_FIX_REVISION`, as a standalone module."""
    source = subprocess.run(
        ['git', 'show', f'{PRE_FIX_REVISION}:aiida_yambo/workflows/yamborestart.py'],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    module_path = tmp_path / 'yamborestart_pre_fix.py'
    module_path.write_text(source)
    spec = importlib.util.spec_from_file_location('yamborestart_pre_fix', module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def test_pre_fix_source_raises_keyerror(tmp_path, monkeypatch):
    """Negative control: the handler at the branch point raises for this input.

    Same fake `self`/`calculation`, same missing key, only the source
    predates the fix. If this stopped raising, the "after" tests below would
    not be discriminating between the fix and an unrelated no-op.
    """
    pre_fix_module = _load_pre_fix_module(tmp_path)
    monkeypatch.setattr(pre_fix_module, 'update_dict', lambda _dict, *args, **kwargs: _dict)

    handler = pre_fix_module.YamboRestart._handle_parallelism_error.__wrapped__
    fake_self = _make_fake_self(resources={'num_machines': 1, 'num_mpiprocs_per_machine': 1})
    calculation = _FakeCalculation()

    with pytest.raises(KeyError, match='num_cores_per_mpiproc'):
        handler(fake_self, calculation)


def test_parallelism_handler_defaults_omp_threads_when_key_absent():
    """The fixed handler falls back to 1 OMP thread instead of raising."""
    handler = YamboRestart._handle_parallelism_error.__wrapped__
    fake_self = _make_fake_self(resources={'num_machines': 1, 'num_mpiprocs_per_machine': 1})
    calculation = _FakeCalculation(wrote_dbs=False)

    handler(fake_self, calculation)

    assert fake_self.ctx.inputs.metadata.options.prepend_text == '\nexport OMP_NUM_THREADS=1'


def test_memory_handler_defaults_omp_threads_when_key_absent():
    """The fixed handler falls back to 1 OMP thread instead of raising."""
    handler = YamboRestart._handle_memory_error.__wrapped__
    fake_self = _make_fake_self(resources={'num_machines': 1, 'num_mpiprocs_per_machine': 1})
    calculation = _FakeCalculation(wrote_dbs=False)

    handler(fake_self, calculation)

    assert fake_self.ctx.inputs.metadata.options.prepend_text == '\nexport OMP_NUM_THREADS=1'
