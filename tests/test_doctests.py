"""Run the doctest examples embedded in the native extension module.

The pure-Python wrapper (``convolve_rs/__init__.py``) is covered by pytest's
``--doctest-modules``, but that cannot collect from the compiled
``convolve_rs._convolve_rs`` extension, so its docstrings are checked here
explicitly. ``module=False`` disables ``DocTestFinder``'s module-membership
check, which would otherwise skip PyO3 objects (their ``__module__`` is not
set to the extension module).
"""

from __future__ import annotations

import doctest

import convolve_rs._convolve_rs


def test_native_module_doctests() -> None:
    finder = doctest.DocTestFinder(exclude_empty=True)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    tests = finder.find(convolve_rs._convolve_rs, module=False)
    assert tests, "no doctests found in convolve_rs._convolve_rs"

    for test in tests:
        runner.run(test)

    results = runner.summarize(verbose=False)
    assert results.attempted > 0, "doctests were found but none ran"
    assert results.failed == 0, f"{results.failed} doctest(s) failed"
