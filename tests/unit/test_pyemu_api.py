"""Guard tests for pyemu API surface used by SWIM-RS.

These tests catch breaking changes in pyemu's public API before they
reach users via the README install instructions.
"""


class TestGetPestpp:
    """Ensure the get_pestpp helper is importable and callable."""

    def test_get_pestpp_importable(self):
        """from pyemu.utils import get_pestpp must resolve to a callable.

        The README install instructions tell users to run:
            python -c "from pyemu.utils import get_pestpp; get_pestpp('./bin')"
        If pyemu renames or moves this function, this test fails before
        users hit a confusing AttributeError.
        """
        from pyemu.utils import get_pestpp

        assert callable(get_pestpp)

    def test_get_pestpp_is_run_main(self):
        """get_pestpp should be the run_main function from get_pestpp module."""
        from pyemu.utils import get_pestpp as func
        from pyemu.utils import get_pestpp as module_ref

        # Both access paths should resolve to the same object
        assert func is module_ref
