from flamemirror.ml.sandbox import Sandbox


def test_sandbox_execute_success():
    sandbox = Sandbox()
    code = "value = 10\n"
    result = sandbox.execute(code)
    assert result.success
    assert result.stderr == ""


def test_sandbox_execute_failure():
    sandbox = Sandbox()
    result = sandbox.execute("raise ValueError('boom')\n")
    assert not result.success
    assert isinstance(result.exception, Exception)


def test_sandbox_evaluate_function(tmp_path):
    sandbox = Sandbox()
    code = "def add(a, b):\n    return a + b\n"
    result = sandbox.evaluate_function(code, "add", 2, 3)
    assert result.success


def test_sandbox_evaluate_missing_function():
    sandbox = Sandbox()
    result = sandbox.evaluate_function("", "missing")
    assert not result.success
    assert "Function not defined" in result.stderr
