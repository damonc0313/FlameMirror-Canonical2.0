from flamemirror.validator import CodeValidator


def test_validator_accepts_valid_code(tmp_path):
    validator = CodeValidator()
    code = "def hello():\n    return 'world'\n"
    result = validator.validate(code, tmp_path / "hello.py")
    assert result.passed
    assert result.metrics["complexity"] >= 0


def test_validator_reports_errors(tmp_path):
    validator = CodeValidator()
    code = "def broken(:\n    pass"
    result = validator.validate(code, tmp_path / "broken.py")
    assert not result.passed
    assert any("SyntaxError" in diag for diag in result.diagnostics)


def test_validator_complexity_counts(tmp_path):
    validator = CodeValidator()
    code = (
        "def complicated(x):\n"
        "    for i in range(x):\n"
        "        if i % 2:\n"
        "            x += i\n"
        "    with open(__file__, 'r') as handle:\n"
        "        handle.read()\n"
        "    return x\n"
    )
    result = validator.validate(code, tmp_path / "complex.py")
    assert result.passed
    assert result.metrics["complexity"] >= 2.5
