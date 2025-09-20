import pytest

from flamemirror.ml.model import CodexBackend, GraphformicCoder


def test_codex_backend_requires_configuration():
    backend = CodexBackend(api_key=None)
    with pytest.raises(RuntimeError):
        backend.invoke("prompt", None)


def test_graphformiccoder_local_synthesis_includes_ast():
    coder = GraphformicCoder()
    code = coder.generate("do something", ast="Module(body=[])\n")
    assert "AST hint" in code
