from pathlib import Path

from flamemirror.autonomous_agent import PlanStep
from flamemirror.generator import CodeGenerator
from flamemirror.ml.model import CodexBackend, GraphformicCoder


def test_code_generator_fallback(tmp_path):
    generator = CodeGenerator(tmp_path)
    step = PlanStep(description="Create helper", target_file=Path("helper.py"), priority=1)
    result = generator.generate(step, context_lines=["Use dataclasses"])
    assert result.target_path.exists()
    assert "Create helper" in result.code
    assert not result.used_ml


def test_code_generator_with_ml(tmp_path):
    backend = CodexBackend(api_key="dummy", enabled=True)
    model = GraphformicCoder(backend=backend)
    generator = CodeGenerator(tmp_path, ml_model=model)
    step = PlanStep(
        description="ML plan",
        target_file=Path("ml.py"),
        priority=2,
        metadata={"ast": "Module()"},
    )
    result = generator.generate(step)
    assert result.used_ml
    assert "codex_solution" in result.code
