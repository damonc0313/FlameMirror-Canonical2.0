from flamemirror.fuzzy.engine import FuzzyGuidanceEngine


def test_fuzzy_guidance_returns_advice(tmp_path):
    rules_path = tmp_path / "rules.yaml"
    content = "\n".join(
        [
            "rules:",
            "  - name: low-coverage",
            "    conditions:",
            "      coverage:",
            "        lt: 0.9",
            "    advice: Improve tests",
        ]
    )
    rules_path.write_text(content + "\n", encoding="utf-8")
    engine = FuzzyGuidanceEngine(rules_path)
    advice = engine.evaluate({"coverage": 0.5})
    assert advice[0] == "Improve tests"


def test_fuzzy_guidance_default_rule(tmp_path):
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text("rules: []\n", encoding="utf-8")
    engine = FuzzyGuidanceEngine(rules_path)
    advice = engine.evaluate({"coverage": 1.0})
    assert advice == ["No fuzzy adjustments"]
