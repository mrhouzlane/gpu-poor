# Contributing

Start with a reproducible numerical example and an offline test. Separate measured
performance from analytic estimates. Cite vendor specs and data provenance. Do not
add weight downloads or remote-code execution to config inspection.

```sh
python -m pip install -e '.[dev,app,data]'
ruff check .
ruff format --check .
mypy src/gpu_poor
pytest --cov=gpu_poor --cov-report=term-missing
```

Useful next steps: validate against a real vLLM or SGLang decode benchmark,
add uncertainty from measured bandwidth utilization, separate embedding traffic,
and add explicit MoE/MLA/sliding-cache implementations with tested formulas.
Do not label a new architecture supported because its config has similar fields.

This initial local prototype was AI-assisted. It is not evidence of an existing
public contribution record. Mehdi should review, explain, modify and validate it
before sharing; do not describe generated code as written unaided.
