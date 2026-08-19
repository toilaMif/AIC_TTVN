# Manifests

- `dataset.jsonl`: all audited source videos.
- `audit-report.json`: audit summary and warnings.
- `demo-10.jsonl`: frozen small demo selection.

Regenerate the audit after changing source metadata:

```powershell
uv run python -m retrieval.cli.data audit
```
