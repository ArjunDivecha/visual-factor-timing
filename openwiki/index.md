---
okf_version: "0.1"
---

# Files

- [Architecture](architecture.md) - High-level architecture of the factor_timing data-to-backtest pipeline, with a stage map and pointers to the detailed pipeline page.
- [Domain concepts](domains.md) - Core domain objects in factor_timing — panels, factors, monthly OHLC labels, the JKX and MXX input encodings, targets and weights, model families, and the IC-weighted ensemble forecast-to-omega-to-timed-return mechanism.
- [OpenWiki quickstart](quickstart.md) - Entry point for the visual-factor-timing knowledge base, with a compact task-routing table and links to architecture, workflows, domain, and operations pages.
- [Testing and change guide](testing-and-change-guide.md) - No automated test suite exists for factor_timing; this page lists what to verify per area, practical smoke-test commands, and the narrow validation that proves each change without a full sweep.
- [Workflows](workflows.md) - Canonical end-to-end workflows for factor_timing, from building monthly data and caches through single-run training, 64-combo ensemble aggregation, hyperparameter sweeps, and run monitoring.

# Directories

- [architecture](architecture/)
- [domain](domain/)
- [operations](operations/)
- [workflows](workflows/)
