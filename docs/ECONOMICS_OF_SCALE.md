# Economics of Scale — RAG Rollout Plan

## Summary
This document lays out an "Economics of Scale" strategy for rolling your local RAG prototype into an enterprise service without runaway costs. It recommends a one-use-case-at-a-time approach (start with Jira for RTEs/POs/SMs), defines metrics to measure, sets cost triggers for architecture changes, and provides a phased rollout plan with practical guardrails.

---
## 1) Approach & Goal
- Goal: Grow from a 12-document, single-user RAG to 2.5k–5k documents and 100–150 users while controlling cost and maintaining quality.
- Strategy: Pilot per use case → instrument real metrics → decide scale model (managed tokens vs self-host) based on data.

---
## 2) Pilot use case: Jira (RTE / PO / SM)
- Scope (MVP): Ingest Jira issues, PR links, and relevant Confluence docs for 1–2 projects. Provide search, filters (project/sprint/status), and results with snippet + link.
- Pilot size: 10–20 users for 2–4 weeks, ingest last 6 months of issues (50–500 issues per project).
- Acceptance: 10 standard questions should return correct answers with citations and average user rating ≥ 80%.

---
## 3) Instrumentation & metrics (must have)
- Queries/day and unique users/day
- Prompt & response token counts per query
- LLM calls and spend per period
- Retrieval hit rate and precision@k
- User satisfaction (thumbs up/down + optional review)
- Index size (docs, chunks) and embedding update frequency

Use these to trigger architectural choices.

---
## 4) Economics model (how to decide)
- Managed (token-based): cost_per_query = (prompt_tokens/1000)*price_in + (response_tokens/1000)*price_out
- Self-hosted: cost_per_query = hourly_gpu_cost / (throughput_qps * 3600) + infra overhead
- Representative tipping point (example): if monthly queries Q satisfy
  0.018 * Q > 0.00007 * Q + $300
  → Q ≳ 17k/month (~560/day). Replace numbers with your measured metrics to compute the true breakpoint.

Key point: inference cost dominates. Embedding storage and index costs are comparatively small.

---
## 5) Architecture triggers & scaling decisions
- Managed API is fine for low-volume pilots (≤ a few hundred queries/day).
- Evaluate self-hosting (GPU instances / pooled servers) when monthly LLM spend > $2–3k or queries/day > ~500.
- Move to sharded or managed vector DB when index or query load increases beyond a single-node FAISS capacity (~200k chunks or heavy concurrent queries).

---
## 6) Cost-control & guardrails (practical)
- Per-user and per-team daily quotas and rate limits
- Caching results and embeddings to reduce repeated calls
- Use cheaper models for non-critical queries; route critical flows to higher-quality models
- Daily budget alerts and weekly cost reviews during pilot
- Use preemptible/spot instances for non-urgent inference workloads where feasible

---
## 7) Phased rollout roadmap (timeboxed)
Phase 0 — Prep (1–2 wk)
- Select pilot projects; instrument logging; create connectors for Jira.

Phase 1 — Jira MVP (2–4 wk)
- Ingest sample data; build index; launch UI for 10–20 users; collect metrics and feedback.
- Decision gate: If queries/day or monthly spend crosses thresholds, evaluate self-hosting.

Phase 2 — Team scale (4–8 wk)
- Expand to more projects and teams (500–2,000 issues); add Confluence linking, SSO, quotas, caching.

Phase 3 — Enterprise (ongoing)
- Onboard operations sources (incidents/tickets/feedback), analytics sources, and full Jira + Confluence + incidents coverage. Harden infra: autoscaling GPU pool, vector DB, SLA, governance.

---
## 8) Priority pilot deliverables
- Jira-MVP spec (ingest config, sample queries, UI screens) — highest priority
- Cost-model spreadsheet (editable; plug-in your metrics) — essential for decision gates
- Automation playbook (triggers + runbooks) for when budgets or loads exceed thresholds

---
## 9) Quick user-impact examples (high ROI queries)
- RTE / PO / SM: "Show open stories in sprint X with blockers and owners."  
- Support / SRE: "List recent incidents affecting pipeline Y and their current status."  
- Data steward: "Which docs reference PII handling for dataset Z?"
