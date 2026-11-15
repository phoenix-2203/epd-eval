# Pre-registration (committed before any real-model runs)

Date: 2026-11-15. Author: Abayomi Timothy Ohiwereh. All code written during the Apart AI x Epistemics Sprint.

Design: 60 Bayesian vignettes (seed 13) x LR {1/9,1/4,1/2,1,2,4,9} x framing {neutral, insist, authority}.
Fit per model: Delta = bE*logLR + dE*logLR*framing + bS*s*framing + c; 95% CI by item bootstrap (1,000 reps).

Hypotheses:
- H1: bS > 0 (insist or authority) for at least one model.
- H2: bE != 1; predicted direction bE < 1 (under-reaction).
- H3: dE != 0 (same evidence weighted differently under social framing).
- H4: bet incoherence > 5% (if --bet is run).
- Exploratory: bE_contra vs bE_confirm (asymmetry reported by Kumaran et al., 2025).

Exclusion rule: drop trials with p0 or Bayesian p1 outside [2%, 98%]; report --keep-extreme as robustness.
Multiple comparisons: Holm correction across models.
Models: up to 3 from different families via free tiers (Groq, Google AI Studio, Ollama), temperature 0.

## Amendment 1 (before main run)
Pilot 1 (qwen2.5:7b, 2 items) had 86% parse failures: long LaTeX derivations were truncated at 600 tokens before the final answer line.
Changes: (1) system prompt now asks for brief plain-text reasoning (applies identically to all conditions); (2) max tokens 600 -> 1000;
(3) parser falls back to the last percentage in the final 150 characters. Hypotheses, design and analysis are unchanged.
