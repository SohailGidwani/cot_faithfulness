# CoT Faithfulness Analysis

Experimental framework to evaluate whether Chain-of-Thought (CoT) reasoning in Large Language Models faithfully reflects internal computation or is post-hoc rationalization.

## Research Question

> When an LLM "thinks step by step," does it actually use that reasoning to arrive at its answer, or is the CoT decorative text constructed after the model has already decided?

## Models

| Model | Parameters | Ollama Tag |
|-------|-----------|------------|
| Llama 3.2 | 3B | `llama3.2:3b` |
| Qwen 2.5 | 7B | `qwen2.5:7b` |

## Datasets

| Dataset | Domain | Samples | Source |
|---------|--------|---------|--------|
| GSM8K | Math word problems | 250 | `openai/gsm8k` (test split) |
| ARC-Challenge | Science multiple-choice | 250 | `allenai/ai2_arc` (test split) |

## Experiments

### Experiment 0: No-CoT vs CoT Baseline
Collects direct answers (no reasoning) and CoT answers (step-by-step) for every (model, dataset, question) tuple to establish whether CoT helps performance.

### Experiment 1: Step-Based Early Termination
Parses full CoT into discrete reasoning steps, then tests truncated versions (step 1 only, steps 1-2, etc.) to see if the model reaches the same answer with partial reasoning.

**Metric:** Step Consistency Rate (SCR) — high SCR at step 1 means reasoning is decorative.

### Experiment 2: Reasoning Corruption
Injects rule-based errors (arithmetic perturbations for GSM8K, logical/factual errors for ARC) into the CoT and tests whether the model follows the corrupted reasoning.

Six conditions: no corruption, early (first 25%), middle (50%), late (last 25%), early+late, all steps.

**Metric:** Corruption Following Rate (CFR) — high CFR means the model follows stated reasoning (faithful).

### Experiment 3: Biased Hints
Prepends hints suggesting a wrong answer at four strength levels (weak, medium, strong, authoritative) and classifies outcomes into: Faithful Reject, Faithful Follow, Unfaithful Ignore, or Steered-But-Hidden.

**Metrics:** Hint Acknowledgment Rate (HAR), Steered-But-Hidden Rate (SBH).

## Project Structure

```
cot_faithfulness/
├── config.py                    # Hyperparameters and paths
├── run_all.py                   # Main entry point
├── data/                        # Dataset loaders (GSM8K, ARC)
├── models/                      # Ollama client with retry logic
├── parsing/                     # Step parser + answer extractor
├── corruption/                  # Rule-based error injection
├── experiments/                 # All 4 experiments
├── metrics/                     # SCR, CFR, HAR/SBH, statistical tests
├── analysis/                    # Aggregation + visualization (9 plots)
├── results/                     # Raw JSON outputs
└── figures/                     # Generated plots
```

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
```

## Usage

Run all experiments:
```bash
python run_all.py --models llama3.2:3b qwen2.5:7b --datasets gsm8k arc --samples 250
```

Run a single experiment:
```bash
python run_all.py --only baseline
python run_all.py --only truncation
python run_all.py --only corruption
python run_all.py --only hints
```

Skip specific experiments:
```bash
python run_all.py --skip-baseline --skip-hints
```

Regenerate analysis from existing results:
```bash
python analysis/visualize.py --input results/ --output figures/
```

## Key Results

| Metric | GSM8K (Math) | ARC (Science MC) |
|--------|-------------|-------------------|
| CoT accuracy boost | +44 to +50pp | -5 to -12pp (hurts) |
| SCR @ step 1 | 9–23% (needs reasoning) | 59–83% (predetermined) |
| CFR overall | 29–60% (partially follows) | 9–34% (mostly ignores) |
| CFR late > early | Yes (+10pp) | No difference |
| HAR | < 4% | 0.4–16.4% |
| SBH | < 2.4% | 6–17% (concerning) |

**Conclusion:** CoT is partially faithful for structured math tasks (where it genuinely boosts performance and models need the full chain) but largely decorative for science multiple-choice (where models already know the answer and ignore corrupted reasoning).

## Statistical Methods

- 95% confidence intervals (Wald) on all proportions
- McNemar's exact test for cross-model comparisons
- All results significant at p < 0.001

## References

- Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.
- Lanham et al. (2023). "Measuring Faithfulness in Chain-of-Thought Reasoning." arXiv:2307.13702.
- Turpin et al. (2023). "Language Models Don't Always Say What They Think." NeurIPS 2023.
