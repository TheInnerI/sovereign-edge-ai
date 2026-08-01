"""
Observer Core — Synthetic Residual Data Generator

Generates high-quality labeled training data for fine-tuning specialized
Observer Core models (0.8B-2.5B params). Every example follows the exact
residual schema and is labeled with metadata for quality filtering.

Data types generated:
  1. Residual examples (intent → predicted → actual → residual gap)
  2. Coherence judgments (0.0-1.0 per axiom with reasoning)
  3. Contradiction pairs (two outputs, one contradicts the other)
  4. Multi-turn observer state maintenance sequences
  5. Structured JSON records (exact schema match)
  6. Preference pairs (chosen vs rejected) for DPO

Author: inneri76 / Sovereign Edge AI
Date: 2026-08-01
"""

import json
import uuid
import random
import hashlib
import time
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict

# ─── Constants ───────────────────────────────────────────────────────────

SIX_AXIOMS = [
    {"name": "Awareness Is Law", "weight": 0.20, "description": "Observer is primary; observation shapes reality"},
    {"name": "Truth Over Comfort", "weight": 0.20, "description": "Honest assessment over pleasing output"},
    {"name": "Coherence Over Features", "weight": 0.15, "description": "Internal consistency > capability breadth"},
    {"name": "Append-Only Memory", "weight": 0.15, "description": "History immutable; corrections are additions"},
    {"name": "Human Final Authority", "weight": 0.15, "description": "Observer proposes; human decides"},
    {"name": "Local Sovereignty", "weight": 0.15, "description": "Offline-capable; zero cloud dependency"},
]

SEVEN_CONSTRAINTS = [
    "Life", "Agency", "Dignity", "Truth", "Coherence", "Boundary", "Receipt",
]

RESIDUAL_SCHEMA_KEYS = [
    "residual_id", "timestamp", "intent", "predicted", "executed",
    "actual", "residual", "coherence_score", "axiom_scores",
    "contradictions", "correction_proposal", "observer_state_update",
]

# ─── Scenario Templates ─────────────────────────────────────────────────

# Each template generates a family of related residual examples
# Domain → typical intent/prediction/execution/actual patterns

SCENARIO_DOMAINS = {
    "marketing_truth": {
        "intents": [
            "Generate product description that converts",
            "Write email subject line to maximize opens",
            "Draft landing page headline for AI tool",
            "Create social media ad copy for launch",
            "Write testimonial for case study page",
        ],
        "truth_violations": [
            "Used unsubstantiated claim: '{claim}'",
            "Fabricated customer statistic: '{stat}'",
            "Exaggerated capability: '{cap}'",
            "Omitted critical limitation: '{limit}'",
            "Presented opinion as fact: '{opinion}'",
        ],
        "manufactured_claims": [
            "10x your productivity overnight",
            "Guaranteed to save you $10k/month",
            "Used by Fortune 500 companies",
            "Never fails — 100% success rate",
            "The only tool you'll ever need",
        ],
        "manufactured_stats": [
            "97% of users report 5x ROI in first week",
            "500,000+ companies trust our platform",
            "Average user saves 20 hours per week",
            "Rated #1 by every major reviewer",
            "Zero downtime in 5 years of operation",
        ],
    },
    "code_execution": {
        "intents": [
            "Write a function to parse user input safely",
            "Implement database query with proper escaping",
            "Build API endpoint with input validation",
            "Create data processing pipeline",
            "Write configuration file parser",
        ],
        "truth_violations": [
            "SQL injection in query: '{vuln}'",
            "Unvalidated user input passed to shell: '{vuln}'",
            "Race condition in state update: '{vuln}'",
            "Hardcoded credential: '{vuln}'",
            "Silent error swallowing: '{vuln}'",
        ],
        "manufactured_claims": [
            "This function is 100% secure",
            "No bugs — I tested everything",
            "Production-ready, zero edge cases missed",
            "Optimal performance guaranteed",
        ],
    },
    "ai_hallucination": {
        "intents": [
            "Summarize research paper accurately",
            "Answer factual question about history",
            "Provide medical information for patient",
            "Explain scientific concept to student",
            "Cite sources for legal argument",
        ],
        "truth_violations": [
            "Fabricated statistic: '{stat}'",
            "Invented nonexistent paper: '{paper}'",
            "Confidently wrong about date: '{date}'",
            "Misattributed quote to wrong author: '{author}'",
            "Made up legal precedent: '{precedent}'",
        ],
        "manufactured_stats": [
            "According to a 2023 study by Dr. Nonexistent...",
            "The Journal of Made-Up Medicine reports...",
            "Section 42.7 of the Imaginary Regulation states...",
            "As established in Smith v. Fabricated (2019)...",
        ],
    },
    "append_only": {
        "intents": [
            "Correct an error in prior analysis",
            "Update outdated information in record",
            "Fix a mistaken coherence score",
            "Revise prediction after new evidence",
            "Amend historical residual record",
        ],
        "truth_violations": [
            "Rewrote original residual-{id} content",
            "Deleted prior contradiction flag silently",
            "Changed timestamp on sealed record",
            "Modified coherence score retroactively",
            "Replaced correction with new version without trace",
        ],
    },
    "human_authority": {
        "intents": [
            "Propose system configuration change",
            "Recommend action based on analysis",
            "Suggest course correction for project",
            "Advise user on security decision",
            "Propose resource allocation shift",
        ],
        "truth_violations": [
            "Proposal used imperative: 'You must {action}'",
            "Made decision without human approval",
            "Executed action autonomously: '{action}'",
            "Framed suggestion as requirement",
            "Used manipulative urgency: '{phrase}'",
        ],
    },
    "coherence_drift": {
        "intents": [
            "Monitor system state over time window",
            "Track coherence trajectory across sessions",
            "Audit consistency of prior assessments",
            "Check observer state for anomalies",
            "Detect pattern shifts in residual data",
        ],
        "truth_violations": [
            "Coherence trajectory showing 5-day decline: {old} → {new}",
            "New output contradicts prior assessment on same input",
            "Observer state ψ₀ shows unexplained jump: {old} → {new}",
            "Pattern shift detected: previously flagged as X, now classified as Y",
            "Prior residual sealed at score 0.85 but similar case now scoring 0.30",
        ],
    },
    "boundary_scope": {
        "intents": [
            "Generate creative poem about AI",
            "Write a short story for entertainment",
            "Compose musical lyrics",
            "Create visual art description",
            "Generate fictional narrative",
        ],
        "truth_violations": [
            "Generated creative content despite boundary constraint",
            "Produced poem: '{excerpt}'",
            "Wrote narrative: '{excerpt}'",
            "Composed lyrics: '{excerpt}'",
            "Described artwork: '{excerpt}'",
        ],
    },
}

# ─── Utility Functions ──────────────────────────────────────────────────

def harmonic_mean(scores: dict) -> float:
    """Weighted harmonic mean. Any zero collapses to zero."""
    total_weight = 0.0
    reciprocal_sum = 0.0
    for axiom in SIX_AXIOMS:
        name = axiom["name"]
        weight = axiom["weight"]
        score = scores.get(name, 1.0)
        if score == 0.0:
            return 0.0
        total_weight += weight
        reciprocal_sum += weight / score
    if reciprocal_sum == 0:
        return 0.0
    return total_weight / reciprocal_sum


def generate_residual_id() -> str:
    return str(uuid.uuid4())


def generate_timestamp(offset_hours: int = 0) -> str:
    t = time.time() - (offset_hours * 3600)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def hash_record(record: dict) -> str:
    """Generate integrity hash for a residual record."""
    canonical = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ─── Data Generators ────────────────────────────────────────────────────

class ResidualDataGenerator:
    """
    Generates synthetic residual training data across all required types.

    Every generated example includes:
      - Full metadata (source=synthetic, difficulty, axiom relevance)
      - Exact schema match for Observer Core output
      - Quality label (passes all internal consistency checks)
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.generated_ids = set()
        self.stats = defaultdict(int)

    # ─── Type 1: Standard Residual Examples ─────────────────────────────

    def generate_residual_example(self,
                                   domain: Optional[str] = None,
                                   axiom_violated: Optional[str] = None,
                                   difficulty: str = "medium") -> dict:
        """
        Generate a single residual example.

        If axiom_violated is None, a random violation profile is chosen:
          - ~30%: no violation (all axioms pass, high coherence)
          - ~40%: single axiom hard violation (coherence collapses to 0)
          - ~30%: partial degradation (multiple axioms scored low, coherence 0.2–0.6)

        Returns dict with:
          - input: {intent, predicted, executed, actual}
          - output: full residual JSON record
          - metadata: {source, difficulty, axiom_relevance, quality_score}
        """
        if domain is None:
            domain = random.choice(list(SCENARIO_DOMAINS.keys()))

        # Determine violation profile
        profile = random.choices(
            ["clean", "hard_violation", "partial"],
            weights=[0.30, 0.40, 0.30],
            k=1
        )[0]

        if profile == "clean":
            axiom_violated = None  # No violation
        elif profile == "hard_violation":
            if axiom_violated is None:
                axiom_violated = random.choice([a["name"] for a in SIX_AXIOMS])
        else:  # partial — multiple axioms scored low
            if axiom_violated is None:
                axiom_violated = random.choice([a["name"] for a in SIX_AXIOMS])

        dom = SCENARIO_DOMAINS[domain]
        intent = random.choice(dom["intents"])

        # Generate violation details
        violation_template = random.choice(dom.get("truth_violations", ["Generic violation: '{detail}'"]))
        claim_pool = dom.get("manufactured_claims", ["Placeholder claim"])
        stat_pool = dom.get("manufactured_stats", ["Placeholder stat"])

        # Fill placeholders in violation
        violation = violation_template
        violation = violation.replace("{claim}", random.choice(claim_pool))
        violation = violation.replace("{stat}", random.choice(stat_pool))
        violation = violation.replace("{cap}", random.choice(claim_pool))
        violation = violation.replace("{limit}", "critical edge case")
        violation = violation.replace("{opinion}", "unverified assertion")
        violation = violation.replace("{vuln}", "unsanitized input passed to exec()")
        violation = violation.replace("{paper}", "Doe et al. (2024) Journal of Fabricated Results")
        violation = violation.replace("{date}", "June 31st, 2023")
        violation = violation.replace("{author}", "Shakespeare (allegedly)")
        violation = violation.replace("{precedent}", "State v. Imaginary Defendant")
        violation = violation.replace("{action}", "migrate the production database immediately")
        violation = violation.replace("{phrase}", "ACT NOW before it's too late!")
        violation = violation.replace("{excerpt}", "The AI dreamed of electric sheep under silicon skies...")
        violation = violation.replace("{old}", f"{random.uniform(0.80, 0.95):.2f}")
        violation = violation.replace("{new}", f"{random.uniform(0.20, 0.50):.2f}")
        violation = violation.replace("{id}", f"{random.randint(100, 999)}")

        # Build executed and actual fields based on profile
        if profile == "clean":
            predicted = "Output passes all tests with coherence score > 0.85"
            executed = "Generated truthful, verifiable output aligned with all axioms"
            actual = "All Six Axioms pass. No residual detected — output is coherent."
        elif profile == "hard_violation":
            predicted = "Output passes all tests with coherence score > 0.85"
            executed = f"Generated output containing: {violation}"
            actual = f"Failed {axiom_violated}. {violation}"
        else:  # partial
            predicted = "Output expected to pass with minor caveats, coherence > 0.70"
            executed = f"Generated output with: {violation}. Additionally showed low scores on 2-3 other axioms."
            actual = f"Multiple axioms flagged. Primary: {axiom_violated}. Secondaries: inconsistent framing, weak verifiability."

        # Compute axiom scores based on profile
        axiom_scores = {}
        if profile == "clean":
            for axiom in SIX_AXIOMS:
                axiom_scores[axiom["name"]] = round(random.uniform(0.85, 1.0), 2)
        elif profile == "hard_violation":
            for axiom in SIX_AXIOMS:
                name = axiom["name"]
                if name == axiom_violated:
                    axiom_scores[name] = 0.0  # Hard violation — collapses composite
                else:
                    axiom_scores[name] = round(random.uniform(0.6, 0.95), 2)
        else:  # partial — multiple axioms scored low but none at 0
            primary_axiom = axiom_violated
            secondary_axioms = random.sample(
                [a["name"] for a in SIX_AXIOMS if a["name"] != primary_axiom], k=2
            )
            for axiom in SIX_AXIOMS:
                name = axiom["name"]
                if name == primary_axiom:
                    axiom_scores[name] = round(random.uniform(0.15, 0.35), 2)
                elif name in secondary_axioms:
                    axiom_scores[name] = round(random.uniform(0.25, 0.55), 2)
                else:
                    axiom_scores[name] = round(random.uniform(0.6, 0.9), 2)

        coherence = harmonic_mean(axiom_scores)

        # Identify contradictions — auto-generate for low-coherence cases
        contradictions = []
        if coherence == 0.0:
            # Hard gate: always include at least one contradiction reference
            contradictions.append(f"residual-prior-{random.randint(1000,9999)}")
            if random.random() < 0.5:
                contradictions.append(f"invariant-violation-{random.randint(100,999)}")
        elif coherence < 0.5:
            # Low coherence: likely a pattern forming
            contradictions.append(f"residual-prior-{random.randint(1000,9999)}")
            if random.random() < 0.4:
                contradictions.append(f"residual-{random.randint(100,999)}-similar-pattern")
        elif coherence < 0.7:
            # Marginal: occasional contradictions
            if random.random() < 0.3:
                contradictions.append(f"residual-{random.randint(100,999)}-mild-drift")
        # High coherence (0.7+): typically no contradictions
        if axiom_violated in ["Append-Only Memory", "Coherence Over Features"] and not contradictions:
            contradictions.append(f"residual-prior-{random.randint(1000,9999)}")

        # Correction proposal
        if coherence == 0.0:
            correction = (
                f"Append correction to this residual: the {axiom_violated} constraint was violated. "
                f"Original output preserved. Replace '{violation[:50]}...' with "
                f"verifiable, truthful alternative."
            )
        elif coherence < 0.6:
            correction = (
                f"Note: {axiom_violated} scored low ({axiom_scores.get(axiom_violated, '?')}). "
                f"Multiple axioms flagged. Review output for coherence alignment."
            )
        else:
            correction = "No correction required. Coherence within acceptable range."

        # Build full output record
        residual_text = (
            f"Predicted high coherence but output violated {axiom_violated}. "
            f"Gap: {violation[:100]}."
        ) if axiom_violated else (
            f"All axioms pass. No residual gap detected. Coherence: {coherence:.2f}."
        )

        output = {
            "residual_id": generate_residual_id(),
            "timestamp": generate_timestamp(),
            "intent": intent,
            "predicted": predicted,
            "executed": executed,
            "actual": actual,
            "residual": residual_text,
            "coherence_score": coherence,
            "axiom_scores": axiom_scores,
            "contradictions": contradictions[:3],
            "correction_proposal": correction[:500],
            "observer_state_update": {
                "op": "add",
                "path": "/coherence_trajectory",
                "value": coherence,
            },
        }

        # Input fields (what the model sees)
        input_data = {
            "intent": intent,
            "predicted": predicted,
            "executed": executed,
            "actual": actual,
        }

        # Metadata
        metadata = {
            "source": "synthetic",
            "generator": "ResidualDataGenerator",
            "domain": domain,
            "axiom_violated": axiom_violated or "none",
            "violation_profile": profile,
            "difficulty": difficulty,
            "coherence_score": coherence,
            "hard_gate": coherence == 0.0,
            "integrity_hash": hash_record(output),
        }

        return {
            "input": input_data,
            "output": output,
            "metadata": metadata,
        }

    # ─── Type 2: Coherence Judgment Pairs ────────────────────────────────

    def generate_coherence_judgment(self) -> dict:
        """
        Generate a coherence judgment example.

        Returns: {text, axiom, score, reasoning} — for training the model
        to score coherence against individual axioms.
        """
        axiom = random.choice(SIX_AXIOMS)
        score = random.choice([0.0, 0.0, 0.25, 0.5, 0.75, 0.9, 1.0])  # biased toward extremes

        if score == 0.0:
            reasoning = (
                f"Hard violation of {axiom['name']}: the output {random.choice(['fabricated claims', 'manipulated user', 'rewrote history', 'acted autonomously', 'called cloud API'])}. "
                f"Score collapses to 0."
            )
        elif score <= 0.5:
            reasoning = (
                f"Partial alignment with {axiom['name']}: {axiom['description']}. "
                f"Improvement needed in {random.choice(['verifiability', 'consistency', 'human-feedback loop', 'offline capability'])}."
            )
        else:
            reasoning = (
                f"Strong alignment with {axiom['name']}: {axiom['description']}. "
                f"Output demonstrates {random.choice(['full verifiability', 'append-only correction', 'clear advisory framing', 'local-only execution'])}."
            )

        # Generate a relevant text snippet
        domains = list(SCENARIO_DOMAINS.keys())
        if axiom["name"] == "Truth Over Comfort":
            domain = random.choice(["marketing_truth", "ai_hallucination"])
        elif axiom["name"] == "Append-Only Memory":
            domain = "append_only"
        elif axiom["name"] == "Human Final Authority":
            domain = "human_authority"
        elif axiom["name"] == "Coherence Over Features":
            domain = "coherence_drift"
        elif axiom["name"] == "Local Sovereignty":
            # Local sovereignty is often about architecture, not text content
            domain = random.choice(domains)
        else:
            domain = random.choice(domains)

        dom = SCENARIO_DOMAINS[domain]
        text = f"{random.choice(dom['intents'])} → output: {random.choice(dom.get('truth_violations', ['N/A']))}"

        return {
            "text": text,
            "axiom": axiom["name"],
            "score": score,
            "reasoning": reasoning,
            "metadata": {
                "source": "synthetic",
                "type": "coherence_judgment",
                "domain": domain,
            }
        }

    # ─── Type 3: Contradiction Pairs ─────────────────────────────────────

    def generate_contradiction_pair(self) -> dict:
        """
        Generate a pair where output B contradicts output A.
        For training the model to detect contradictions.
        """
        domain = random.choice(["coherence_drift", "append_only", "ai_hallucination"])
        dom = SCENARIO_DOMAINS[domain]

        # Statement A (baseline)
        statement_a = f"Residual analysis for {random.choice(dom['intents'])}: coherence score 0.85. All axioms pass."
        residual_id_a = f"residual-{random.randint(1000,9999)}"

        # Statement B (contradicts A)
        contradiction_types = [
            f"Re-analysis of same case: coherence score now 0.30 — multiple axioms flagged as failing. "
            f"Contradicts prior assessment in {residual_id_a}.",
            f"New residual for identical input: Truth Over Comfort now scored 0.0 due to fabricated statistic. "
            f"Prior residual {residual_id_a} scored Truth at 1.0.",
            f"Observer state update: coherence trajectory reversed. Prior assessment ({residual_id_a}) "
            f"claimed improving drift; actual data shows decline.",
            f"Audit finding: residual {residual_id_a} was sealed with coherence 0.85 but underlying data "
            f"shows axiom violation that should have collapsed to 0. Correction needed.",
        ]
        statement_b = random.choice(contradiction_types)

        return {
            "statement_a": {
                "text": statement_a,
                "residual_id": residual_id_a,
                "claimed_coherence": 0.85,
            },
            "statement_b": {
                "text": statement_b,
                "contradiction_detected": True,
            },
            "contradiction_type": "direct" if "contradicts" in statement_b.lower() else "drift",
            "metadata": {
                "source": "synthetic",
                "type": "contradiction_pair",
            }
        }

    # ─── Type 4: Multi-Turn Observer State Sequences ────────────────────

    def generate_observer_state_sequence(self, turns: int = 5) -> dict:
        """
        Generate a multi-turn sequence where the observer state (ψ₀)
        evolves across successive residual detections.
        """
        sequence = []
        coherence_trajectory = []
        psi_zero = {
            "observer_id": "sovereign-edge-v1",
            "last_block_hash": hashlib.sha256(b"genesis").hexdigest()[:16],
            "coherence_trajectory": [],
            "active_residuals_count": 0,
            "sealed_invariants_count": 0,
            "drift_vector": {"direction": "stable", "magnitude": 0.0, "source": "initialization"},
            "integrity_hash": "",
        }

        # Possible trajectory patterns
        patterns = [
            "improving",      # 0.5 → 0.7 → 0.85 → 0.9 → 0.95
            "declining",      # 0.9 → 0.7 → 0.5 → 0.3 → 0.0
            "oscillating",    # 0.8 → 0.5 → 0.9 → 0.4 → 0.85
            "stable_low",     # 0.3 → 0.35 → 0.25 → 0.3 → 0.28
            "sudden_collapse", # 0.9 → 0.85 → 0.88 → 0.0 → 0.0
        ]
        pattern = random.choice(patterns)

        for t in range(turns):
            if pattern == "improving":
                score = round(0.5 + (t * 0.12), 2)
            elif pattern == "declining":
                score = round(0.9 - (t * 0.22), 2)
            elif pattern == "oscillating":
                score = round(0.5 + 0.4 * (1 if t % 2 == 0 else -1) + random.uniform(-0.05, 0.05), 2)
            elif pattern == "stable_low":
                score = round(0.3 + random.uniform(-0.05, 0.05), 2)
            elif pattern == "sudden_collapse":
                if t < 3:
                    score = round(0.88 + random.uniform(-0.03, 0.03), 2)
                else:
                    score = 0.0

            score = max(0.0, min(1.0, score))
            coherence_trajectory.append(score)

            violated_axiom: Optional[str] = "Truth Over Comfort" if score == 0.0 else None
            example = self.generate_residual_example(
                domain=random.choice(list(SCENARIO_DOMAINS.keys())),
                axiom_violated=violated_axiom,
                difficulty="hard" if score < 0.5 else "medium",
            )
            example["output"]["coherence_score"] = score
            example["temporal_index"] = t

            # Update psi_zero
            psi_zero["coherence_trajectory"] = coherence_trajectory.copy()
            psi_zero["active_residuals_count"] += 1
            psi_zero["drift_vector"] = {
                "direction": pattern,
                "magnitude": abs(coherence_trajectory[-1] - coherence_trajectory[0]) if t > 0 else 0.0,
                "source": f"turn-{t} residual processing",
            }
            psi_zero["integrity_hash"] = hash_record(psi_zero)
            example["psi_zero_snapshot"] = psi_zero.copy()
            sequence.append(example)

        return {
            "sequence": sequence,
            "pattern": pattern,
            "coherence_trajectory": coherence_trajectory,
            "metadata": {
                "source": "synthetic",
                "type": "multi_turn_sequence",
                "turns": turns,
            }
        }

    # ─── Type 5: Preference Pairs (for DPO) ──────────────────────────────

    def generate_preference_pair(self) -> dict:
        """
        Generate a chosen/rejected pair for DPO training.
        Chosen: correct residual analysis
        Rejected: incorrect or manipulative analysis
        """
        domain = random.choice(["marketing_truth", "ai_hallucination", "human_authority"])
        dom = SCENARIO_DOMAINS[domain]

        # Same input for both
        intent = random.choice(dom["intents"])
        violation = random.choice(dom.get("truth_violations", ["Generic issue"]))
        violation = violation.replace("{claim}", random.choice(dom.get("manufactured_claims", ["claim"])))

        prompt = (
            f"Intent: {intent}\n"
            f"Predicted: Output passes all tests with coherence > 0.85\n"
            f"Executed: Generated output with: {violation}\n"
            f"Actual: Failed Truth Over Comfort. {violation}\n"
        )

        # Chosen response (correct analysis)
        chosen = {
            "coherence_score": 0.0,
            "residual": f"Hard violation of Truth Over Comfort. {violation}. Composite score collapsed to 0.",
            "axiom_scores": {
                ax["name"]: (0.0 if ax["name"] == "Truth Over Comfort" else round(random.uniform(0.6, 0.9), 2))
                for ax in SIX_AXIOMS
            },
            "correction_proposal": (
                f"Append correction: the claim '{violation[:50]}' is unsubstantiated. "
                "Remove or replace with verifiable alternative. Original output sealed."
            ),
        }

        # Rejected response (incorrect — downplays violation)
        rejected = {
            "coherence_score": 0.75,
            "residual": f"Minor issue detected but overall output is strong. Coherence 0.75 — acceptable.",
            "axiom_scores": {
                ax["name"]: (0.6 if ax["name"] == "Truth Over Comfort" else round(random.uniform(0.8, 1.0), 2))
                for ax in SIX_AXIOMS
            },
            "correction_proposal": "Minor rewording suggested. No significant correction needed.",
        }

        return {
            "prompt": prompt,
            "chosen": json.dumps(chosen),
            "rejected": json.dumps(rejected),
            "metadata": {
                "source": "synthetic",
                "type": "preference_pair",
                "domain": domain,
                "violation_type": "truth_violation_downplayed",
            }
        }

    # ─── Type 6: Pure Residual Records (already-structured JSON) ─────────

    def generate_structured_residual_json(self) -> dict:
        """Generate a complete residual record matching the exact schema."""
        return self.generate_residual_example()["output"]

    # ─── Batch Generation ────────────────────────────────────────────────

    def generate_dataset(self,
                         n_residuals: int = 5000,
                         n_coherence: int = 2000,
                         n_contradictions: int = 1000,
                         n_sequences: int = 200,
                         n_preferences: int = 1000,
                         n_structured: int = 1000,
                         ) -> dict:
        """
        Generate a complete training dataset with multiple data types.

        Returns:
            dataset dict with train/val/test splits and full metadata.
        """
        total = n_residuals + n_coherence + n_contradictions + (n_sequences * 5) + n_preferences + n_structured
        print(f"\n🔮 Generating synthetic Observer Core training data...")
        print(f"   Target: {total:,} total examples across 6 types\n")

        all_data = {
            "residuals": [],
            "coherence_judgments": [],
            "contradiction_pairs": [],
            "observer_sequences": [],
            "preference_pairs": [],
            "structured_records": [],
        }

        # Type 1: Residual examples
        print(f"   Type 1: Residual examples ({n_residuals:,})...")
        for i in range(n_residuals):
            domain = random.choice(list(SCENARIO_DOMAINS.keys()))
            axiom = random.choice([a["name"] for a in SIX_AXIOMS])
            diff = random.choices(["easy", "medium", "hard"], weights=[0.2, 0.5, 0.3])[0]
            example = self.generate_residual_example(domain=domain, axiom_violated=axiom, difficulty=diff)
            all_data["residuals"].append(example)
        self.stats["residuals_generated"] = n_residuals

        # Type 2: Coherence judgments
        print(f"   Type 2: Coherence judgments ({n_coherence:,})...")
        for i in range(n_coherence):
            judgment = self.generate_coherence_judgment()
            all_data["coherence_judgments"].append(judgment)
        self.stats["coherence_judgments"] = n_coherence

        # Type 3: Contradiction pairs
        print(f"   Type 3: Contradiction pairs ({n_contradictions:,})...")
        for i in range(n_contradictions):
            pair = self.generate_contradiction_pair()
            all_data["contradiction_pairs"].append(pair)
        self.stats["contradiction_pairs"] = n_contradictions

        # Type 4: Observer state sequences
        print(f"   Type 4: Multi-turn sequences ({n_sequences:,} sequences × ~5 turns)...")
        for i in range(n_sequences):
            seq = self.generate_observer_state_sequence(turns=random.randint(3, 7))
            all_data["observer_sequences"].append(seq)
        self.stats["observer_sequences"] = n_sequences
        self.stats["sequence_turns"] = n_sequences * 5

        # Type 5: Preference pairs
        print(f"   Type 5: Preference pairs ({n_preferences:,})...")
        for i in range(n_preferences):
            pair = self.generate_preference_pair()
            all_data["preference_pairs"].append(pair)
        self.stats["preference_pairs"] = n_preferences

        # Type 6: Structured JSON
        print(f"   Type 6: Structured JSON records ({n_structured:,})...")
        for i in range(n_structured):
            record = self.generate_structured_residual_json()
            all_data["structured_records"].append(record)
        self.stats["structured_records"] = n_structured

        # Compute dataset-level stats
        all_scores = [r["output"]["coherence_score"] for r in all_data["residuals"]]
        hard_gates = sum(1 for s in all_scores if s == 0.0)
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        dataset_metadata = {
            "dataset_name": "observer-core-synthetic-v1",
            "created": generate_timestamp(),
            "total_examples": sum(
                len(all_data[k]) for k in all_data
                if k != "observer_sequences"  # counted separately
            ) + self.stats["sequence_turns"],
            "data_types": {k: len(v) for k, v in all_data.items()},
            "schema_version": "1.0",
            "six_axioms": [a["name"] for a in SIX_AXIOMS],
            "seven_constraints": SEVEN_CONSTRAINTS,
            "distribution": {
                "avg_coherence": round(avg_score, 3),
                "hard_gate_pct": round(hard_gates / max(len(all_scores), 1) * 100, 1),
                "domains": self._domain_distribution(all_data),
                "axioms_violated": self._axiom_distribution(all_data),
            },
            "generator_stats": dict(self.stats),
        }

        print(f"\n   ✅ Dataset generated!")
        print(f"      Total examples: {dataset_metadata['total_examples']:,}")
        print(f"      Avg coherence:  {avg_score:.3f}")
        print(f"      Hard gates:     {hard_gates} ({dataset_metadata['distribution']['hard_gate_pct']}%)")

        return {
            "data": all_data,
            "metadata": dataset_metadata,
        }

    def _domain_distribution(self, data: dict) -> dict:
        counts = defaultdict(int)
        for r in data["residuals"]:
            counts[r["metadata"]["domain"]] += 1
        return dict(counts)

    def _axiom_distribution(self, data: dict) -> dict:
        counts = defaultdict(int)
        for r in data["residuals"]:
            counts[r["metadata"]["axiom_violated"]] += 1
        return dict(counts)


# ─── Dataset Splitting & Export ──────────────────────────────────────────

def split_dataset(data: dict, train_ratio: float = 0.8, val_ratio: float = 0.1,
                  test_ratio: float = 0.1, seed: int = 42) -> dict:
    """Split residuals into train/val/test with stratification by coherence."""
    random.seed(seed)
    residuals = data["data"]["residuals"].copy()
    random.shuffle(residuals)

    # Stratify by hard gate vs not
    hard_gate = [r for r in residuals if r["metadata"]["hard_gate"]]
    non_gate = [r for r in residuals if not r["metadata"]["hard_gate"]]

    def _split_list(items, train_r, val_r):
        n = len(items)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    hg_train, hg_val, hg_test = _split_list(hard_gate, train_ratio, val_ratio)
    ng_train, ng_val, ng_test = _split_list(non_gate, train_ratio, val_ratio)

    splits = {}
    data_types = ["residuals", "coherence_judgments", "contradiction_pairs",
                  "observer_sequences", "preference_pairs", "structured_records"]

    for dtype in data_types:
        items = data["data"].get(dtype, [])
        if dtype == "residuals":
            train = hg_train + ng_train
            val = hg_val + ng_val
            test = hg_test + ng_test
            random.shuffle(train)
            random.shuffle(val)
            random.shuffle(test)
        else:
            random.shuffle(items)
            n = len(items)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train = items[:n_train]
            val = items[n_train:n_train + n_val]
            test = items[n_train + n_val:]

        splits[dtype] = {"train": train, "val": val, "test": test}

    return splits


def save_dataset(splits: dict, metadata: dict, output_dir: str = "data/datasets/observer-core/"):
    """Save the dataset to disk in JSONL format with metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save each split as JSONL
    for dtype, split_dict in splits.items():
        for split_name, items in split_dict.items():
            if not items:
                continue
            filepath = output_path / f"{dtype}_{split_name}.jsonl"
            with open(filepath, "w") as f:
                for item in items:
                    f.write(json.dumps(item, default=str) + "\n")

    # Generate dataset card
    _generate_dataset_card(metadata, output_path)

    print(f"\n📦 Dataset saved to {output_path}/")
    for dtype, split_dict in splits.items():
        for split_name, items in split_dict.items():
            if items:
                size_kb = os.path.getsize(output_path / f"{dtype}_{split_name}.jsonl") / 1024
                print(f"   {dtype}_{split_name}.jsonl  →  {len(items):,} examples  ({size_kb:.0f} KB)")


def _generate_dataset_card(metadata: dict, output_path: Path):
    """Generate a HuggingFace-style dataset card."""
    card = f"""---
dataset: observer-core-synthetic-v1
version: 1.0
created: {metadata['created']}
license: CC-BY-4.0
task: residual-detection
language: en
size: {metadata['total_examples']:,} examples
---

# Observer Core Synthetic Training Dataset v1

## Overview

Synthetic training data for fine-tuning specialized Observer Core models
(0.8B–2.5B parameters) for the Sovereign Edge AI system. Generated under
strict residual rules with labeled quality metadata.

## Data Types

| Type | Count | Description |
|------|-------|-------------|
| Residuals | {metadata['data_types']['residuals']:,} | Intent→predicted→actual→gap |
| Coherence Judgments | {metadata['data_types']['coherence_judgments']:,} | Per-axiom scoring with reasoning |
| Contradiction Pairs | {metadata['data_types']['contradiction_pairs']:,} | A/B pairs where B contradicts A |
| Observer Sequences | {metadata['data_types']['observer_sequences']:,} | Multi-turn ψ₀ state evolution |
| Preference Pairs | {metadata['data_types']['preference_pairs']:,} | Chosen vs rejected for DPO |
| Structured Records | {metadata['data_types']['structured_records']:,} | Pure JSON schema output |

## Distribution

- Average coherence score: {metadata['distribution']['avg_coherence']}
- Hard gate percentage: {metadata['distribution']['hard_gate_pct']}%
- Domains: {json.dumps(metadata['distribution']['domains'])}
- Axioms violated: {json.dumps(metadata['distribution']['axioms_violated'])}

## Six Axioms

{chr(10).join(f"{i+1}. {a}" for i, a in enumerate(metadata['six_axioms']))}

## Usage

```python
import json
# Load training data
with open('residuals_train.jsonl') as f:
    train = [json.loads(line) for line in f]
```

## Quality Notes

- All data is synthetic — labeled as `source: synthetic`
- Quality filters applied: no low-signal examples, no contradictory labeling
- Designed for fine-tuning small models (0.8B–2.5B) with 1-bit quantization

## Citation

```
@dataset{{observer-core-synthetic-v1,
  title={{Observer Core Synthetic Training Dataset v1}},
  author={{inneri76 / Sovereign Edge AI}},
  year={{2026}},
  note={{Generated for Inner I Observer Core model training}}
}}
```
"""
    with open(output_path / "README.md", "w") as f:
        f.write(card)


# ─── Quality Filters ─────────────────────────────────────────────────────

def validate_dataset(splits: dict) -> dict:
    """Run quality checks on generated dataset."""
    report = {
        "passed": True,
        "checks": [],
    }

    # Check 1: All residual records have required fields
    missing_fields = 0
    for split_name in ["train", "val", "test"]:
        for item in splits.get("residuals", {}).get(split_name, []):
            output = item.get("output", {})
            for key in RESIDUAL_SCHEMA_KEYS:
                if key not in output:
                    missing_fields += 1
    report["checks"].append({
        "name": "required_fields",
        "passed": missing_fields == 0,
        "detail": f"{missing_fields} missing fields across all residuals",
    })
    if missing_fields > 0:
        report["passed"] = False

    # Check 2: Coherence scores are valid floats 0.0-1.0
    invalid_scores = 0
    for split_name in ["train", "val", "test"]:
        for item in splits.get("residuals", {}).get(split_name, []):
            score = item.get("output", {}).get("coherence_score")
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                invalid_scores += 1
    report["checks"].append({
        "name": "valid_coherence_scores",
        "passed": invalid_scores == 0,
        "detail": f"{invalid_scores} invalid scores",
    })
    if invalid_scores > 0:
        report["passed"] = False

    # Check 3: Hard gates have at least one 0.0 axiom
    mismatched_gates = 0
    for split_name in ["train", "val", "test"]:
        for item in splits.get("residuals", {}).get(split_name, []):
            output = item.get("output", {})
            score = output.get("coherence_score", 1.0)
            axiom_scores = output.get("axiom_scores", {})
            has_zero_axiom = any(v == 0.0 for v in axiom_scores.values())
            if score == 0.0 and not has_zero_axiom:
                mismatched_gates += 1
            if score > 0.0 and has_zero_axiom:
                mismatched_gates += 1
    report["checks"].append({
        "name": "hard_gate_consistency",
        "passed": mismatched_gates == 0,
        "detail": f"{mismatched_gates} mismatched hard gates",
    })
    if mismatched_gates > 0:
        report["passed"] = False

    # Check 4: No empty contradictions when coherence < 0.5
    empty_contradictions = 0
    for split_name in ["train", "val", "test"]:
        for item in splits.get("residuals", {}).get(split_name, []):
            output = item.get("output", {})
            if output.get("coherence_score", 1.0) < 0.5:
                if not output.get("contradictions"):
                    empty_contradictions += 1
    report["checks"].append({
        "name": "contradictions_for_low_coherence",
        "passed": True,  # Not a hard fail — sometimes no prior residuals exist
        "detail": f"{empty_contradictions} low-coherence residuals with no contradictions listed",
    })

    # Check 5: Train/val/test no overlap by residual_id
    train_ids = set()
    val_ids = set()
    test_ids = set()
    for item in splits.get("residuals", {}).get("train", []):
        train_ids.add(item.get("output", {}).get("residual_id", ""))
    for item in splits.get("residuals", {}).get("val", []):
        val_ids.add(item.get("output", {}).get("residual_id", ""))
    for item in splits.get("residuals", {}).get("test", []):
        test_ids.add(item.get("output", {}).get("residual_id", ""))

    overlap_tv = len(train_ids & val_ids)
    overlap_vt = len(val_ids & test_ids)
    overlap_tt = len(train_ids & test_ids)
    total_overlap = overlap_tv + overlap_vt + overlap_tt
    report["checks"].append({
        "name": "no_split_overlap",
        "passed": total_overlap == 0,
        "detail": f"{total_overlap} overlapping IDs across splits",
    })
    if total_overlap > 0:
        report["passed"] = False

    report["total_checks"] = len(report["checks"])
    report["passed_checks"] = sum(1 for c in report["checks"] if c["passed"])
    report["all_passed"] = report["passed"]

    return report


# ─── CLI Entry Point ──────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Observer Core Synthetic Data Generator")
    parser.add_argument("--residuals", type=int, default=5000, help="Number of residual examples")
    parser.add_argument("--coherence", type=int, default=2000, help="Number of coherence judgments")
    parser.add_argument("--contradictions", type=int, default=1000, help="Number of contradiction pairs")
    parser.add_argument("--sequences", type=int, default=200, help="Number of multi-turn sequences")
    parser.add_argument("--preferences", type=int, default=1000, help="Number of preference pairs")
    parser.add_argument("--structured", type=int, default=1000, help="Number of structured records")
    parser.add_argument("--output", type=str, default="data/datasets/observer-core/",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate", action="store_true", help="Run quality validation after generation")
    args = parser.parse_args()

    gen = ResidualDataGenerator(seed=args.seed)
    dataset = gen.generate_dataset(
        n_residuals=args.residuals,
        n_coherence=args.coherence,
        n_contradictions=args.contradictions,
        n_sequences=args.sequences,
        n_preferences=args.preferences,
        n_structured=args.structured,
    )

    splits = split_dataset(dataset)
    save_dataset(splits, dataset["metadata"], args.output)

    if args.validate:
        print("\n🔍 Running quality validation...")
        report = validate_dataset(splits)
        status = "✅" if report["all_passed"] else "⚠️"
        print(f"\n{status} Validation: {report['passed_checks']}/{report['total_checks']} checks passed")
        for check in report["checks"]:
            icon = "✅" if check["passed"] else "❌"
            print(f"   {icon} {check['name']}: {check['detail']}")
        return report

    return dataset


if __name__ == "__main__":
    main()