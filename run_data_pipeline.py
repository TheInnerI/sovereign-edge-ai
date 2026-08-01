#!/usr/bin/env python3
"""
Sovereign Edge AI — Complete Data Pipeline

Generates synthetic residual data, ingests real datasets, merges them,
runs quality validation, and produces a ready-to-train dataset.

Usage:
    python run_data_pipeline.py --full         # 11k synthetic + real, full dataset
    python run_data_pipeline.py --quick        # 1k synthetic + real, for testing
    python run_data_pipeline.py --validate     # Validate existing dataset
    python run_data_pipeline.py --stats        # Show dataset statistics
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from src.data.generator import (
    ResidualDataGenerator, split_dataset, save_dataset, validate_dataset,
)
from src.data.ingest_real import ingest_all_real_data, save_real_dataset


OUTPUT_DIR = "data/datasets/observer-core/"


def run_full_pipeline(seed: int = 42):
    """Generate full 11k synthetic + real dataset."""
    print("=" * 60)
    print("🔮 Sovereign Edge AI — Full Data Pipeline")
    print("=" * 60)
    
    # Step 1: Synthetic data
    print("\n📊 STEP 1/4: Generating synthetic residual data (11,000 examples)...")
    gen = ResidualDataGenerator(seed=seed)
    dataset = gen.generate_dataset(
        n_residuals=5000,
        n_coherence=2000,
        n_contradictions=1000,
        n_sequences=200,
        n_preferences=1000,
        n_structured=1000,
    )
    
    # Step 2: Real data
    print("\n📊 STEP 2/4: Ingesting real residual datasets...")
    real_data = ingest_all_real_data()
    
    # Step 3: Merge
    print("\n📊 STEP 3/4: Merging synthetic + real data...")
    dataset["data"]["residuals"].extend(real_data["data"]["residuals"])
    dataset["metadata"]["real_examples"] = len(real_data["data"]["residuals"])
    dataset["metadata"]["total_examples"] += len(real_data["data"]["residuals"])
    
    # Recompute distribution
    all_scores = [r["output"]["coherence_score"] for r in dataset["data"]["residuals"]]
    hard_gates = sum(1 for s in all_scores if s == 0.0)
    dataset["metadata"]["distribution"] = {
        "avg_coherence": round(sum(all_scores) / max(len(all_scores), 1), 3),
        "hard_gate_pct": round(hard_gates / max(len(all_scores), 1) * 100, 1),
        "real_examples": len(real_data["data"]["residuals"]),
        "synthetic_examples": len(dataset["data"]["residuals"]) - len(real_data["data"]["residuals"]),
    }
    
    # Step 4: Split and save
    print("\n📊 STEP 4/4: Splitting (80/10/10) and saving...")
    splits = split_dataset(dataset)
    save_dataset(splits, dataset["metadata"], OUTPUT_DIR)
    save_real_dataset(real_data, OUTPUT_DIR)
    
    # Validate
    print("\n🔍 Validation...")
    report = validate_dataset(splits)
    print(f"   {report['passed_checks']}/{report['total_checks']} checks passed")
    for check in report["checks"]:
        icon = "✅" if check["passed"] else "❌"
        print(f"   {icon} {check['name']}: {check['detail']}")
    
    # Summary
    _print_summary(dataset)
    return dataset


def run_quick_pipeline(seed: int = 42):
    """Generate 1k synthetic + real for testing."""
    print("🔮 Quick pipeline (1k examples)...")
    
    gen = ResidualDataGenerator(seed=seed)
    dataset = gen.generate_dataset(
        n_residuals=500, n_coherence=200, n_contradictions=100,
        n_sequences=20, n_preferences=100, n_structured=100,
    )
    
    real_data = ingest_all_real_data()
    dataset["data"]["residuals"].extend(real_data["data"]["residuals"])
    
    splits = split_dataset(dataset)
    save_dataset(splits, dataset["metadata"], OUTPUT_DIR)
    
    print("\n✅ Quick pipeline complete. Ready for model testing.")
    return dataset


def cmd_validate():
    """Validate existing dataset."""
    print("🔍 Validating existing dataset...")
    output_path = Path(OUTPUT_DIR)
    
    files = list(output_path.glob("residuals_*.jsonl"))
    if not files:
        print("❌ No dataset found. Run pipeline first.")
        return
    
    # Load and validate
    all_data = {"residuals": {"train": [], "val": [], "test": []}}
    for f in files:
        split_name = f.stem.split("_")[-1]
        with open(f) as fh:
            for line in fh:
                all_data["residuals"][split_name].append(json.loads(line))
    
    report = validate_dataset(all_data)
    print(f"\n   {report['passed_checks']}/{report['total_checks']} checks passed")
    for check in report["checks"]:
        icon = "✅" if check["passed"] else "❌"
        print(f"   {icon} {check['name']}: {check['detail']}")
    return report


def cmd_stats():
    """Print dataset statistics."""
    output_path = Path(OUTPUT_DIR)
    
    print("\n📊 Sovereign Edge AI — Dataset Statistics")
    print("=" * 60)
    
    total = 0
    for f in sorted(output_path.glob("*.jsonl")):
        count = sum(1 for _ in open(f))
        size_kb = os.path.getsize(f) / 1024
        print(f"   {f.name:45s} {count:>6,} rows  ({size_kb:>6.0f} KB)")
        total += count
    
    # Show metadata if available
    meta_file = output_path / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"\n   Total examples:     {meta.get('total_examples', '?'):,}")
        print(f"   Data types:         {len(meta.get('data_types', {}))}")
        if 'distribution' in meta:
            d = meta['distribution']
            print(f"   Avg coherence:      {d.get('avg_coherence', '?')}")
            print(f"   Hard gate %:        {d.get('hard_gate_pct', '?')}%")
    
    real_file = output_path / "metadata_real.json"
    if real_file.exists():
        with open(real_file) as f:
            real = json.load(f)
        print(f"\n   Real data sources:  {len(real.get('sources', {}))}")
        print(f"   Real examples:      {real.get('total_examples', '?'):,}")
    
    print(f"\n   Total files:        {len(list(output_path.glob('*.jsonl')))}")
    print(f"   Output directory:   {output_path}")


def _print_summary(dataset: dict):
    """Print training-ready summary."""
    meta = dataset["metadata"]
    dist = meta.get("distribution", {})
    
    print("\n" + "=" * 60)
    print("🎯 DATASET READY FOR TRAINING")
    print("=" * 60)
    print(f"   Total examples:     {meta.get('total_examples', 0):,}")
    print(f"   Synthetic:          {dist.get('synthetic_examples', meta.get('total_examples', 0)):,}")
    print(f"   Real:               {dist.get('real_examples', 0):,}")
    print(f"   Data types:         6 (residuals, coherence, contradictions, sequences, preferences, structured)")
    print(f"   Avg coherence:      {dist.get('avg_coherence', '?')}")
    print(f"   Hard gates:         {dist.get('hard_gate_pct', '?')}%")
    print(f"   Splits:             80/10/10 (train/val/test)")
    print(f"   Schema version:     1.0")
    print(f"   License:            CC-BY-4.0 (synthetic) + MIT/CC-BY-4.0 (real)")
    print(f"   Location:           {OUTPUT_DIR}")
    print("\n   Next: Phase 2 — Model selection & fine-tuning")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Sovereign Edge AI — Complete Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--full", action="store_true", help="Generate full 11k dataset")
    parser.add_argument("--quick", action="store_true", help="Generate 1k quick dataset")
    parser.add_argument("--validate", action="store_true", help="Validate existing dataset")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.full:
        run_full_pipeline(seed=args.seed)
    elif args.quick:
        run_quick_pipeline(seed=args.seed)
    elif args.validate:
        cmd_validate()
    elif args.stats:
        cmd_stats()
    else:
        parser.print_help()
        print("\nTip: Start with '--quick' for a 1k test dataset, then '--full' for production.")


if __name__ == "__main__":
    main()