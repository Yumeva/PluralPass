from __future__ import annotations

import argparse
import json

from pluralpass.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="pluralpass")
    parser.add_argument(
        "command",
        choices=[
            "download",
            "preprocess",
            "audit",
            "make-splits",
            "make-grouped-kfold-splits",
            "sensitivity-cohorts",
            "baselines",
            "train",
            "evaluate",
            "report-formal-results",
            "set-efficiency",
            "split-bias",
            "neural-split-bias",
            "observation-stress",
            "counterfactual-stress",
            "calibration-diagnostics",
            "baseline-matrix",
            "candidate-mlp-comparison",
            "grouped10-stability",
            "display-tables",
            "ablation-plan",
            "ablation-status",
            "recommendation-ablations",
            "trainable-ablation-summary",
            "coach-stimuli",
            "coach-analysis",
        ],
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fold")
    parser.add_argument("--predictions")
    parser.add_argument("--responses")
    parser.add_argument("--scenarios")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "download":
        from pluralpass.data.download import download_open_360

        result = download_open_360(config)
    elif args.command == "preprocess":
        from pluralpass.data.preprocess import preprocess

        result = preprocess(config)
    elif args.command == "audit":
        from pluralpass.data.audit import audit_dataset

        result = audit_dataset(config)
    elif args.command == "make-splits":
        from pluralpass.data.splits import make_splits

        result = make_splits(config)
    elif args.command == "make-grouped-kfold-splits":
        from pluralpass.data.splits import make_grouped_kfold_splits

        result = make_grouped_kfold_splits(config)
    elif args.command == "sensitivity-cohorts":
        from pluralpass.data.sensitivity import build_sensitivity_cohorts

        result = build_sensitivity_cohorts(config)
    elif args.command == "train":
        if not args.fold:
            parser.error("train requires --fold")
        from pluralpass.training import train_ensemble

        result = {"checkpoints": [str(x) for x in train_ensemble(config, args.fold)]}
    elif args.command == "baselines":
        if not args.fold:
            parser.error("baselines requires --fold")
        from pluralpass.evaluation.baselines import evaluate_baselines

        result = evaluate_baselines(config, args.fold)
    elif args.command == "evaluate":
        if not args.fold:
            parser.error("evaluate requires --fold")
        from pluralpass.training import evaluate_fold

        result = evaluate_fold(config, args.fold)
    elif args.command == "report-formal-results":
        from pluralpass.evaluation.reporting import collect_formal_results

        result = collect_formal_results(config)
    elif args.command == "set-efficiency":
        from pluralpass.evaluation.set_efficiency import evaluate_set_efficiency

        result = evaluate_set_efficiency(config)
    elif args.command == "split-bias":
        from pluralpass.evaluation.split_bias import evaluate_event_random_bias

        result = evaluate_event_random_bias(config)
    elif args.command == "neural-split-bias":
        from pluralpass.evaluation.neural_split_bias import evaluate_neural_split_bias

        result = evaluate_neural_split_bias(config)
    elif args.command == "observation-stress":
        from pluralpass.evaluation.stress import evaluate_observation_stress

        result = evaluate_observation_stress(config)
    elif args.command == "counterfactual-stress":
        from pluralpass.evaluation.counterfactual_stress import evaluate_counterfactual_stress

        result = evaluate_counterfactual_stress(config)
    elif args.command == "calibration-diagnostics":
        from pluralpass.evaluation.calibration import evaluate_calibration_diagnostics

        result = evaluate_calibration_diagnostics(config)
    elif args.command == "baseline-matrix":
        from pluralpass.evaluation.baseline_matrix import collect_baseline_matrix

        result = collect_baseline_matrix(config)
    elif args.command == "candidate-mlp-comparison":
        from pluralpass.evaluation.neural_baseline import compare_candidate_mlp_baseline

        result = compare_candidate_mlp_baseline(config)
    elif args.command == "grouped10-stability":
        from pluralpass.evaluation.neural_baseline import compare_grouped10_stability

        result = compare_grouped10_stability(config)
    elif args.command == "display-tables":
        from pluralpass.evaluation.display_tables import build_display_tables

        result = build_display_tables(config)
    elif args.command == "ablation-plan":
        from pluralpass.evaluation.ablations import build_ablation_plan

        result = build_ablation_plan(config)
    elif args.command == "ablation-status":
        from pluralpass.evaluation.ablations import collect_ablation_status

        result = collect_ablation_status(config)
    elif args.command == "recommendation-ablations":
        from pluralpass.evaluation.ablations import evaluate_recommendation_ablations

        result = evaluate_recommendation_ablations(config)
    elif args.command == "trainable-ablation-summary":
        from pluralpass.evaluation.ablations import summarize_trainable_ablations

        result = summarize_trainable_ablations(config)
    elif args.command == "coach-stimuli":
        from pluralpass.coach.stimuli import generate_stimuli

        result = generate_stimuli(config, args.predictions)
    elif args.command == "coach-analysis":
        if not args.responses or not args.scenarios:
            parser.error("coach-analysis requires --responses and --scenarios")
        from pluralpass.coach.analysis import analyse_coach_responses

        result = analyse_coach_responses(
            args.responses,
            args.scenarios,
            args.output or "outputs/PluralPass_coach_analysis.json",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
