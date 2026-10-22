"""Train, compare, tune and save the churn model.

Usage:
    python -m churnguard.train                     # full run with tuning and MLflow tracking
    python -m churnguard.train --trials 10         # quicker tuning
    python -m churnguard.train --no-tracking       # skip MLflow

Workflow
1. Clean and validate the data, then hold out 20% as a test set that is touched only once.
2. Compare candidate models with stratified 5-fold cross-validation on the training set.
3. Tune XGBoost with Optuna, again scored by cross-validation.
4. Pick the best model by cross-validated ROC-AUC and choose its decision threshold
   from out-of-fold predictions.
5. Fit on the full training set, evaluate once on the test set, explain with SHAP,
   and save the model artifact plus reports.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from churnguard import __version__
from churnguard.config import load_config, resolve
from churnguard.data import FEATURE_COLUMNS, ID_COLUMN, clean, load_raw, split_xy, validate
from churnguard.evaluate import (
    choose_threshold,
    classification_metrics,
    gain_at,
    plot_confusion,
    plot_feature_importance,
    plot_gain,
    plot_roc_pr,
)
from churnguard.explain import ChurnExplainer
from churnguard.features import build_pipeline
from churnguard.models import CANDIDATE_MODELS, make_model, xgboost_search_space
from churnguard.predict import save_artifact


class Tracker:
    """Thin wrapper so MLflow is optional (tests and quick runs can switch it off)."""

    def __init__(self, enabled: bool, uri: str, experiment: str, root: Path):
        self.enabled = enabled
        if enabled:
            import mlflow

            if uri.startswith("sqlite:///") and not Path(uri[10:]).is_absolute():
                uri = "sqlite:///" + (root / uri[10:]).as_posix()
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            self.mlflow = mlflow

    def run(self, name: str):
        return self.mlflow.start_run(run_name=name) if self.enabled else nullcontext()

    def log(self, params: dict | None = None, metrics: dict | None = None,
            artifacts: list[Path] | None = None) -> None:
        if not self.enabled:
            return
        if params:
            self.mlflow.log_params({k: v for k, v in params.items() if v is not None})
        if metrics:
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        for path in artifacts or []:
            self.mlflow.log_artifact(str(path))


def cross_validated_proba(name: str, params: dict, X, y, cv, spw: float) -> np.ndarray:
    """Out-of-fold churn probabilities: every row is predicted by a model that never saw it."""
    pipeline = build_pipeline(make_model(name, params, scale_pos_weight=spw))
    return cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]


def tune_xgboost(X, y, cv, spw: float, n_trials: int, timeout: int, seed: int) -> dict:
    """Search XGBoost hyperparameters with Optuna (TPE sampler), maximising CV ROC-AUC."""
    import optuna
    from sklearn.model_selection import cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = xgboost_search_space(trial)
        pipeline = build_pipeline(make_model("xgboost", params, scale_pos_weight=spw))
        return cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    print(f"  Optuna best CV ROC-AUC {study.best_value:.4f} after {len(study.trials)} trials")
    return study.best_params


def run_training(config: dict[str, Any], root: Path, track: bool = True,
                 n_trials: int | None = None) -> dict[str, Any]:
    started = time.time()
    seed = config["split"]["random_state"]
    reports_dir = resolve(config["output"]["reports_dir"], root)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data -------------------------------------------------------------------------
    df = clean(load_raw(resolve(config["data"]["raw_path"], root)))
    validate(df, require_target=True)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["split"]["test_size"], stratify=y, random_state=seed
    )
    print(f"Data: {len(df):,} customers, churn rate {y.mean():.1%}. "
          f"Train {len(X_train):,} / test {len(X_test):,}")

    cv = StratifiedKFold(n_splits=config["cv"]["n_splits"], shuffle=True, random_state=seed)
    spw = float((y_train == 0).sum() / (y_train == 1).sum())  # imbalance weight for XGBoost
    tracker = Tracker(track and config["tracking"]["enabled"], config["tracking"]["uri"],
                      config["tracking"]["experiment"], root)

    # 2. Compare candidate models with cross-validation ------------------------------
    candidates: dict[str, dict[str, Any]] = {}
    for name in CANDIDATE_MODELS:
        print(f"Cross-validating {name} ...")
        with tracker.run(name):
            oof = cross_validated_proba(name, {}, X_train, y_train, cv, spw)
            cv_metrics = classification_metrics(y_train, oof)
            candidates[name] = {"params": {}, "oof": oof, "cv": cv_metrics}
            tracker.log(params={"model": name},
                        metrics={"cv_roc_auc": cv_metrics["roc_auc"],
                                 "cv_pr_auc": cv_metrics["pr_auc"]})

    # 3. Tune XGBoost ------------------------------------------------------------------
    tuning = config["tuning"]
    if tuning["enabled"]:
        trials = n_trials or tuning["n_trials"]
        print(f"Tuning xgboost with Optuna ({trials} trials) ...")
        with tracker.run("xgboost_tuned"):
            best_params = tune_xgboost(X_train, y_train, cv, spw, trials,
                                       tuning["timeout_seconds"], seed)
            oof = cross_validated_proba("xgboost", best_params, X_train, y_train, cv, spw)
            cv_metrics = classification_metrics(y_train, oof)
            candidates["xgboost_tuned"] = {"params": best_params, "oof": oof, "cv": cv_metrics}
            tracker.log(params={"model": "xgboost_tuned", **best_params},
                        metrics={"cv_roc_auc": cv_metrics["roc_auc"],
                                 "cv_pr_auc": cv_metrics["pr_auc"]})

    # 4. Select the best model and its threshold (training data only) ---------------
    best_name = max(candidates, key=lambda n: candidates[n]["cv"]["roc_auc"])
    best = candidates[best_name]
    threshold = choose_threshold(y_train, best["oof"], config["threshold"]["strategy"],
                                 config["threshold"]["min_recall"])
    print(f"Best model by CV ROC-AUC: {best_name} (threshold {threshold:.3f})")

    # 5. Fit every candidate on the full training set and score the test set once -----
    fitted, test_proba = {}, {}
    top_fraction = config["business"]["top_fraction"]
    comparison_rows = []
    for name, cand in candidates.items():
        pipeline = build_pipeline(make_model(name, cand["params"], scale_pos_weight=spw))
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        fitted[name], test_proba[name] = pipeline, proba
        comparison_rows.append({
            "model": name,
            "cv_roc_auc": cand["cv"]["roc_auc"],
            "cv_pr_auc": cand["cv"]["pr_auc"],
            "test_roc_auc": classification_metrics(y_test, proba)["roc_auc"],
            "test_pr_auc": classification_metrics(y_test, proba)["pr_auc"],
            f"test_gain_top_{int(top_fraction * 100)}pct": gain_at(y_test, proba, top_fraction),
        })

    final = fitted[best_name]
    test_metrics = classification_metrics(y_test, test_proba[best_name], threshold)
    test_metrics["gain_top_fraction"] = gain_at(y_test, test_proba[best_name], top_fraction)
    test_metrics["top_fraction"] = top_fraction

    baseline = classification_metrics(y_test, test_proba["logistic_regression"])
    print(f"Test ROC-AUC {test_metrics['roc_auc']:.4f} (baseline LR {baseline['roc_auc']:.4f}), "
          f"recall {test_metrics['recall']:.1%}, precision {test_metrics['precision']:.1%}, "
          f"top {top_fraction:.0%} reaches {test_metrics['gain_top_fraction']:.1%} of churners")

    # 6. Explain ------------------------------------------------------------------------
    background = X_train.sample(n=min(200, len(X_train)), random_state=seed)
    explainer = ChurnExplainer(final, background)
    importance = explainer.global_importance(X_test)

    # 7. Reports and figures --------------------------------------------------------------
    figures = {
        "roc_pr": figures_dir / "roc_pr_curves.png",
        "confusion": figures_dir / "confusion_matrix.png",
        "gain": figures_dir / "cumulative_gain.png",
        "importance": figures_dir / "shap_importance.png",
    }
    plot_roc_pr(y_test, test_proba, figures["roc_pr"])
    plot_confusion(test_metrics, figures["confusion"],
                   f"{best_name} @ threshold {threshold:.2f}")
    plot_gain(y_test, test_proba[best_name], figures["gain"], top_fraction)
    plot_feature_importance(importance, figures["importance"])

    comparison = pd.DataFrame(comparison_rows).sort_values("cv_roc_auc", ascending=False)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    (reports_dir / "model_comparison.md").write_text(
        comparison.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8"
    )

    summary = {
        "version": __version__,
        "best_model": best_name,
        "best_params": best["params"],
        "threshold": threshold,
        "threshold_strategy": config["threshold"],
        "cv_metrics": best["cv"],
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline,
        "feature_importance": importance.round(4).to_dict(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": float(y.mean()),
        "training_seconds": round(time.time() - started, 1),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 8. Save the model artifact ----------------------------------------------------------
    model_path = save_artifact(
        resolve(config["output"]["model_path"], root),
        final,
        threshold,
        background,
        metadata={k: summary[k] for k in
                  ["version", "best_model", "best_params", "threshold", "test_metrics",
                   "cv_metrics", "n_train", "n_test", "churn_rate"]},
    )

    with tracker.run(f"final_{best_name}"):
        tracker.log(
            params={"model": best_name, "threshold": round(threshold, 4), **best["params"]},
            metrics={f"test_{k}": v for k, v in test_metrics.items()
                     if isinstance(v, float)},
            artifacts=list(figures.values()) + [reports_dir / "metrics.json", model_path],
        )

    # Sample customers (without labels) for the dashboard demo.
    sample = df.loc[X_test.index[:300], [ID_COLUMN, *FEATURE_COLUMNS]]
    sample_path = root / "data" / "sample" / "sample_customers.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False)

    print(f"Saved model to {model_path} in {summary['training_seconds']}s")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ChurnGuard model.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--no-tracking", action="store_true", help="Disable MLflow tracking")
    parser.add_argument("--trials", type=int, default=None, help="Override Optuna trial count")
    args = parser.parse_args()

    from churnguard.config import PROJECT_ROOT

    config = load_config(args.config)
    run_training(config, PROJECT_ROOT, track=not args.no_tracking, n_trials=args.trials)


if __name__ == "__main__":
    main()
