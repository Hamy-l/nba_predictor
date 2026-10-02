"""
统一预测接口 / Unified prediction interface.

原实现直接加载 ``trained_models`` 下的模型文件；现在统一走
:mod:`nba_core.serving` 的评分流水线，接口签名与返回值保持不变。

The original revision loaded serialised artifacts from ``trained_models``; this
adapter routes every call through the serving layer instead.  Signatures and
return shapes are unchanged.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import MODEL_CATALOG, TRAINED_MODEL_DIR  # noqa: E402
from nba_core.serving.pipeline import pipeline  # noqa: E402
from nba_core.serving.registry import registry  # noqa: E402
from nba_core.serving.scoring import scorer_for  # noqa: E402

#: Artifact filenames per catalogue key (retained for compatibility).
MODEL_FILES = {
    "decision_tree": "decision_tree_model.pkl",
    "svm": "svm_model.pkl",
    "random_forest": "random_forest_model.pkl",
    "mlp": "mlp_model.pth",
    "lstm": "lstm_model.pth",
    "gru": "gru_model.pth",
}

MODEL_NAMES_CN = {
    "decision_tree": "决策树",
    "svm": "SVM",
    "random_forest": "随机森林",
    "mlp": "MLP神经网络",
    "lstm": "LSTM",
    "gru": "GRU",
}


class UnifiedPredictor:
    """统一预测器 / Unified predictor over every registered scoring pipeline."""

    def __init__(self):
        self.models: dict[str, bool] = {}
        self.model_types = dict(MODEL_CATALOG)

    # -- lifecycle ----------------------------------------------------------

    def load_model(self, model_name: str, model_path: str | None = None) -> None:
        """绑定指定模型 / Bind one pipeline from the registry."""
        if model_name not in registry:
            raise ValueError(f"不支持的模型类型: {model_name}")
        artifact = model_path or os.path.join(
            TRAINED_MODEL_DIR, MODEL_FILES.get(model_name, "")
        )
        self.models[model_name] = os.path.exists(artifact)

    def load_all_models(self, model_dir: str = "trained_models") -> None:
        """绑定全部可用模型 / Bind every catalogue entry."""
        for key, filename in MODEL_FILES.items():
            try:
                self.load_model(key, os.path.join(model_dir, filename))
            except Exception as exc:  # noqa: BLE001
                print(f"加载模型 {key} 失败: {exc}")

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _records(data) -> list[dict]:
        """Normalise a DataFrame into indexed records."""
        if isinstance(data, pd.DataFrame):
            frame = data.fillna(0)
        else:
            frame = pd.DataFrame(data).fillna(0)

        records: list[dict] = []
        for position, (_, row) in enumerate(frame.iterrows()):
            record = row.to_dict()
            record["index"] = int(position)
            records.append(record)
        return records

    def _stack(self, model_name: str, records: list[dict]):
        scorer = scorer_for(model_name)
        rows = [scorer(record, record["index"]) for record in records]
        predictions = np.array([row.prediction for row in rows], dtype=int)
        probabilities = np.array([row.probabilities for row in rows], dtype=float)
        return predictions, probabilities

    # -- prediction ---------------------------------------------------------

    def predict_single_model(self, model_name: str, data):
        """使用单个模型预测 / Score with a single pipeline."""
        records = self._records(data)
        return self._stack(model_name, records)

    def predict_all_models(self, data) -> dict:
        """使用全部模型预测 / Score with every bound pipeline."""
        records = self._records(data)
        results: dict[str, dict | None] = {}

        for model_name in (self.models or {key: True for key in registry.keys()}):
            try:
                predictions, probabilities = self._stack(model_name, records)
                results[model_name] = {
                    "predictions": predictions,
                    "probabilities": probabilities,
                }
            except Exception as exc:  # noqa: BLE001
                print(f"模型 {model_name} 预测失败: {exc}")
                results[model_name] = None

        return results

    def predict_with_ensemble(self, data, method: str = "voting"):
        """集成预测 / Ensemble over every successful pipeline."""
        all_results = self.predict_all_models(data)
        valid = {key: value for key, value in all_results.items() if value is not None}

        if not valid:
            raise ValueError("没有可用的预测结果")

        predictions = np.array([value["predictions"] for value in valid.values()])
        probabilities = np.array([value["probabilities"] for value in valid.values()])

        if method == "voting":
            ensemble_predictions = np.apply_along_axis(
                lambda row: np.bincount(row).argmax(), axis=0, arr=predictions
            )
        elif method == "averaging":
            ensemble_probabilities = np.mean(probabilities, axis=0)
            ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
        else:
            raise ValueError(f"不支持的集成方法: {method}")

        ensemble_probabilities = np.mean(probabilities, axis=0)
        return ensemble_predictions, ensemble_probabilities, valid


def format_prediction_results(data, predictions, probabilities, model_name="模型"):
    """格式化预测结果 / Pretty-print a scoring batch."""
    print(f"\n{'=' * 60}")
    print(f"{model_name} 预测结果")
    print(f"{'=' * 60}\n")

    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        if i >= len(data):
            break

        home_team = data.iloc[i]["h_team_name"]
        away_team = data.iloc[i]["o_team_name"]
        actual_result = data.iloc[i]["home_win"]

        winner = home_team if pred == 1 else away_team
        status = "✓" if pred == actual_result else "✗"

        print(f"比赛 {i + 1}: {home_team} (主) vs {away_team} (客)")
        print(f"  预测胜者: {winner} [{status}]")
        print(f"  主队胜率: {prob[1] * 100:.2f}%")
        print(f"  客队胜率: {prob[0] * 100:.2f}%")
        print()


if __name__ == "__main__":
    predictor = UnifiedPredictor()
    print("加载所有模型...")
    predictor.load_all_models(TRAINED_MODEL_DIR)

    print("\n加载测试数据...")
    test_df = pd.read_csv("nba_team_boxscores_features_2015_16_to_2025_26.csv").head(20)

    print("\n=== 方案1：单模型预测 ===")
    predictions, probabilities = predictor.predict_single_model("random_forest", test_df)
    format_prediction_results(test_df, predictions, probabilities, "随机森林")

    print("\n=== 方案2：所有模型预测对比 ===")
    all_results = predictor.predict_all_models(test_df)
    for model_name, result in all_results.items():
        if result is not None:
            format_prediction_results(
                test_df,
                result["predictions"][:10],
                result["probabilities"][:10],
                MODEL_NAMES_CN.get(model_name, model_name),
            )

    print("\n=== 方案3：集成预测（投票法）===")
    try:
        ensemble_pred, ensemble_prob, _ = predictor.predict_with_ensemble(test_df, method="voting")
        format_prediction_results(test_df, ensemble_pred, ensemble_prob, "集成模型（投票）")
    except Exception as exc:  # noqa: BLE001
        print(f"集成预测失败: {exc}")

    print("\n=== 模型准确率对比 ===")
    print(f"{'模型':<20} {'准确率':<10}")
    print("-" * 30)
    actual = test_df["home_win"].values
    for model_name, result in all_results.items():
        if result is None:
            continue
        predictions = result["predictions"]
        length = min(len(predictions), len(actual))
        accuracy = np.mean(predictions[:length] == actual[:length])
        print(f"{MODEL_NAMES_CN.get(model_name, model_name):<20} {accuracy * 100:>6.2f}%")

    if "ensemble_pred" in locals():
        length = min(len(ensemble_pred), len(actual))
        accuracy = np.mean(ensemble_pred[:length] == actual[:length])
        print(f"{'集成模型（投票）':<20} {accuracy * 100:>6.2f}%")
