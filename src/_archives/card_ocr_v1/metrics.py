from collections.abc import Sequence

import numpy as np


def _levenshtein_distance(source: Sequence, target: Sequence) -> int:
    if source == target:
        return 0
    if len(source) == 0:
        return len(target)
    if len(target) == 0:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for i, source_token in enumerate(source, start=1):
        current_row = [i]
        for j, target_token in enumerate(target, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (0 if source_token == target_token else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _normalize_for_match(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)


def build_ocr_metrics(processor):
    pad_token_id = processor.tokenizer.pad_token_id

    def compute_metrics(eval_prediction) -> dict:
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.where(predictions == -100, pad_token_id, predictions)
        labels = np.where(labels == -100, pad_token_id, labels)

        decoded_predictions = processor.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

        exact_matches = []
        normalized_exact_matches = []
        cer_values = []
        wer_values = []

        for predicted_text, target_text in zip(decoded_predictions, decoded_labels, strict=False):
            predicted_text = predicted_text.strip()
            target_text = target_text.strip()

            exact_matches.append(float(predicted_text == target_text))
            normalized_exact_matches.append(
                float(_normalize_for_match(predicted_text) == _normalize_for_match(target_text))
            )

            cer_values.append(
                _safe_divide(
                    _levenshtein_distance(list(predicted_text), list(target_text)),
                    max(1, len(target_text)),
                )
            )

            predicted_words = predicted_text.split()
            target_words = target_text.split()
            wer_values.append(
                _safe_divide(
                    _levenshtein_distance(predicted_words, target_words),
                    max(1, len(target_words)),
                )
            )

        return {
            "exact_match": float(np.mean(exact_matches)),
            "normalized_exact_match": float(np.mean(normalized_exact_matches)),
            "cer": float(np.mean(cer_values)),
            "wer": float(np.mean(wer_values)),
        }

    return compute_metrics
