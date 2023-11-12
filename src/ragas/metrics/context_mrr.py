from __future__ import annotations

import sys
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset
from sentence_transformers import CrossEncoder

from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import CallbackManager

logger = logging.getLogger("Evaluation-Tab")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
logger.addHandler(handler)

@dataclass
class ContextMeanReciprocalRank(MetricWithLLM):
    """
    Uses a cross encoder to score semantic similarity between the 
    ground truth context and each retrieved document, then calculates
    the Mean Reciprocal Rank of the retrieved documents.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    embeddings:
        The cross-encoder model to be used.
        Defaults cross-encoder/stsb-roberta-large
        Other good options https://huggingface.co/spaces/mteb/leaderboard
    threshold:
        The threshold used to map output to binary
        Default 0.6
    """

    name: str = "context_mrr"
    evaluation_mode: EvaluationMode = EvaluationMode.ga
    batch_size: int = 16
    embeddings: str | None = None
    threshold: float | None = 0.6

    latest_logs: dict = field(default_factory=dict)

    def __post_init__(self: t.Self):
        if self.embeddings is None:
            self.cross_encoder = CrossEncoder("cross-encoder/stsb-roberta-large")
        else:
            self.cross_encoder = CrossEncoder(self.embeddings)

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        questions = dataset["question"]
        ground_truths = dataset["ground_truths"]
        contexts = dataset["contexts"]
        answers = dataset["answer"]

        if "ground_truth_context" not in dataset:
            raise ValueError(("ContextMeanReciprocalRank error: "
                              "source_text not found in dataset")
            )
        else:
            ground_truth_context = dataset["ground_truth_context"]

        # Log each item added to latest_logs
        self._log_and_update('question', questions[0])
        self._log_and_update('ground_truth_answer', ground_truths[0])
        self._log_and_update('retrieved_documents', contexts[0])
        self._log_and_update('ground_truth_context', ground_truth_context)
        self._log_and_update('threshold', self.threshold)
        self._log_and_update('generated_answer', answers[0])
        self._log_and_update('model_kwargs', self.llm.llm.model_kwargs)

        retrieved_contexts = contexts[0]

        inputs = [list(item) for item in list(zip(ground_truth_context * \
            len(retrieved_contexts), retrieved_contexts))]

        # Log threshold
        scores = self.cross_encoder.predict(
            inputs, batch_size=self.batch_size, convert_to_numpy=True
        )
        # Log scores
        self._log_and_update('scores', scores.tolist())
        assert isinstance(scores, np.ndarray), "Expects ndarray"
        scores = scores >= self.threshold  # type: ignore
        self.log_and_update('thresholded_scores', scores.tolist())

        # Calculate the mean reciprocal rank
        mean_reciprocal_rank = 0
        for rank, score in enumerate(scores):
            if score:
                mean_reciprocal_rank = 1/(rank+1)

        return [mean_reciprocal_rank]

    def _log_and_update(self, key, value):
        """
        Helper method to log the addition of a new item to latest_logs.
        """
        self.latest_logs[key] = value
        logger.debug(f"ContextMeanReciprocalRank - {key}: {value}")
