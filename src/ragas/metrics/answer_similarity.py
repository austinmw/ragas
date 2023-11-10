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
class AnswerSimilarity(MetricWithLLM):
    """
    Scores the semantic similarity of ground truth with generated answer.
    cross encoder score is used to quantify semantic similarity.
    SAS paper: https://arxiv.org/pdf/2108.06130.pdf

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    embeddings:
        The cross-encoder model to be used.
        Defaults cross-encoder/stsb-TinyBERT-L-4
        Other good options https://huggingface.co/spaces/mteb/leaderboard
    threshold:
        The threshold if given used to map output to binary
        Default 0.5
    """

    name: str = "answer_similarity"
    evaluation_mode: EvaluationMode = EvaluationMode.ga
    batch_size: int = 15
    embeddings: str | None = None
    threshold: float | None = 0.5

    latest_logs: dict = field(default_factory=dict)

    def __post_init__(self: t.Self):
        if self.embeddings is None:
            self.cross_encoder = CrossEncoder("cross-encoder/stsb-TinyBERT-L-4")

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

        # Log each item added to latest_logs
        self._log_and_update('question', questions[0])
        self._log_and_update('ground_truth_answer', ground_truths[0])
        self._log_and_update('retrieved_documents', contexts[0])
        self._log_and_update('generated_answer', answers[0])
        self._log_and_update('model_kwargs', self.llm.llm.model_kwargs)

        ground_truths = [item[0] for item in ground_truths]
        inputs = [list(item) for item in list(zip(ground_truths, answers))]
        # Log threshold
        self._log_and_update('threshold', self.threshold)
        scores = self.cross_encoder.predict(
            inputs, batch_size=self.batch_size, convert_to_numpy=True
        )
        # Log scores
        self._log_and_update('scores', scores.tolist())

        assert isinstance(scores, np.ndarray), "Expects ndarray"
        if self.threshold:
            scores = scores >= self.threshold  # type: ignore
        # Log thresholded scores
        self._log_and_update('thresholded_scores', scores.tolist())

        return scores.tolist()

    def _log_and_update(self, key, value):
        """
        Helper method to log the addition of a new item to latest_logs.
        """
        self.latest_logs[key] = value
        #logger.debug(f"AnswerSimilarity - {key}: {value}")
