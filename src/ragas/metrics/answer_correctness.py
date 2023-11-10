from __future__ import annotations

import sys
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset

from ragas.metrics.answer_similarity import AnswerSimilarity
from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.metrics.faithfulness import Faithfulness

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import CallbackManager

logger = logging.getLogger("Evaluation-Tab")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
logger.addHandler(handler)

@dataclass
class AnswerCorrectness(MetricWithLLM):

    """
    Measures answer correctness compared to ground truth as a combination of
    semantic similarity and factuality

    Attributes
    ----------
    name: string
        The name of the metrics
    batch_size: int
        batch size for evaluation
    weights:
        a list of two weights corresponding to semantic similarity and factuality
        Defaults [0.5, 0.5]
    answer_similarity:
        The AnswerSimilarity object
    faithfulness
        The faithfulness object
    """

    name: str = "answer_correctness"
    evaluation_mode: EvaluationMode = EvaluationMode.qga
    batch_size: int = 15
    weights: list[float] = field(default_factory=lambda: [0.5, 0.5])
    answer_similarity: AnswerSimilarity | None = None
    faithfulness: Faithfulness | None = None

    latest_logs: dict = field(default_factory=dict)

    def __post_init__(self: t.Self):
        if self.answer_similarity is None:
            self.answer_similarity = AnswerSimilarity(
                llm=self.llm, batch_size=self.batch_size
            )
        if self.faithfulness is None:
            self.faithfulness = Faithfulness(llm=self.llm, batch_size=self.batch_size)

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:

        questions = dataset["question"]
        ground_truths = dataset["ground_truths"]
        contexts = dataset["contexts"]
        answer = dataset["answer"]

        # Log each item added to latest_logs
        self._log_and_update('question', questions[0])
        self._log_and_update('ground_truth_answer', ground_truths[0])
        self._log_and_update('retrieved_documents', contexts[0])
        self._log_and_update('generated_answer', answer[0])
        self._log_and_update('faithfulness model_kwargs', self.faithfulness.llm.llm.model_kwargs)
        self._log_and_update('similarity model_kwargs', self.answer_similarity.llm.llm.model_kwargs)

        faith_scores = self.faithfulness._score_batch(dataset)  # type: ignore
        similarity_scores = self.answer_similarity._score_batch(dataset)  # type: ignore

        self._log_and_update('faithfulness_scores', faith_scores)
        self._log_and_update('similarity_scores', similarity_scores)

        scores_stacked = np.vstack([faith_scores, similarity_scores])
        scores = np.average(
            scores_stacked,
            axis=0,
            weights=self.weights,
        )
        self._log_and_update('weighted_scores', scores.tolist())

        return scores.tolist()


    def _log_and_update(self, key, value):
        """
        Helper method to log the addition of a new item to latest_logs.
        """
        self.latest_logs[key] = value
        #logger.debug(f"AnswerCorrectness - {key}: {value}")
