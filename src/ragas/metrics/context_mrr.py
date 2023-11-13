from __future__ import annotations

import sys
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset
from sentence_transformers import CrossEncoder
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity

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
    Uses a the max of a thresholded semantic similarity score between the
    ground truth context and each retrieved document or ground truth answer,
    then calculates the Mean Reciprocal Rank of the max score above a threshold.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    embeddings:
        The embedding model to be used.
        Defaults hkunlp/instructor-xl

    threshold:
        The threshold used to map output to binary
        Default 0.9
    """

    name: str = "context_mean_reciprocal_rank"
    evaluation_mode: EvaluationMode = EvaluationMode.qgacs
    batch_size: int = 16
    threshold: float | None = 0.9
    instruction: str = "Represent the document for retrieval:"
    compare: str = "answer"

    latest_logs: dict = field(default_factory=dict)

    def __post_init__(self: t.Self):

        self.model = INSTRUCTOR('hkunlp/instructor-xl')

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
        ground_truth_context = dataset["ground_truth_context"]


        # Log each item added to latest_logs
        self._log_and_update('question', questions[0])
        self._log_and_update('ground_truth_answer', ground_truths[0])
        self._log_and_update('retrieved_documents', contexts[0])
        self._log_and_update('ground_truth_context', ground_truth_context)
        self._log_and_update('threshold', self.threshold)
        self._log_and_update('generated_answer', answers[0])
        self._log_and_update('model_kwargs', self.llm.llm.model_kwargs)

        if self.compare == "ground_truth_context":

            if ground_truth_context is None:
                raise ValueError(("ContextMeanReciprocalRank error: "
                                "ground_truth_context is None")
                )

        retrieved_contexts = contexts[0]

        retrieved_contexts_with_instruction = [
            [self.instruction, context] for context in retrieved_contexts
        ]

        compare_with_instruction = [[self.instruction, dataset[self.compare][0]]]

        embeddings_a = self.model.encode(compare_with_instruction)
        embeddings_b = self.model.encode(retrieved_contexts_with_instruction)
        similarity_scores = cosine_similarity(embeddings_a, embeddings_b)[0]

        # Log scores
        self._log_and_update(f'{self.compare}_similarity_scores', similarity_scores)
        thresholded_similarity_scores = similarity_scores >= self.threshold
        self._log_and_update(
            'thresholded_similarity_scores', 
            thresholded_similarity_scores
        )

        # Calculate the mean reciprocal rank
        mean_reciprocal_rank = 0
        max_score_rank = -1
        for rank, score in enumerate(thresholded_similarity_scores):
            if score:
                max_score_rank = rank
                mean_reciprocal_rank = 1/(rank+1)
                break

        if max_score_rank == -1:
            mean_reciprocal_rank = 0.0

        self._log_and_update('mean_reciprocal_rank', mean_reciprocal_rank)
        return [mean_reciprocal_rank]

    def _log_and_update(self, key, value):
        """
        Helper method to log the addition of a new item to latest_logs.
        """
        self.latest_logs[key] = value
        logger.debug(f"ContextMeanReciprocalRank - {key}: {value}")
