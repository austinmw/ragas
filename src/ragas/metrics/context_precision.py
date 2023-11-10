from __future__ import annotations

import sys
import logging
import typing as t
from dataclasses import dataclass, field
from typing import List

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM

logger = logging.getLogger("Evaluation-Tab")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
logger.addHandler(handler)


CONTEXT_PRECISION = HumanMessagePromptTemplate.from_template(
    """<instructions>
Given a question and a context, verify if the information in the given context is useful in answering the question. 
Answer only with a single word of either "Yes" or "No" and nothing else.
</instructions>

<example_input>
<question>
What is the significance of the Statue of Liberty in New York City?
</question>

<context>
The Statue of Liberty National Monument and Ellis Island Immigration Museum are managed by the National Park Service and are in both New York and New Jersey. They are joined in the harbor by Governors Island National Monument. Historic sites under federal management on Manhattan Island include Stonewall National Monument; Castle Clinton National Monument; Federal Hall National Memorial; Theodore Roosevelt Birthplace National Historic Site; General Grant National Memorial (Grant's Tomb); African Burial Ground National Monument; and Hamilton Grange National Memorial. Hundreds of properties are listed on the National Register of Historic Places or as a National Historic Landmark.
</context>
</example_input>

<example_response>Yes</example_response>

Here is the question and context for you to analyze:
<question>
{question}
</question>

<context>
{context}
</context>

Assistant: The single word answer is: """  # noqa: E501
)

@dataclass
class ContextPrecision(MetricWithLLM):
    """
    Average Precision is a metric that evaluates whether all of the
    relevant items selected by the model are ranked higher or not.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_precision"
    evaluation_mode: EvaluationMode = EvaluationMode.qc
    batch_size: int = 15

    latest_logs: dict = field(default_factory=dict)

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list:
        prompts = []

        questions = dataset["question"]
        ground_truths = dataset["ground_truths"]
        contexts = dataset["contexts"]
        answer = dataset["answer"]

        contexts = contexts[0]
        questions *= len(contexts)

        # Log each item added to latest_logs
        self._log_and_update('question', questions[0])
        self._log_and_update('ground_truth_answer', ground_truths[0])
        self._log_and_update('retrieved_documents', contexts)
        self._log_and_update('generated_answer', answer[0])
        self._log_and_update("num_contexts", len(contexts))

        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:

            prompt_contexts = []
            for qstn, ctx in zip(questions, contexts):
                human_prompt = CONTEXT_PRECISION.format(question=qstn, context=ctx)
                prompt_contexts.append(human_prompt.content)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))
            self._log_and_update('prompts', prompt_contexts)

            results = self.llm.generate(
                prompts,
                n=1,
                callbacks=batch_group,
            )

            responses = [[i.text.strip() for i in r][0] for r in results.generations]
            self._log_and_update('responses', responses)

            binary_responses = [1 if 'yes' in r.lower() else 0 for r in responses]
            self._log_and_update('binary_responses', binary_responses)

            document_lengths = [len(ctx) for ctx in contexts]
            self._log_and_update('document_lengths', document_lengths)

            weighted_precisions = []
            num_true_positives = 0
            total_length_positive = sum([length for r, length in zip(binary_responses, document_lengths) if r == 1])
            self._log_and_update('total_length_positive', total_length_positive)

            for i, (response, length) in enumerate(zip(binary_responses, document_lengths)):
                if response == 1:
                    num_true_positives += 1
                    precision_at_i = num_true_positives / (i + 1)
                    weighted_precisions.append(precision_at_i * length)
                else:
                    weighted_precisions.append(0)
            self._log_and_update('weighted_precisions', weighted_precisions)

            weighted_sum_precision = sum(weighted_precisions)
            self._log_and_update('weighted_sum_precision', weighted_sum_precision)
            average_precision = weighted_sum_precision / total_length_positive if total_length_positive else 1e-6
            self._log_and_update('context_average_precision', average_precision)

            scores = [average_precision]

        return scores

    def _log_and_update(self, key, value):
        """
        Helper method to log the addition of a new item to latest_logs.
        """
        self.latest_logs[key] = value
        #logger.debug(f"ContextPrecision - {key}: {value}")


@dataclass
class ContextOutsideInPrecision(MetricWithLLM):
    """
    ContextOutsideInPrecision is a metric that evaluates the relevance of items
    by ranking them in an alternating outside-in manner starting from the last: 
    N, 1, N-1, 2, N-2, 3, etc., where N is the total number of items.

    Relevant for the following:
    https://python.langchain.com/docs/modules/data_connection/document_transformers/post_retrieval/long_context_reorder

    Attributes
    ----------
    name : str
        Name of the metric.
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_outsidein_precision"
    evaluation_mode: EvaluationMode = EvaluationMode.qc
    batch_size: int = 15

    latest_logs: dict = field(default_factory=dict)

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list:
        prompts = []

        questions = dataset["question"]
        ground_truths = dataset["ground_truths"]
        contexts = dataset["contexts"]
        answer = dataset["answer"]

        contexts = self._reorder_contexts_outside_in(contexts[0])
        questions *= len(contexts)

        # Log each item added to latest_logs
        self._log_and_update('question', questions[0])
        self._log_and_update('ground_truth_answer', ground_truths[0])
        self._log_and_update('retrieved_documents', contexts)
        self._log_and_update('generated_answer', answer[0])
        self._log_and_update("num_contexts", len(contexts))
        self._log_and_update('model_kwargs', self.llm.llm.model_kwargs)

        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:

            prompt_contexts = []
            for qstn, ctx in zip(questions, contexts):
                human_prompt = CONTEXT_PRECISION.format(question=qstn, context=ctx)
                prompt_contexts.append(human_prompt.content)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))
            self._log_and_update('prompts', prompt_contexts)

            results = self.llm.generate(
                prompts,
                n=1,
                callbacks=batch_group,
            )

            responses = [[i.text.strip() for i in r][0] for r in results.generations]
            self._log_and_update('responses', responses)

            binary_responses = [1 if 'yes' in r.lower() else 0 for r in responses]
            self._log_and_update('binary_responses', binary_responses)

            document_lengths = [len(ctx) for ctx in contexts]
            self._log_and_update('document_lengths', document_lengths)

            weighted_precisions = []
            num_true_positives = 0
            total_length_positive = sum([length for r, length in zip(binary_responses, document_lengths) if r == 1])
            self._log_and_update('total_length_positive', total_length_positive)

            for i, (response, length) in enumerate(zip(binary_responses, document_lengths)):
                if response == 1:
                    num_true_positives += 1
                    precision_at_i = num_true_positives / (i + 1)
                    weighted_precisions.append(precision_at_i * length)
                else:
                    weighted_precisions.append(0)
            self._log_and_update('weighted_precisions', weighted_precisions)

            weighted_sum_precision = sum(weighted_precisions)
            self._log_and_update('weighted_sum_precision', weighted_sum_precision)
            average_precision = weighted_sum_precision / total_length_positive if total_length_positive else 1e-6
            self._log_and_update('context_average_precision', average_precision)

            scores = [average_precision]

        return scores

    def _log_and_update(self, key, value):
        """
        Helper method to log the addition of a new item to latest_logs.
        """
        self.latest_logs[key] = value
        #logger.debug(f"ContextOutsideInPrecision - {key}: {value}")

    @staticmethod
    def _reorder_contexts_outside_in(contexts):
        """
        Sorts a list based on the pattern seen in the example: [2, 4, 5, 3, 1].
        The pattern is even indices first in ascending order followed by odd indices in descending order.
        Another way to see it is moving outside-in, starting from the end.
        """
        # Separate the list into even and odd indexed elements
        even_indices = contexts[::2]
        odd_indices = contexts[1::2]
        # Reverse the list with odd indices
        odd_indices.reverse()
        # Combine the lists
        reordered = even_indices + odd_indices
        reordered.reverse()
        return reordered
