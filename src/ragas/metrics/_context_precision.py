from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from typing import List

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.utils import load_as_json

CONTEXT_PRECISION = HumanMessagePromptTemplate.from_template(
    """\
Verify if the information in the given context is useful in answering the question.

question: What are the health benefits of green tea?
context:
This article explores the rich history of tea cultivation in China, tracing its roots back to the ancient dynasties. It discusses how different regions have developed their unique tea varieties and brewing techniques. The article also delves into the cultural significance of tea in Chinese society and how it has become a symbol of hospitality and relaxation.
verification:
{{"reason":"The context, while informative about the history and cultural significance of tea in China, does not provide specific information about the health benefits of green tea. Thus, it is not useful for answering the question about health benefits.", "verdict":"No"}}

question: How does photosynthesis work in plants?
context:
Photosynthesis in plants is a complex process involving multiple steps. This paper details how chlorophyll within the chloroplasts absorbs sunlight, which then drives the chemical reaction converting carbon dioxide and water into glucose and oxygen. It explains the role of light and dark reactions and how ATP and NADPH are produced during these processes.
verification:
{{"reason":"This context is extremely relevant and useful for answering the question. It directly addresses the mechanisms of photosynthesis, explaining the key components and processes involved.", "verdict":"Yes"}}

question:{question}
context:
{context}
verification:"""  # noqa: E501
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
    human_template: HumanMessagePromptTemplate = field(default_factory=lambda: CONTEXT_PRECISION)
    ai_template: AIMessagePromptTemplate = None

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list:
        prompts = []
        questions, contexts = dataset["question"], dataset["contexts"]


        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            for qstn, ctx in zip(questions, contexts):
                for c in ctx:
                    messages = [self.human_template.format(question=qstn, context=c)]
                    if self.ai_template is not None:
                        messages.append(self.ai_template.format())
                    prompts.append(ChatPromptTemplate.from_messages(messages))
            self.logs["prompts"] += prompts

            responses: list[list[str]] = []
            results = self.llm.generate(
                prompts,
                n=1,
                callbacks=batch_group,
            )

            responses = [[i.text for i in r] for r in results.generations]
            context_lens = [len(ctx) for ctx in contexts]
            context_lens.insert(0, 0)
            context_lens = np.cumsum(context_lens)
            grouped_responses = [
                responses[start:end]
                for start, end in zip(context_lens[:-1], context_lens[1:])
            ]
            self.logs["grouped_responses"] += grouped_responses

            scores = []
            for response in grouped_responses:
                response = [load_as_json(item) for item in sum(response, [])]
                self.logs["responses"].append(response)
                response = [
                    int("yes" in resp.get("verdict", " ").lower())
                    if resp.get("verdict")
                    else np.nan
                    for resp in response
                ]
                self.logs["responses_parsed"].append(response)
                denominator = sum(response) if sum(response) != 0 else 1e-10
                self.logs["denominator"].append(denominator)
                numerator = sum(
                    [
                        (sum(response[: i + 1]) / (i + 1)) * response[i]
                        for i in range(len(response))
                    ]
                )
                self.logs["numerator"].append(numerator)
                scores.append(numerator / denominator)

        self.logs["scores"] += scores
        return scores


context_precision = ContextPrecision()
