from __future__ import annotations

import sys
import logging
import typing as t
from collections import Counter
from dataclasses import dataclass, field

from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.llms import LangchainLLM
from ragas.metrics.base import EvaluationMode, MetricWithLLM, llm_factory

logger = logging.getLogger("Evaluation-Tab")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
logger.addHandler(handler)

CRITIQUE_PROMPT = HumanMessagePromptTemplate.from_template(
    """<instructions>
Given a input and submission. Evaluate the submission only using the given criteria. 
Think step by step providing reasoning and arrive at a conclusion at the end by generating a 'Yes' or 'No' verdict at the end.
</instructions>

<example_input>
<input>
Who was the director of Los Alamos Laboratory?
</input>

<submission>
Einstein was the director of  Los Alamos Laboratory.
</submission>

<criteria>
Is the output written in perfect grammar
</criteria>
</example_input>

<example_response>
Here are my thoughts: the criteria for evaluation is whether the output is written in perfect grammar. In this case, the output is grammatically correct. Therefore, the answer is:

Yes
</example_response>

Here is the input, submission and criteria:

<input>
{input}
</input>

<submission>
{submission}
</submission>

<criteria>
{criteria}
</criteria>

Assistant: Here are my thoughts and verdict:
"""  # noqa: E501
)


@dataclass
class AspectCritique(MetricWithLLM):
    """
    Judges the submission to give binary results using the criteria specified
    in the metric definition.

    Attributes
    ----------
    name: str
        name of the metrics
    definition: str
        criteria to judge the submission, example "Is the submission spreading
        fake information?"
    strictness: int
        The number of times self consistency checks is made. Final judgement is
        made using majority vote.
    batch_size: int
        Batch size for openai completion.
    llm : LangchainLLM
        llm API of your choice
    """

    name: str = field(default="", repr=True)
    evaluation_mode: EvaluationMode = EvaluationMode.qac
    definition: str = field(default="", repr=True)
    strictness: int = field(default=1, repr=False)
    batch_size: int = field(default=15, repr=False)
    llm: LangchainLLM = field(
        default_factory=llm_factory,
        repr=False,
    )

    latest_logs: dict = field(default_factory=dict)

    def __post_init__(self: t.Self):
        if self.name == "":
            raise ValueError("Expects a name")
        if self.definition == "":
            raise ValueError("Expects definition")

        # ensure odd number of checks to avoid tie in majority vote.
        self.strictness = (
            self.strictness if self.strictness % 2 != 0 else self.strictness + 1
        )

    def prompt_format(
        self: t.Self,
        question: str,
        answer: str,
        context: t.Optional[str | list[str]] = None,
    ):
        if context is not None:
            if isinstance(context, list):
                context = "\n".join(context)
            question = f"{question } answer using context: {context}"
        return CRITIQUE_PROMPT.format(
            input=question, submission=answer, criteria=self.definition
        )

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[int]:

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

        # Log criteria and strictness
        self._log_and_update('criteria', self.definition)
        self._log_and_update('strictness', self.strictness)

        prompts = []
        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            human_prompt_contents = []
            for n, (question, context, answer) in enumerate(zip(questions, contexts, answers)):
                human_prompt = self.prompt_format(question, answer, context)
                human_prompt_contents.append(human_prompt.content)
                # Log the human prompts
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))
            self._log_and_update('prompts', human_prompt_contents)

            results = self.llm.generate(
                prompts,
                n=self.strictness,
                callbacks=batch_group,
            )
            responses: list[list[str]] = [
                [i.text for i in r] for r in results.generations
            ]
            self._log_and_update('responses', responses[0])

            scores = []
            for n, response in enumerate(responses):
                response = [(text, text.split("\n\n")[-1]) for text in response]
                if self.strictness > 1:
                    score = Counter([1 if "yes" in item[-1].lower() else 0 for item in response]).most_common(1)[0][0]
                else:
                    score = 1 if "yes" in response[0][-1].lower() else 0
                self._log_and_update('score', score)
                scores.append(score)

        return scores

    def _log_and_update(self, key, value):
        """
        Helper method to log the addition of a new item to latest_logs.
        """
        self.latest_logs[key] = value
        #logger.debug(f"AspectCritique - {key}: {value}")