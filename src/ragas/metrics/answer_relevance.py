from __future__ import annotations

import sys
import logging
import os
import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import trace_as_chain_group
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.exceptions import OpenAIKeyNotFound
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import CallbackManager

logger = logging.getLogger("Evaluation-Tab")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
logger.addHandler(handler)


QUESTION_GEN = HumanMessagePromptTemplate.from_template(
    """
<instructions>
Generate a question for the given answer. Follow the exact output format as shown in the example response. Do not add anything extra in the response!
</instructions>

<example_input>
<answer>The PSLV-C56 mission is scheduled to be launched on Sunday, 30 July 2023 at 06:30 IST / 01:00 UTC. It will be launched from the Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, India</answer>
</example_input>

<example_response>
Question: When is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?
</example_response>

Here is the input:
<answer>{answer}</answer>

Assistant:
Question: """  # noqa: E501
)


@dataclass
class AnswerRelevancy(MetricWithLLM):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Attributes
    ----------
    name: string
        The name of the metrics
    batch_size: int
        batch size for evaluation
    strictness: int
        Here indicates the number questions generated per answer.
        Ideal range between 3 to 5.
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy"
    evaluation_mode: EvaluationMode = EvaluationMode.qa
    batch_size: int = 15
    strictness: int = 3
    embeddings: Embeddings | None = None

    latest_logs: dict = field(default_factory=dict)

    def __post_init__(self: t.Self):
        if self.embeddings is None:
            oai_key = os.getenv("OPENAI_API_KEY", "no-key")
            self.embeddings = OpenAIEmbeddings(openai_api_key=oai_key)  # type: ignore

    def init_model(self):
        super().init_model()

        if isinstance(self.embeddings, OpenAIEmbeddings):
            if self.embeddings.openai_api_key == "no-key":
                raise OpenAIKeyNotFound

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
        assert len(answers)  == 1, "Only one answer is allowed per question"

        # Log each item added to latest_logs
        self._log_and_update('question', questions[0])
        self._log_and_update('ground_truth_answer', ground_truths[0])
        self._log_and_update('retrieved_documents', contexts[0])
        self._log_and_update('generated_answer', answers[0])
        self._log_and_update('model_kwargs', self.llm.llm.model_kwargs)

        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            prompts = []
            human_prompt_contents = []
            for n, ans in enumerate(answers):
                human_prompt = QUESTION_GEN.format(answer=ans)
                human_prompt_contents.append(human_prompt.content)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            self._log_and_update('prompts', human_prompt_contents)

            results = self.llm.generate(
                prompts,
                n=self.strictness,
                callbacks=batch_group,
            )
            results = [[i.text for i in r] for r in results.generations]

            scores = []
            for n, (question, gen_questions) in enumerate(zip(questions, results)):
                cosine_similary_list = self.calculate_similarity(question, gen_questions)
                self._log_and_update('cosine_similary_list', cosine_similary_list.tolist())
                cosine_sim_mean = cosine_similary_list.mean()
                self._log_and_update('cosine_similarity', cosine_sim_mean)
                scores.append(cosine_sim_mean)

        return scores

    def calculate_similarity(
        self: t.Self, question: str, generated_questions: list[str]
    ):
        assert self.embeddings is not None
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)

        self._log_and_update('generated_questions', generated_questions)

        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)
        )
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )

    def _log_and_update(self, key, value):
        """
        Helper method to log the addition of a new item to latest_logs.
        """
        self.latest_logs[key] = value
        #logger.debug(f"AnswerRelevancy - {key}: {value}")
