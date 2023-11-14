from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.utils import load_as_json

if t.TYPE_CHECKING:
    from datasets import Dataset


LONG_FORM_ANSWER_PROMPT = HumanMessagePromptTemplate.from_template(
    """\
Create one or more statements from each sentence in the given answer.

question: Who was  Albert Einstein and what is he best known for?
answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
statements in json:
{{
    "statements": [
        "Albert Einstein was born in Germany.",
        "Albert Einstein was best known for his theory of relativity."
    ]
}}

question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
answer: alcohol
statements in json:
{{
    "statements": [
        "Cadmium Chloride is slightly soluble in alcohol."
    ]
}}

question: Were Hitler and Benito Mussolini of the same nationality?
answer: Sorry, I can't provide answer to that question.
statements in json:
{{
    "statements": []
}}

question:{question}
answer: {answer}
statements in json:"""  # noqa: E501
)


NLI_STATEMENTS_MESSAGE = HumanMessagePromptTemplate.from_template(
    """
 Natural language inference. Only use "Yes" or "No" as verdict.

Context:
John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
statement_1: John is majoring in Biology.
statement_2: John is taking a course on Artificial Intelligence. 
statement_3: John is a dedicated student. 
statement_4: John has a part-time job.
Answer:
[
    {{
        "statement_1": "John is majoring in Biology.",
        "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
        "verdict": "No"
    }},
    {{
        "statement_2": "John is taking a course on Artificial Intelligence.",
        "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
        "verdict": "No"
    }},
    {{
        "statement_3": "John is a dedicated student.",
        "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
        "verdict": "Yes"
    }},
    {{
        "statement_4": "John has a part-time job.",
        "reason": "There is no information given in the context about John having a part-time job.",
        "verdict": "No"
    }}
]

Context:
Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.
statement_1: Albert Einstein was a genius.
Answer:
[
     {{
        "statement_1": "Albert Einstein was a genius.",
        "reason": "The context and statement are unrelated"
        "verdict": "No"
    }}
]

Context:
Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.
statement_1: Nil
Answer:
[
     {{
        "statement_1": "Nil",
        "reason": "The statement is invalid",
        "verdict": "No"
    }}
]


context:
{context}
statements:
{statements}
Answer:
"""  # noqa: E501
)


@dataclass
class Faithfulness(MetricWithLLM):
    name: str = "faithfulness"
    evaluation_mode: EvaluationMode = EvaluationMode.qac
    batch_size: int = 15
    statements_human_template: HumanMessagePromptTemplate = field(default_factory=lambda: LONG_FORM_ANSWER_PROMPT)
    statements_ai_template: AIMessagePromptTemplate = None
    verdict_human_template: HumanMessagePromptTemplate = field(default_factory=lambda: NLI_STATEMENTS_MESSAGE)
    verdict_ai_template: AIMessagePromptTemplate = None

    def _score_batch(
        self: t.Self,
        ds: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        """
        returns the NLI score for each (q, c, a) pair
        """

        question, answer, contexts = ds["question"], ds["answer"], ds["contexts"]

        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            prompts = []
            for q, a in zip(question, answer):
                human_prompt = self.statements_human_template.format(question=q, answer=a)
                messages = [human_prompt]
                if self.statements_ai_template is not None:
                    ai_prompt = self.statements_ai_template.format()
                    messages.append(ai_prompt)
                prompts.append(ChatPromptTemplate.from_messages(messages))
            self.logs["statement_prompts"] += prompts

            result = self.llm.generate(prompts, callbacks=batch_group)
            self.logs["statement_results"] += result

            prompts = []
            for context, output in zip(contexts, result.generations):
                statements = load_as_json(output[0].text).get("statements", [])
                statements = statements if statements != [] else ["Nil"]
                self.logs["statements"].append(statements)
                statements_str: str = "\n".join(
                    [f"statement_{i+1}: {st}" for i, st in enumerate(statements)]
                )
                self.logs["statements_str"].append(statements_str)
                contexts_str: str = "\n".join(context)
                self.logs["contexts_str"].append(contexts_str)
                human_prompt = self.verdict_human_template.format(
                    context=contexts_str, statements=statements_str
                )
                messages = [human_prompt]
                if self.verdict_ai_template is not None:
                    ai_prompt = self.verdict_ai_template.format()
                    messages.append(ai_prompt)
                prompts.append(ChatPromptTemplate.from_messages(messages))
            self.logs["verdict_prompts"] += prompts

            result = self.llm.generate(prompts, callbacks=batch_group)
            outputs = result.generations
            self.logs["verdict_result"] += outputs
            verdict_score_map = {"yes": 1, "no": 0, "null": np.nan}
            scores = []
            for output in outputs:
                output = load_as_json(output[0].text)
                output = output if output else []
                faithful_statements = sum(
                    verdict_score_map.get(dict.get("verdict", "").lower(), np.nan)
                    for dict in output
                )
                self.logs["faithful_statements"].append(faithful_statements)
                num_statements = len(output)
                self.logs["num_statements"].append(num_statements)
                if num_statements:
                    score = faithful_statements / num_statements
                else:
                    score = np.nan
                scores.append(score)

        self.logs["scores"] += scores
        return scores


faithfulness = Faithfulness()
