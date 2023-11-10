from __future__ import annotations

import sys
import logging
import typing as t
from dataclasses import dataclass, field

from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger("Evaluation-Tab")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
logger.addHandler(handler)

#################
# NLI Score
#################

LONG_FORM_ANSWER_PROMPT = HumanMessagePromptTemplate.from_template(
    """<instructions>
Given a question and answer, create one or more statements from each sentence in the given answer. Each sentence should be a standalone statement and includes a subject.
Follow the exact output format as shown in the example responses. Notice that there should not be any blank lines in your response!
</instructions>

<example_input>
<question>
Who was  Albert Einstein and what is he best known for?
</question>

<answer>
He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
</answer>
</example_input>

<example_response>
statements:
Albert Einstein was born in Germany.
Albert Einstein was best known for his theory of relativity.
</example_response>

<example_input>
<question>
Cadmium Chloride is slightly soluble in this chemical, it is also called what?
</question>

<answer>
alcohol
</answer>
</example_input>

<example_response>
statements:
Cadmium Chloride is slightly soluble in alcohol.
</example_response>

<example_input>
<question>
Were Shahul and Jithin of the same nationality?
</question>

<answer>
They were from different countries.
</answer>
</example_input>

<example_response>
statements:
Shahul and Jithin were from different countries.
</example_response>

Now here is the question and answer for you to create statements from:
<question>
{question}
</question>

<answer>
{answer}
</answer>

Assistant: Here is the response for the above question and answer:
statements:
"""  # noqa: E501
)


NLI_STATEMENTS_MESSAGE = HumanMessagePromptTemplate.from_template(
    """
<instructions>
Consider the given context and following statements, then determine whether they are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. 
Follow the exact output format as shown in the below example. Importantly, NEVER use two consecutive newlines in your response.
</instructions>

<example_input>
<context>
John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
</context>

<statements>
1. John is majoring in Biology.\n2. John is taking a course on Artificial Intelligence.\n3. John is a dedicated student.\n4. John has a part-time job.\n5. John is interested in computer programming.
</statements>
</example_input>

<example_response>
Answer:
1. John is majoring in Biology.
Explanation: John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.  Verdict: No.
2. John is taking a course on Artificial Intelligence.
Explanation: The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI. Verdict: No.
3. John is a dedicated student.
Explanation: The prompt states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication. Verdict: Yes.
4. John has a part-time job.
Explanation: There is no information given in the context about John having a part-time job. Therefore, it cannot be deduced that John has a part-time job.  Verdict: No.
5. John is interested in computer programming.
Explanation: The context states that John is pursuing a degree in Computer Science, which implies an interest in computer programming. Verdict: Yes.
Final verdict for each statement in order: No. No. Yes. No. Yes.
</example_response>

Now here is the context and statements for you to classify:

<context>
{context}
</context>

<statements>
{statements}
</statements>

Assistant: Here is the answer in the exact example_response format without any consecutive newlines:
Answer:
"""  # noqa: E501
)


@dataclass
class Faithfulness(MetricWithLLM):
    name: str = "faithfulness"
    evaluation_mode: EvaluationMode = EvaluationMode.qac
    batch_size: int = 15

    latest_logs: dict = field(default_factory=dict)

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        """
        returns the NLI score for each (q, c, a) pair
        """
        prompts = []
        self.latest_logs = {}
        questions = dataset["question"]
        ground_truths = dataset["ground_truths"]
        contexts = dataset["contexts"]
        answers = dataset["answer"]

        # Log each item added to latest_logs
        self._log_and_update('question', questions[0])
        self._log_and_update('ground_truth_answer', ground_truths[0])
        self._log_and_update('retrieved_documents', contexts[0])
        self._log_and_update('generated_answer', answers[0])
        self._log_and_update("num_contexts", len(contexts[0]))
        self._log_and_update('model_kwargs', self.llm.llm.model_kwargs)

        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:

            for n, (q, a) in enumerate(zip(questions, answers)):
                human_prompt = LONG_FORM_ANSWER_PROMPT.format(question=q, answer=a)
                self._log_and_update('first_prompt', human_prompt.content)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            result = self.llm.generate(prompts, callbacks=batch_group)

            list_statements: list[list[str]] = []
            for output in result.generations:
                self._log_and_update('first_result', output[0].text)
                statements = output[0].text.split("\n")
                list_statements.append(statements)
                self._log_and_update('statements', statements)

            prompts = []
            for n, (context, statements) in enumerate(zip(contexts, list_statements)):
                statements_str: str = "\n".join(
                    [f"{i+1}.{st}" for i, st in enumerate(statements)]
                )
                contexts_str: str = "\n".join(context)
                human_prompt = NLI_STATEMENTS_MESSAGE.format(
                    context=contexts_str, statements=statements_str
                )
                self._log_and_update('second_prompt', human_prompt.content)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            result = self.llm.generate(prompts, callbacks=batch_group)
            self._log_and_update('second_result', result.generations[0][0].text)
            outputs = result.generations

            scores = []
            final_answer = "Final verdict for each statement in order:"
            final_answer = final_answer.lower()

            outputs = [output[0].text.lower().strip() for output in outputs]
            assert len(outputs) == 1, "Only one response expected"
            output = outputs[0]
            self._log_and_update('parsed_second_result', output)

            if output.find(final_answer) != -1:
                self._log_and_update('found_final_answer', True)

                output = output[output.find(final_answer) + len(final_answer) :]

                answers = [1 if "yes" in answer else 0 for answer in output.split(".") if answer != ""]

                numerator = sum(answers)
                denominator = len(list_statements[0])
                score = numerator / denominator if denominator else 1e-6

            else:
                self._log_and_update('found_final_answer', False)

                numerator = output.count("verdict: yes")
                denominator = len(list_statements[0])
                score = numerator / denominator if denominator else 1e-6

            self._log_and_update('numerator', numerator)
            self._log_and_update('denominator', denominator)
            self._log_and_update('answer_faithfulness', score)

            scores.append(score)

        return scores

    def _log_and_update(self, key, value):
        """
        Helper method to log the addition of a new item to latest_logs.
        """
        self.latest_logs[key] = value
        #logger.debug(f"Faithfulness - {key}: {value}")
