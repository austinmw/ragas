from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass

from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)

#################
# NLI Score
#################
# LONG_FORM_ANSWER_PROMPT = HumanMessagePromptTemplate.from_template(
#     """\
# Given a question and answer, create one or more statements from each sentence in the given answer.
# question: Who was  Albert Einstein and what is he best known for?
# answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
# statements:\nAlbert Einstein was born in Germany.\nAlbert Einstein was best known for his theory of relativity.
# question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
# answer: alcohol
# statements:\nCadmium Chloride is slightly soluble in alcohol.
# question: Were Shahul and Jithin of the same nationality?
# answer: They were from different countries.
# statements:\nShahul and Jithin were from different countries.
# question:{question}
# answer: {answer}
# statements:\n"""  # noqa: E501
# )

LONG_FORM_ANSWER_PROMPT = HumanMessagePromptTemplate.from_template(
    """<instructions>
Given a question and answer, create one or more statements from each sentence in the given answer. Each sentence should be a standalone statement and includes a subject.
Follow the exact output format as shown in the example responses. Notice that there should not be any blank lines in your response!
</instructions>

<example_input>
<question>Who was  Albert Einstein and what is he best known for?</question>
<answer>He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.</answer>
</example_input>

<example_response>
statements:
Albert Einstein was born in Germany.
Albert Einstein was best known for his theory of relativity.
</example_response>

<example_input>
<question>Cadmium Chloride is slightly soluble in this chemical, it is also called what?</question>
<answer>alcohol</answer>
</example_input>

<example_response>
statements:
Cadmium Chloride is slightly soluble in alcohol.
</example_response>

<example_input>
<question>Were Shahul and Jithin of the same nationality?</question>
<answer>They were from different countries.</answer>
</example_input>

<example_response>
statements:
Shahul and Jithin were from different countries.
</example_response>


<question>{question}</question>
<answer>{answer}</answer>

Assistant: Here is the response for the above question and answer:
statements:
"""  # noqa: E501
)



# NLI_STATEMENTS_MESSAGE = HumanMessagePromptTemplate.from_template(
#     """
# Prompt: Natural language inference
# Consider the given context and following statements, then determine whether they are supported by the information present in the context.Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.

# Context:\nJohn is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
# statements:\n1. John is majoring in Biology.\n2. John is taking a course on Artificial Intelligence.\n3. John is a dedicated student.\n4. John has a part-time job.\n5. John is interested in computer programming.\n
# Answer:
# 1. John is majoring in Biology.
# Explanation: John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.  Verdict: No.
# 2. John is taking a course on Artificial Intelligence.
# Explanation: The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI. Verdict: No.
# 3. John is a dedicated student.
# Explanation: The prompt states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication. Verdict: Yes.
# 4. John has a part-time job.
# Explanation: There is no information given in the context about John having a part-time job. Therefore, it cannot be deduced that John has a part-time job.  Verdict: No.
# 5. John is interested in computer programming.
# Explanation: The context states that John is pursuing a degree in Computer Science, which implies an interest in computer programming. Verdict: Yes.
# Final verdict for each statement in order: No. No. Yes. No. Yes.
# context:\n{context}
# statements:\n{statements}
# Answer:
# """  # noqa: E501
# )
NLI_STATEMENTS_MESSAGE = HumanMessagePromptTemplate.from_template(
    """
<instructions>
Consider the given context and following statements, then determine whether they are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. 
Follow the exact output format as shown in the below example. Importantly, NEVER use two consecutive newlines in your response.
</instructions>

<example_input>
Context:\nJohn is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.</context>
Statements:\n1. John is majoring in Biology.\n2. John is taking a course on Artificial Intelligence.\n3. John is a dedicated student.\n4. John has a part-time job.\n5. John is interested in computer programming.
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

Context:\n{context}
Statements:\n{statements}

Assistant: Here is the answer in the exact example_response format without any consecutive newlines:
Answer:
"""  # noqa: E501
)


@dataclass
class Faithfulness(MetricWithLLM):
    name: str = "faithfulness"
    evaluation_mode: EvaluationMode = EvaluationMode.qac
    batch_size: int = 15

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
        prompts = []

        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            for n, (q, a) in enumerate(zip(question, answer)):
                human_prompt = LONG_FORM_ANSWER_PROMPT.format(question=q, answer=a)
                # Log human prompt
                logger.debug((f"Faithfulness: LONG_FORM_ANSWER_PROMPT human_prompt.content {n}:\n"
                              f"{human_prompt.content}"))
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            result = self.llm.generate(prompts, callbacks=batch_group)


            list_statements: list[list[str]] = []
            for output in result.generations:
                # Log result
                logger.debug((f"Faithfulness: LONG_FORM_ANSWER_PROMPT result {n}:\n"
                              f"{output[0].text}"))
                # use only the first generation for each prompt
                statements = output[0].text.split("\n")
                # Log parsed statements
                logger.debug((f"Faithfulness: LONG_FORM_ANSWER_PROMPT parsed statements"
                              f"{n}:\n{statements}"))
                list_statements.append(statements)

            prompts = []
            for n, (context, statements) in enumerate(zip(contexts, list_statements)):
                statements_str: str = "\n".join(
                    [f"{i+1}.{st}" for i, st in enumerate(statements)]
                )
                contexts_str: str = "\n".join(context)
                human_prompt = NLI_STATEMENTS_MESSAGE.format(
                    context=contexts_str, statements=statements_str
                )
                # Log human prompt
                logger.debug((f"Faithfulness: NLI_STATEMENTS_MESSAGE "
                              f"human_prompt.content {n}:\n{human_prompt.content}"))
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            result = self.llm.generate(prompts, callbacks=batch_group)
            #print(f"RESULT {len(result.generations)}:\n{result.generations[0][0].text}")
            outputs = result.generations

            scores = []
            final_answer = "Final verdict for each statement in order:"
            final_answer = final_answer.lower()
            for i, output in enumerate(outputs):
                # Log result
                logger.debug((f"Faithfulness: NLI_STATEMENTS_MESSAGE result {i}:\n"
                              f"{output[0].text}"))
                output = output[0].text.lower().strip()
                logger.debug((f"Faithfulness: NLI_STATEMENTS_MESSAGE parsed output"
                              f"{i}:\n{output}"))
                if output.find(final_answer) != -1:
                    logger.debug((f"Faithfulness: found '{final_answer}' in output {i}"))
                    output = output[output.find(final_answer) + len(final_answer) :]
                    logger.debug((f"Faithfulness: output {i} after removing the "
                                  f"final answer: {output}"))
                    score = sum(
                        0 if "yes" in answer else 1
                        for answer in output.strip().split(".")
                        if answer != ""
                    )
                    # Log the raw score before normalization
                    logger.debug((f"Faithfulness: raw score before normalization:"
                                  f"{score}"))

                    score = score / len(list_statements[i])

                    # Log the normalized score
                    logger.debug((f"Faithfulness: normalized score: {score}"))
                else:
                    # Log that the final answer was not found in the output
                    logger.debug((f"Faithfulness: '{final_answer}' not found in output."
                                  " Calculating score based on 'verdict: no'."))

                    score = max(0, output.count("verdict: no")) / len(
                        list_statements[i]
                    )

                # Log the score to be appended to scores list
                logger.debug((f"Faithfulness: score to be appended for statement{i}:"
                              f" {1 - score}"))

                scores.append(1 - score)

        return scores


faithfulness = Faithfulness()
