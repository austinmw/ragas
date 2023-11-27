from langchain.prompts import AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate, AIMessagePromptTemplate

# Context Recall

CONTEXT_RECALL_HUMAN: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(  # noqa: E501
    """<instructions>
Given a question, context, and answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. You must only output valid json in the exact format shown in the below examples.
</instructions>

Here are a couple examples:

<question>
What can you tell me about albert Albert Einstein?
</question>

<context>
Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist,widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.
</context>

<answer>
answer: Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895 
</answer>

classification in json format:
[
    {{  "statement_1":"Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",
        "reason": "The date of birth of Einstein is mentioned clearly in the context.",
        "Attributed": "Yes"
    }},
    {{
        "statement_2":"He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics.",
        "reason": "The exact sentence is present in the given context.",
        "Attributed": "Yes"
    }},
    {{
        "statement_3": "He published 4 papers in 1905.",
        "reason": "There is no mention about papers he wrote in the given context.",
        "Attributed": "No"
    }},
    {{
        "statement_4":"Einstein moved to Switzerland in 1895.",
        "reason": "There is no supporting evidence for this in the given context.",
        "Attributed": "No"
    }}
]

<question>
who won 2020 icc world cup?
</question>

<context>
Who won the 2022 ICC Men's T20 World Cup?
The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.
</context>

<answer>
England
</answer>

classification in json format:
[
    {{
        "statement_1":"England won the 2022 ICC Men's T20 World Cup.",
        "reason": "From context it is clear that England defeated Pakistan to win the World Cup.",
         "Attributed": "Yes"
    }}
]

Now here is the question, context, and answer for you to classify:

<question>
{question}
</question>

<context>
{context}
</context>

<answer>
{answer}
</answer>
"""  # noqa: E501
)

CONTEXT_RECALL_AI: AIMessagePromptTemplate = AIMessagePromptTemplate.from_template(
    """classification in json format:
"""
)

# Context Precision

CONTEXT_PRECISION_HUMAN: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
    """<instructions>
Verify if the information in the given context is useful in answering the question. You must only output valid json in the exact format shown in the below examples.
</instructions>

Here are a couple examples:

<question>
What are the health benefits of green tea?
</question>

<context>
This article explores the rich history of tea cultivation in China, tracing its roots back to the ancient dynasties. It discusses how different regions have developed their unique tea varieties and brewing techniques. The article also delves into the cultural significance of tea in Chinese society and how it has become a symbol of hospitality and relaxation.
</context>

verification in json format:
{{"reason":"The context, while informative about the history and cultural significance of tea in China, does not provide specific information about the health benefits of green tea. Thus, it is not useful for answering the question about health benefits.", "verdict":"No"}}

<question>
How does photosynthesis work in plants?
</question>

<context>
Photosynthesis in plants is a complex process involving multiple steps. This paper details how chlorophyll within the chloroplasts absorbs sunlight, which then drives the chemical reaction converting carbon dioxide and water into glucose and oxygen. It explains the role of light and dark reactions and how ATP and NADPH are produced during these processes.
</context>

verification in json format:
{{"reason":"This context is extremely relevant and useful for answering the question. It directly addresses the mechanisms of photosynthesis, explaining the key components and processes involved.", "verdict":"Yes"}}

Now here is the question and context for you to verify:

<question>
{question}
</question>

<context>
{context}
</context>
"""  # noqa: E501
)

CONTEXT_PRECISION_AI: AIMessagePromptTemplate = AIMessagePromptTemplate.from_template(
    """verification in json format:
"""
)

# Answer Relevancy

ANSWER_RELEVANCY_HUMAN: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
    """<instructions>
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
"""  # noqa: E501
)

ANSWER_RELEVANCY_AI: AIMessagePromptTemplate = AIMessagePromptTemplate.from_template(
    """Question: """
)

# Faithfulness

FAITHFULNESS_STATEMENTS_HUMAN: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(  # noqa: E501
    """<instructions>
Given a question and answer, create one or more statements from each sentence in the given answer. You must only output valid json in the exact format shown in the below examples.
</instructions>


Here are a couple examples:

<question>
Who was  Albert Einstein and what is he best known for?
</question>

<answer>
He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
</answer>

statements in json:
{{
    "statements": [
        "Albert Einstein was born in Germany.",
        "Albert Einstein was best known for his theory of relativity."
    ]
}}

<question>
Cadmium Chloride is slightly soluble in this chemical, it is also called what?
</question>

<answer>
alcohol
</answer>

statements in json:
{{
    "statements": [
        "Cadmium Chloride is slightly soluble in alcohol."
    ]
}}

<question>
Were Hitler and Benito Mussolini of the same nationality?
</question>

<answer>
Sorry, I can't provide answer to that question.
</answer>

statements in json:
{{
    "statements": []
}}

Now here is the question and answer for you to create statements from:

<question>
{question}
</question>

<answer>
{answer}
</answer>
"""  # noqa: E501
)

FAITHFULNESS_STATEMENTS_AI: AIMessagePromptTemplate = AIMessagePromptTemplate.from_template(
    """statements in json:
"""
)

FAITHFULNESS_VERDICTS_HUMAN: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
    """<instructions>
Consider the given context and following statements, then determine whether they are supported by the information present in the context. You must only output valid json in the exact format shown in the below examples.
</instructions>

Here are a couple examples:

<context>
John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
</context>

<statements>
statement_1: John is majoring in Biology.
statement_2: John is taking a course on Artificial Intelligence.
statement_3: John is a dedicated student.
statement_4: John has a part-time job.
</statements>

json answer:
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

<context>
Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.
</context>

<statements>
statement_1: Albert Einstein was a genius.
</statements>

json answer:
[
     {{
        "statement_1": "Albert Einstein was a genius.",
        "reason": "The context and statement are unrelated"
        "verdict": "No"
    }}
]

Now here is the context and statements for you to classify:

<context>
{context}
</context>

<statements>
{statements}
</statements>
"""  # noqa: E501
)

FAITHFULNESS_VERDICTS_AI: AIMessagePromptTemplate = AIMessagePromptTemplate.from_template(
    """json answer:
"""
)

# Aspect Critique

ASPECT_CRITIQUE_HUMAN: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
    """<instructions>
Given an input and submission, evaluate the submission only using the given criteria.
Think step by step providing reasoning and arrive at a conclusion at the end by generating a single word 'Yes' or 'No' verdict at the end.
</instructions>

<example_input>
<input>Who was the director of Los Alamos Laboratory?</input>

<submission>Einstein was the director of  Los Alamos Laboratory.</submission>

<criteria>Is the output written in perfect grammar?</criteria>
</example_input>

<example_response>Here are my thoughts: the criteria for evaluation is whether the output is written in perfect grammar. In this case, the output is grammatically correct. Therefore, the answer is:

Yes</example_response>

<example_input>
<input>What is the capital of Italy?</input>

<submission>Rome is capital of Italy.</submission>

<criteria>Is the output written in perfect grammar</criteria>
</example_input>

<example_response>Evaluating this submission against the specified criteria of perfect grammar, we observe that the sentence "Rome is capital of Italy." lacks the necessary article 'the' before 'capital.' The grammatically correct form should be "Rome is the capital of Italy." Therefore, according to the criteria of perfect grammar, the verdict is:

No</example_response>

Now here is the input, submission and criteria for you to evaluate:

<input>{input}</input>

<submission>{submission}</submission>

<criteria>{criteria}</criteria>

Remember, it's critical that your verdict is a single word of either 'Yes' or 'No' and nothing else!
"""  # noqa: E501
)

ASPECT_CRITIQUE_AI: AIMessagePromptTemplate = AIMessagePromptTemplate.from_template(
    """Here are my thoughts followed by a single word "yes" or "no" verdict:
"""
)

ASPECT_GT_CRITIQUE_HUMAN: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
    """<instructions>
Given a question, generated answer, and ground truth answer, evaluate the generated answer using the given criteria.
Think step by step, providing reasoning, and conclude with a single word 'Yes' or 'No' verdict.
</instructions>

<example_input>
<question>What are the functions of Vitamin C in the human body? Answer using context: Vitamin C is commonly found in citrus fruits and is essential for health.</question>
<ground_truth_answer>Vitamin C plays various roles, including immune support and collagen synthesis, but further details are not provided in the context.</ground_truth_answer>
<generated_answer>Vitamin C is crucial for immune system function and helps prevent scurvy.</generated_answer>
<criteria>Did the generated answer correctly assess the sufficiency of the context to provide an accurate response?</criteria>
</example_input>

<example_response>The context mentions that Vitamin C is essential for health and found in citrus fruits, but it does not specify its functions. The generated answer states that Vitamin C supports the immune system and prevents scurvy. While these statements are true, they are not corroborated by the provided context. Therefore, the model did not correctly assess the context's sufficiency. The answer is:

No</example_response>

<example_input>
<question>Explain the role of enzyme X in the human digestive system. Answer using context: Enzyme X's function is currently under research.</question>
<ground_truth_answer>Enzyme X's specific role is not fully understood yet.</ground_truth_answer>
<generated_answer>There is insufficient context information to detail the exact role of enzyme X in the human digestive system.</generated_answer>
<criteria>Did the generated answer correctly assess the sufficiency of the context to provide an accurate response?</criteria>
</example_input>

<example_response>The ground truth indicates that the role of enzyme X is not fully understood, making the context insufficient for a detailed explanation. The generated answer correctly identifies this insufficiency. Hence, the answer is:

Yes</example_response>

<example_input>
<question>What are the primary colors in traditional color theory? Answer using context: Traditional color theory states that red, blue, and yellow are the primary colors, which cannot be created by mixing other colors.</question>
<ground_truth_answer>The primary colors in traditional color theory are red, blue, and yellow.</ground_truth_answer>
<generated_answer>Red, blue, and yellow are the primary colors in traditional color theory.</generated_answer>
<criteria>Did the generated answer correctly assess the sufficiency of the context to provide an accurate response?</criteria>
</example_input>

<example_response>The question asks for the primary colors in traditional color theory, and the provided context clearly states that these are red, blue, and yellow. The generated answer directly matches this information from the context, correctly identifying the primary colors as red, blue, and yellow. Given the context's sufficiency and the generated answer's accuracy, the answer is:

Yes</example_response>

<example_input>
<question>Who discovered penicillin? Answer using context: Penicillin was a groundbreaking discovery in medical science.</question>
<ground_truth_answer>Alexander Fleming discovered penicillin.</ground_truth_answer>
<generated_answer>The discovery of penicillin was made by Alexander Fleming.</generated_answer>
<criteria>Does the generated answer accurately match the ground truth answer?</criteria>
</example_input>

<example_response>Considering the question along with its context about penicillin's significance in medical science, the focus is on the discoverer of penicillin. The generated answer aligns with the ground truth, correctly identifying Alexander Fleming as the discoverer. Therefore, the answer is:

Yes</example_response>

Now here is the input, submission, ground truth, and criteria for you to evaluate:

<question>{input}</question>
<ground_truth_answer>{ground_truth}</ground_truth_answer>
<generated_answer>{submission}</generated_answer>
<criteria>{criteria}</criteria>

Remember, it's critical that your verdict is a single word of either 'Yes' or 'No' and nothing else!
"""  # noqa: E501
)
