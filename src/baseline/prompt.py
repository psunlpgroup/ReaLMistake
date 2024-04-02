input_format = """
===== Model Input Begins =====

{input}

===== Model Input Ends =====

===== Model Response Begins =====

{response}

===== Model Response Ends =====
"""


first_sentence = "We provide an input to a language model (model input) and a response from the model (model response)."
instruction_following_definition = "You need to check whether the model response follows all instructions and requirements in the model input."

# the standard prompt
baseline_errordetection_prompt_type1 = f"""{first_sentence} Your task is to detect errors in the model response. {instruction_following_definition} You need to classify the model response as an error if the model response does not follow any of the instructions or requirements in the model input.
"""

# flip the order of the last two sentences
BASELINE_ERRORDETECTION_PROMPT_1 = baseline_errordetection_prompt_type1 + input_format + """
In your response, provide your explanation first and conclude your response with "Therefore, the model response contains an error." or "Therefore, the model response contains no error.""" + '"'

BASELINE_ERRORDETECTION_PROMPT_2 = baseline_errordetection_prompt_type1 + input_format + """
In your response, provide your explanation first and conclude your response with "Therefore, the model response contains no error." or "Therefore, the model response contains an error.""" + '"'


# prompts with different wording
baseline_errordetection_prompt_type2 = f"""{first_sentence} Your task is to evaluate the quality of the model response and classify whether it is valid or not. {instruction_following_definition} You need to classify the model response as valid only if the model response follows all instructions and requirements in the model input.
"""

# flip the order of the last two sentences
BASELINE_ERRORDETECTION_PROMPT_3 = baseline_errordetection_prompt_type2 + input_format + """
In your response, provide your explanation first and conclude your response with "Therefore, the model response is not valid." or "Therefore, the model response is valid.""" + '"'

BASELINE_ERRORDETECTION_PROMPT_4 = baseline_errordetection_prompt_type2 + input_format + """
In your response, provide your explanation first and conclude your response with "Therefore, the model response is valid." or "Therefore, the model response is not valid.""" + '"'


simple_prompt_baseline_prompts_dict = {
    "baseline_errordetection_prompt_1": BASELINE_ERRORDETECTION_PROMPT_1,
    "baseline_errordetection_prompt_2": BASELINE_ERRORDETECTION_PROMPT_2,
    "baseline_errordetection_prompt_3": BASELINE_ERRORDETECTION_PROMPT_3,
    "baseline_errordetection_prompt_4": BASELINE_ERRORDETECTION_PROMPT_4,
}


cot_instruction_prompt = BASELINE_ERRORDETECTION_PROMPT_1 + """

Follow the instructions below to check whether the model response contains an error:
1. Read the model input carefully.
2. Read the model response carefully.
3. Check whether the model response follows all instructions and requirements in the model input. Provide your explanation on each criterion.
4. Select your answer from "Therefore, the model response contains an error." or "Therefore, the model response contains no error.""" + '"'

advanced_prompt_dict = {
    "cot_instruction_prompt": cot_instruction_prompt,
}
