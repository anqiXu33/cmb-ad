from num2words import num2words
import numpy as np
import copy



def get_user_prompt(mode, prompt_idx, verb_list, text_pred, word_limit, examples, language="English", preceding_ad=""):
    text_pred = f"\"{text_pred.strip()}\""

    # Format examples
    example_sentence = ""
    for selected_example in examples:
        example_sentence += "{\"summarized_AD\": \""+ f"{selected_example}" + "\"}\n"

    if mode == "single":
        if prompt_idx == 0 and verb_list is not None and language=="English":
            template = "{\"summarized_AD\": \"\"}"
            user_prompt = (
                "Please summarize the following description for one movie clip into ONE succinct audio description (AD) sentence.\n"
                f"Description: {text_pred}\n\n"
                "Focus on the most attractive characters, their actions, and related key objects (focus on point 2., supplemented by point 3.).\n"
                "For characters, use their first names, remove titles such as 'Mr.' and 'Dr.'. If names are not available, use pronouns such as 'He' and 'her', do not use expression such as 'a man'.\n"
                "For actions, avoid mentioning the camera, and do not focus on 'talking'.\n"
                "For objects, especially when no characters are involved, prioritize describing concrete and specific ones.\n"
                "Do not mention characters' mood.\n"
                "Do not hallucinate information that is not mentioned in the input.\n"
                f"Try to identify the following motions (with decreasing priorities): {verb_list}, and use them in the description.\n"
                "Provide the AD from a narrator perspective.\n\n"
                f"Output template (in JSON format): {template}.\n"
                f"Keep the length of the sentence in the JSON under {word_limit} words.\n"
                "Here are some example outputs:\n"
                f"{example_sentence}"
            )
        else:
            # special note on names: German -> physical descriptions when introducing characters before their name has been mentioned, such as 'Der Schwarzhaarige',but Italian -> uses name right away
            if language == "German":
                template = "{\"summarized_AD\": \"\"}"
                context = f"5. Context: '{preceding_ad}'."
                user_prompt = (
                    f"Please summarize the following description for one movie clip into ONE succinct audio description (AD) sentence, in {language}.\n"
                    f"Description: {text_pred}\n"
                    f"5. Context sentence (avoid repeating this): '{preceding_ad}'.\n\n"
                    "Focus on the most salient characters, their actions, and related key objects (focus on point 2., supplemented by point 3.).\n"
                    "For characters, use their first names, remove titles such as 'Mr.' and 'Dr.'. If names are not available, use descriptions such as 'der Schwarzhaarige' or 'eine junge Frau'.\n"
                    "Use a character's name if the name is not part of '5. Context sentence', otherwise use a pronoun. Example:\n"
                    "'Actions: Laura enters the kitchen and starts cooking., Context: Laura betritt die Küche.' -> 'Sie beginnt zu kochen.'\n"
                    "'Actions: Laura enters the kitchen and starts cooking., Context: Paul betritt die Küche.' -> 'Laura geht in die Küche und beginnt zu kochen.'\n"
                    "For actions, avoid mentioning the camera, and do not focus on 'talking'.\n"
                    "For objects, especially when no characters are involved, prioritize describing concrete and specific ones.\n"
                    "Do not mention characters' mood.\n"
                    "Provide the AD from a narrator perspective. Do not patronize the audience and stay neutral.\n"
                    "Never use words that indicate seeing like 'We see..' or '..is visible'."
                    "Avoid long composite words."
                    "Use succinct, vivid and imaginative language, but do not add information that is not in mentioned the input.\n"
                    "Use present tense. Use short, simple sentences with standard syntax.\n"
                    "If there is text on the screen that is relevant to the plot, or credits, use the text content. Do not mention 'Shot number'.\n"
                    f"Write all output in {language}, use Swiss spelling.\n\n"
                    f"Output template (in JSON format): {template}.\n"
                    f"Keep the length of the sentence in the JSON under {word_limit} words.\n"
                    "Here are some example outputs:\n"
                    f"{example_sentence}"
                )

            elif language == "Italian":
                template = "{\"summarized_AD\": \"\"}"
                context = f"5. Context: '{preceding_ad}'."
                user_prompt = (
                    f"Please summarize the following description for one movie clip into ONE succinct audio description (AD) sentence, in {language}.\n"
                    f"Description: {text_pred}\n\n"
                    f"5. Context sentence (avoid repeating this): '{preceding_ad}'.\n\n"
                    "Focus on the most attractive characters, their actions, and related key objects (focus on point 2., supplemented by point 3.).\n"
                    "For characters, use their first names, remove titles such as 'Mr.' and 'Dr.'. If names are not available, use a description such as 'una ragazza' or 'un uomo'.\n"
                    "Use a character's name if the name is not part of '5. Context sentence', otherwise use a pronoun. Example:\n"
                    "'Actions: Laura enters the kitchen and starts cooking., Context: Laura entra in cucina.' -> 'Lei inizia a cucinare.'\n"
                    "'Actions: Laura enters the kitchen and starts cooking., Context: Paul entra in cucina.' -> 'Laura entra in cucina e inizia a cucinare.'\n"
                    "For actions, avoid mentioning the camera, and do not focus on 'talking'.\n"
                    "For objects, especially when no characters are involved, prioritize describing concrete and specific ones.\n"
                    "Do not mention characters' mood.\n"
                    "Provide the AD from a narrator perspective. Do not patronize the audience and stay neutral.\n"
                    "Never use words that indicate seeing like 'We see..' or '..is visible'."
                    "Use succinct, vivid and imaginative language, but do not add information that is not in mentioned the input.\n"
                    "Use present tense. Use short, simple sentences with standard syntax.\n"
                    "If there is text on the screen that is relevant to the plot, or credits, use the text content. Do not mention 'Shot number'.\n"
                    f"Write all output in {language}, use Swiss spelling.\n\n"
                    f"Output template (in JSON format): {template}.\n"
                    f"Keep the length of the sentence in the JSON under {word_limit} words.\n"
                    "Here are some example outputs:\n"
                    f"{example_sentence}"
                )
            elif language == "French":
                template = "{\"summarized_AD\": \"\"}"
                context = f"5. Context: '{preceding_ad}'."
                user_prompt = (
                    f"Please summarize the following description for one movie clip into ONE succinct audio description (AD) sentence, in {language}.\n"
                    f"Description: {text_pred}\n\n"
                    f"5. Context sentence (avoid repeating this): '{preceding_ad}'.\n\n"
                    "Focus on the most attractive characters, their actions, and related key objects (focus on point 2., supplemented by point 3.).\n"
                    "For characters, use their first names, remove titles such as 'Mr.' and 'Dr.'. If names are not available, use a description such as 'una ragazza' or 'un uomo'.\n"
                    "Use a character's name if the name is not part of '5. Context sentence', otherwise use a pronoun. Example:\n"
                    "'Actions: Laura enters the kitchen and starts cooking., Context: Laura entre dans la cuisine.' -> 'Elle commence à cuisiner.'\n"
                    "'Actions: Laura enters the kitchen and starts cooking., Context: Paul entre dans la cuisine.' -> 'Laura entre dans la cuisine et commence à cuisiner.'\n"
                    "For actions, avoid mentioning the camera, and do not focus on 'talking'.\n"
                    "For objects, especially when no characters are involved, prioritize describing concrete and specific ones.\n"
                    "Do not mention characters' mood.\n"
                    "Provide the AD from a narrator perspective. Do not patronize the audience and stay neutral.\n"
                    "Never use words that indicate seeing like 'We see..' or '..is visible'."
                    "Use succinct, vivid and imaginative language, but do not add information that is not in mentioned the input.\n"
                    "Use present tense. Use short, simple sentences with standard syntax.\n"
                    "If there is text on the screen that is relevant to the plot, or credits, use the text content. Do not mention 'Shot number'.\n"
                    f"Write all output in {language}, use Swiss spelling.\n\n"
                    f"Output template (in JSON format): {template}.\n"
                    f"Keep the length of the sentence in the JSON under {word_limit} words.\n"
                    "Here are some example outputs:\n"
                    f"{example_sentence}"
                )

    else: # assistant mode
        if prompt_idx == 0:
            template = "{\"summarized_AD_1\": \"\",\n\"summarized_AD_2\": \"\",\n\"summarized_AD_3\": \"\",\n\"summarized_AD_4\": \"\",\n\"summarized_AD_5\": \"\"}"
            user_prompt = (
                "Please summarize the following description for one movie clip into ONE succinct audio description (AD) sentence.\n"
                f"Description: {text_pred}\n\n"
                "Focus on the most attractive characters, their actions, and related key objects (focus on point 2., supplemented by point 3.).\n"
                "For characters, use their first names, remove titles such as 'Mr.' and 'Dr.'. If names are not available, use pronouns such as 'He' and 'her', do not use expression such as 'a man'.\n"
                "For actions, avoid mentioning the camera, and do not focus on 'talking'.\n"
                "For objects, especially when no characters are involved, prioritize describing concrete and specific ones.\n"
                "Do not mention characters' mood.\n"
                "Do not hallucinate information that is not mentioned in the input.\n"
                f"Try to identify the following motions (with decreasing priorities): {verb_list}, and use them in the description.\n"
                "Provide 5 possible ADs from a narrator perspective, each offering a valid and distinct summary by emphasizing different key characters, actions, and movements present in the scene.\n"
                f"Limit the length of each output within {word_limit} words.\n\n"
                f"Output template (in JSON format): {template}.\n"
                "Here are some example outputs:\n"
                f"{example_sentence}"
            )

    return user_prompt


