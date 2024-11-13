import re
from textwrap import dedent

from dotenv import load_dotenv
from litellm import completion

load_dotenv()


def text_pre_processing(text: str) -> str:
    """
    Preprocess the input text by removing extra newlines and dedenting.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    _tmp = text.strip("\n")
    # replace more than 2 newlines with 2 newlines
    _tmp = re.sub(r"\n{2,}", "\n\n", _tmp)
    # _tmp = text
    _tmp = dedent(_tmp)
    return _tmp


async def chat_with_llm(
    system_prompt: str, user_prompt: str, print_result: bool
) -> str:
    try:
        processed_system_prompt = text_pre_processing(system_prompt)
        processed_user_prompt = text_pre_processing(user_prompt)
        response = completion(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": processed_system_prompt},
                {"role": "user", "content": processed_user_prompt},
            ],
        )
        if print_result:
            print("=*" * 20)
            print("User: \n", processed_user_prompt)
            print("LLM: \n", response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM API request failed: {str(e)}"


async def translate_str_data_with_llm(
    text: str,
    target_lang: str,
    print_progress: bool = False,
    return_first_translation: bool = True,
) -> str:
    """Translate the input text to the target language using Ollama API."""

    if target_lang.lower() not in ("ja"):
        return {"ok": False, "message": "Ollama only supports Japanese translation."}

    system_prompt = (
        "You are a world-class translator and will translate English text to Japanese."
    )
    try:
        initial_translation = await chat_with_llm(
            system_prompt,
            text_pre_processing(
                """
            This is a English to Japanese, Literal Translation task.
            Please provide the Japanese translation for the next sentences.
            You must not include any chat messages to the user in your response.
            ---
            {original_text}
            """.format(original_text=text_pre_processing(text))
            ),
            print_progress,
        )

        # Disabling self-refinement for now, as it is a time-consuming process and
        # found not to be effective in most cases.
        if return_first_translation:
            return {
                "ok": True,
                "data": initial_translation,
            }

        review_comment = await chat_with_llm(
            system_prompt,
            text_pre_processing(
                """
            Orginal Text(English):
            {original_text}
            ---
            Translated Text(Japanese):
            {translated_text}
            ---
            Is there anything in the above Translated Text that does not conform to the local language's grammar, style, natural tone or cultural norms?
            Find mistakes and specify corrected phrase and why it is not appropriate.
            Each bullet should be in the following format:

            * <translated_phrase>
                * Corrected: <corrected_phrase>
                * Why: <reason>
            """.format(original_text=text, translated_text=initial_translation)
            ),
            print_progress,
        )

        final_translation = await chat_with_llm(
            system_prompt,
            text_pre_processing(
                """
            Orginal Text:
            {original_text}
            ---
            Hints for translation:
            {review_comments}
            ---
            Read the Original Text, and Hits for trasnlation above, then provide complete and accurate Japanese translation.
            You must not include any chat messages to the user in your response.
            """.format(
                    original_text=text,
                    # translated_text=text_pre_processing(initial_translation),
                    review_comments=review_comment,
                )
            ),
            print_progress,
        )

    except Exception as e:
        return {"ok": False, "message": f"LLM API request failed: {str(e)}"}

    return {
        "ok": True,
        "data": final_translation,
        # 'progress': {
        #     '01_init': initial_translation,
        #     '02_review': review_comment,
        #     '03_final': final_translation
        # }
    }
