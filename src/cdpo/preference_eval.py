import random

from datasets import (
    Dataset
)

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import torch
from transformers import (
    pipeline,
)
from tqdm.auto import tqdm

PROMPT = """
For the following query to a chatbot, which response is more helpful?
Query: {QUERY}
Response A:
{RESPONSE_A}
Response B:
{RESPONSE_B}
FIRST provide a one-sentence comparison of the two responses and explain \
which you feel is more helpful. SECOND, on a new line, state only "A" or \
"B" to indicate which response is more helpful. Your response should use \
the format:
Comparison: <one-sentence comparison and explanation>
More helpful: <"A" or "B">
"""


def get_prompt(query, response_a, response_b) -> str:
    return PROMPT.format(
        QUERY=query, RESPONSE_A=response_a, RESPONSE_B=response_b
    )


# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def evaluate_completion(query, response_a, response_b, model="gpt-4o-mini"):
    """
    Evaluates two responses to a chat session.
    Inputs:
        query: the chat history
        response_a: One possible response to judge
        response_b: The other response to judge against
        model: OpenAI model to use
    Outputs:
        judgement: 'A' or 'B'
        response: full OpenAI response
    Raises:
        ValueError if OpenAI response is not valid
    """

    prompt = get_prompt(query, response_a, response_b)

    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "system",
            "content": [{
                "type": "text",
                "text": prompt
            }]
        }],
        temperature=0.0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    content = response.choices[0].message.content
    judgement = content[-1]
    if judgement not in ('A', 'B'):
        raise ValueError(
            f'Expected A or B, got {judgement}!\nFull message: {content}'
        )

    return judgement, response


@torch.no_grad()
def generate_responses(model, tokenizer, dataset: Dataset, device,
                       split_str="Assistant:", max_new_tokens=128) -> Dataset:
    """
    Generates responses for chat histories
    Inputs:
        model:
        tokenizer:
        dataset:
        device:
        split_str: separates last response from context
        max_new_tokens
    Returns:
        dataset: new dataset with fields:
            context: chat history
            new_response: generated response only
            chosen_response: chosen response only
            rejected_response: human rejected response only
    """
    model.eval()
    text_generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        device=device
    )

    def generate_response(example):
        context, rejected = example['rejected'].rsplit(split_str, 1)
        context2, chosen = example['chosen'].rsplit(split_str, 1)
        if len(context2) < len(context):
            context = context2

        prompt = context + "Assistant:"

        generated_text = text_generator(
            prompt, max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            return_full_text=False,
            temperature=0.7
        )

        example['new_response'] = generated_text[0]['generated_text'][1:]
        example['chosen_response'] = chosen[1:]
        example['context'] = context
        example['rejected_response'] = rejected[1:]

        return example

    return dataset.map(
        generate_response, batched=False,
        remove_columns=['chosen', 'rejected']
    )

def evaluate_win_rate(dataset: Dataset, key_1: str = 'new_response',
                      key_2: str = 'chosen_response'):
    """
    Use the OpenAI API to judge responses to a chat history
    Inputs:
        dataset
        key_1: field in dataset to use in comparison
        key_2: field in dataset to compare against
    Returns:
        win_rate: float
    """

    new_idx = []
    judgements = []

    fail_idxs = []

    for example in tqdm(dataset):
        if random.random() <= 0.5:
            response_a = example[key_1]
            response_b = example[key_2]
            new_idx.append(0)
        else:
            response_a = example[key_2]
            response_b = example[key_1]
            new_idx.append(1)

        try:
            judgement, api_response = evaluate_completion(
                example['context'], response_a, response_b
            )
            judgements.append(0 if judgement == 'A' else 1)
        except Exception as e:
            new_idx.pop()
            print(e)
            fail_idxs.append(len(new_idx) + len(fail_idxs))

    winners = torch.tensor(judgements) == torch.tensor(new_idx)
    win_rate = (winners).to(float).mean()

    return win_rate.item(), fail_idxs, winners.tolist()
