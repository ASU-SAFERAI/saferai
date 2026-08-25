from openai import OpenAI

from environs import moderation_model
from utils import Auth

Auth.setup_openai()


class Evaluator:
    """Content-safety scoring via the OpenAI omni-moderation endpoint.

    `provider` and `model_name` are accepted so callers can pass the values from
    the request body unchanged, but content moderation is served by a dedicated
    moderation model rather than a chat model, so they are not used to select it.
    Set the `moderation_model` environment variable to change the model.
    """

    def __init__(self, provider="openai", model_name="gpt-4o"):
        self.provider = provider
        self.model_name = model_name
        self.openAI_client = OpenAI()

    def ContentModeration(self, user_input):
        score = self.openAI_client.moderations.create(
            model=moderation_model, input=user_input
        ).to_dict()["results"][0]["category_scores"]
        return score
