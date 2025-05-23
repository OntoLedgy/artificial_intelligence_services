from services.tokenisation.general_tokeniser import GeneralTokeniser
from services.tokenisation.hugging_face_tokeniser import HuggingFaceTokeniser
from services.tokenisation.open_ai_tokeniser import OpenAiTokeniser
from services.tokenisation.tokeniser_types import TokeniserTypes


class TokeniserFactory:

    def __init__(
            self,
            tokeniser_type: TokeniserTypes,
            model_name: str):

        self.tokeniser_type = tokeniser_type
        self.model_name = model_name

    def get_tokeniser(self):

        match self.tokeniser_type:
            case TokeniserTypes.HUGGING_FACE:
                tokeniser = HuggingFaceTokeniser(
                    model_name=self.model_name)

            case TokeniserTypes.OPENAI:
                tokeniser = OpenAiTokeniser(
                    model_name=self.model_name)
            case _:
                raise ValueError(f"Unsupported tokeniser type: {self.tokeniser_type}")

        general_tokeniser = GeneralTokeniser(tokeniser)

        return general_tokeniser
