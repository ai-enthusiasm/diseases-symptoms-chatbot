from transformers import pipeline, AutoTokenizer


TASK = 'token-classification'
MODEL = 'Clinical-AI-Apollo/Medical-NER'
AGGREGATION_STRATEGY = "SIMPLE"

tokenizer = AutoTokenizer.from_pretrained(MODEL, max_length=64, truncation=True)

pipe = pipeline(TASK, model=MODEL, aggregation_strategy=AGGREGATION_STRATEGY, tokenizer=tokenizer)
# output type
# ['entity_group', 'score', 'index', 'word', 'start', 'end']

user_info = {
        'AGE': [],
        'FAMILY_HISTORY':[],
        'HEIGHT':[],
        'FAMILY':[],
        'WEIGHT':[],
        'SEX':[]
    }

class extract_info():

    """
    Extracts information from user text.

    Args:
        user_text (str): The input text.
        user_info (dict): The user information dictionary.
        radius (int, optional): The radius for symptom extraction. Defaults to 1.
        use_radius_range (bool, optional): Whether to use radius range. Defaults to True.

    Returns:
        tuple: A tuple containing the updated user information and extracted symptoms.
    """

    def __init__(self, user_text: str, user_info: dict = user_info) -> tuple:
        self.user_text = user_text
        self.user_info = user_info
        self.key_group = "SIGN_SYMPTOM"
        self.related_groups = ["DETAILED_DESCRIPTION", "DISEASE_DISORDER", "BIOLOGICAL_STRUCTURE"]

    def get_token_info(self) -> list:
        return pipe(self.user_text)

    def extract_symptoms(self, token_info: list) -> list:
        symptoms = []
        for index, token in enumerate(token_info):
            token_type = token["entity_group"]
            if token_type in self.user_info.keys():
                user_info[token_type].append(token["word"])
                continue

            if token_type == self.key_group:
                symptoms.append(token["word"])

        return list(set(symptoms))









