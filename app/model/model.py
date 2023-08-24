import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LLM:
    def __init__(self, model_name, device="cpu"):
        # Model and tokenizer initialization
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_name, device)
        # BCP-47 codes for the 3 available languages + unknown language
        self.lang_codes = {
            "english": "en",
            "español": "es",
            "française": "fr",
            "unknown": "unk"}
        
    def load_model_and_tokenizer(self, model_name, device):
        # Configuration for quantization (only works on GPU)
        bnb_config = BitsAndBytesConfig(
            use_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            use_nested_quant=False,
        )
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Model and tokenizer loaded.")
        return model, tokenizer

    def language_detection(self, input_text):
        print(f"### Input text\n{input_text}")
        # Prompt with one shot for each language
        prompt = f"""Identify the language of the following sentences. Options: 'english', 'español', 'française' .
            * <Identity theft is not a joke, millions of families suffer every year>(english)
            * <Paseo a mi perro por el parque>(español)         
            * <J'ai vu trop de souris à Paris>(française)
            * <{input_text}>"""
        # Generation and extraction of the language tag
        answer_ids = self.model.generate(**self.tokenizer([prompt], return_tensors="pt"), max_new_tokens=10)
        answer = self.tokenizer.batch_decode(answer_ids, skip_special_tokens=False)[0]
        print(answer)
        generation = answer.split(prompt)[1]
        pattern = r'\b(?:' + '|'.join(map(re.escape, self.lang_codes.keys())) + r')\b'
        lang = re.search(pattern, generation, flags=re.IGNORECASE)
        # Returns tag identified or 'unk' if none is detected
        return self.lang_codes[lang.group()] if lang else self.lang_codes["unknown"]
    
    def entity_recognition(self, input_text):
        # Prompt design
        prompt = f"""Identify NER tags of 'location', 'organization', 'person' in the text.
        
        * Text: I saw Carmelo Anthony before the Knicks game in New York. Carmelo Anthony is retired now
        * Tags: <Carmelo Anthony>(person), <Knicks>(organization), <New York>(location), <Carmelo Anthony>(person)
        
        * Text: I will work from Spain for LanguageWire because Spain is warmer than Denmark
        * Tags: <Spain>(location), <LanguageWire>(organization), <Spain>(location), <Denmark>(location)
        
        * Text: Tesla founder Elon Musk is so rich that bought Twitter just for fun
        * Tags: <Tesla>(organization), <Elon Musk>(person), <Twitter>(organization)
        
        * Text: {input_text}
        * Tags: """
        print(prompt)
        # Generation and extraction of the identified entities
        answer_ids = self.model.generate(**self.tokenizer([prompt], return_tensors="pt"), max_new_tokens=100)
        answer = self.tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0].split(prompt)[1]
        entities = re.findall(r'<(.*?)>', answer)
        # Count of the tags detected (ignoring the type of entity)
        entities_count = {}
        for entity in entities:
            if entity in entities_count:
                entities_count[entity] += 1
            else:
                entities_count[entity] = 1
        # Returns a dictionary
        return entities_count