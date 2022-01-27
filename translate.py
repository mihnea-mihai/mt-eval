import requests
from transformers import MarianTokenizer, MarianMTModel
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class MTModel:
    def __init__(self, model_type, src_lang, trg_lang):

        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.model_type = model_type

        if model_type == 'OPUS':
            model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{trg_lang}"
            self.model = MarianMTModel.from_pretrained(model_name)
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            print('Created OPUS model')

        elif model_type == 'MBart50':
            self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            self.tokenizer.src_lang = self.lang_codes[src_lang]
            print('Created MBart50 model')
    
    def translate(self, src_text):

        if self.model_type == 'OPUS':
            batch = self.tokenizer([src_text], return_tensors="pt")
            gen = self.model.generate(**batch)
            print(f'Translated {src_text} with OPUS.')
            return self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

        elif self.model_type == 'LibreTranslate':
            response = requests.post(
                'https://libretranslate.de/translate',
                params={
                    'q': src_text,
                    'source': self.src_lang,
                    'target': self.trg_lang
                }
            )
            print(f'Translated {src_text} with LibreTranslate.')
            print(response.json())
            return response.json()['translatedText']

        elif self.model_type == 'MBart50':
            encoded = self.tokenizer(src_text, return_tensors="pt")
            generated_tokens = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.lang_codes[self.trg_lang]])
            print(f'Translated {src_text} with MBart50.')
            return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    lang_codes = {
        'ro': 'ro_RO',
        'en': 'en_XX',
        'de': 'de_DE'
    }
