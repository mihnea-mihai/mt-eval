import json
from translate import MTModel
import time
from sacrebleu.metrics import BLEU, CHRF, TER

DOMAINS = ['news', 'literary', 'legal_DE', 'medical']
MODEL_NAMES = ['OPUS', 'MBart50', 'LibreTranslate']


def add_translations():

  for domain in DOMAINS:
    models = {
          'LibreTranslate': MTModel(
              'LibreTranslate',
              data[domain]['SRC_lang'],
              data[domain]['REF_lang']
          ),
          'OPUS': MTModel(
              'OPUS',
              data[domain]['SRC_lang'],
              data[domain]['REF_lang']
          ),
          'MBart50': MTModel(
              'MBart50',
              data[domain]['SRC_lang'],
              data[domain]['REF_lang']
          )
      }

    for model_name in MODEL_NAMES:
      for sent in data[domain]['sents']:
        if not sent.get(model_name):
          sent[model_name] = {
              'output text': models[model_name].translate(sent['SRC'])
          }
          time.sleep(3)


def add_DA():
  for domain in DOMAINS:
    for model_name in MODEL_NAMES:
      for sent in data[domain]['sents']:
        if sent.get(model_name):
          if not sent.get(model_name).get('DA_adequacy'):
            print(f"\n\nOriginal sentence:\n\t{sent['SRC']}")
            print(f"\nReference translation:\n\t{sent['REF']}")
            print(f"\nTranslated sentence:\n\t{sent[model_name]['output text']}")
            try:
              sent[model_name]['DA_adequacy'] = float(
                input('\n\nAdequacy (x/10): ')) * 10
            except:
              break



def add_automatic_scores():
  bleu_c = BLEU()
  bleu_s = BLEU(effective_order=True)
  ter = TER()
  chrf = CHRF()

  for domain in DOMAINS:
    for model_name in MODEL_NAMES:
      if not data[domain]['sents'][0].get(model_name):
        break
      hypotheses = [sent[model_name]['output text']
        for sent in data[domain]['sents']]
      references = [[sent['REF'] for sent in data[domain]['sents']]]

      data[domain][model_name] = {
        'bleu': bleu_c.corpus_score(hypotheses,references).score,
        'chrf': chrf.corpus_score(hypotheses,references).score,
        'ter': ter.corpus_score(hypotheses,references).score
      }

      for sent in data[domain]['sents']:
        if sent.get(model_name):
          hypothesis = sent[model_name]['output text']
          references = [sent['REF']]
          sent[model_name].update({
            'bleu': bleu_s.sentence_score(hypothesis, references).score,
            'chrf': chrf.sentence_score(hypothesis, references).score,
            'ter': ter.sentence_score(hypothesis, references).score
          })
        
def avg_DA():
  for domain in DOMAINS:
    for model_name in MODEL_NAMES:
      if not data[domain]['sents'][0].get(model_name):
        break
      scores = [sent[model_name].get('DA_adequacy') for sent in data[domain]['sents']]
      data[domain][model_name]['DA_adequacy'] = sum(scores)/len(scores)


with open('data.json', encoding='utf-8') as file:
  data = json.load(file)

# add_translations()
# add_DA()
# add_automatic_scores()
avg_DA()


with open('data.json', 'w', encoding='utf-8') as file:
  json.dump(data, file, ensure_ascii=False)


