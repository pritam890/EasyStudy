import os
import pickle
import torch
import string
import traceback
import random
import numpy as np
import nltk
import pke
import textdistance
import warnings

from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import sent_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sense2vec import Sense2Vec
from flashtext import KeywordProcessor
from sklearn.metrics.pairwise import cosine_similarity

# Ignore warnings
warnings.filterwarnings("ignore")

# Download necessary nltk data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('brown')

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Utility to load or save a model via pickle
def load_pickle_model(path, loader_fn):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    model = loader_fn()
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return model

# Load models
summary_model = load_pickle_model(
    "models/t5_summary_model.pkl",
    lambda: T5ForConditionalGeneration.from_pretrained("t5-base")
).to(device)

summary_tokenizer = load_pickle_model(
    "models/t5_summary_tokenizer.pkl",
    lambda: T5Tokenizer.from_pretrained("t5-base")
)

question_model = load_pickle_model(
    "models/t5_question_model.pkl",
    lambda: T5ForConditionalGeneration.from_pretrained("ramsrigouthamg/t5_squad_v1")
).to(device)

question_tokenizer = load_pickle_model(
    "models/t5_question_tokenizer.pkl",
    lambda: T5Tokenizer.from_pretrained("ramsrigouthamg/t5_squad_v1")
)

sentence_transformer_model = load_pickle_model(
    "models/sentence_transformer_model.pkl",
    lambda: SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v2")
)

s2v = Sense2Vec().from_disk('models/s2v_old')

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def postprocesstext (content):
  """
  this function takes a piece of text (content), tokenizes it into sentences, capitalizes the first letter of each sentence, and then concatenates the processed sentences into a single string, which is returned as the final result. The purpose of this function could be to format the input content by ensuring that each sentence starts with an uppercase letter.
  """
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final

# Summary Function
def summarizer(text,model=summary_model,tokenizer=summary_tokenizer):
  """
  This function takes the given text along with the model and tokenizer, which summarize the large text into useful information
  """
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  # print (text)
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 75,
                                  max_length=300)

  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()

  return summary

# Keyword Generation 
def get_nouns_multipartite(content):
    """
    This function takes the content text given and then outputs the phrases which are build around the nouns , so that we can use them for context based distractors
    """
    out=[]
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content,language='en')
        pos = {'PROPN', 'NOUN', 'ADJ', 'VERB', 'ADP', 'ADV', 'DET', 'CONJ', 'NUM', 'PRON', 'X'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection( pos=pos)
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
        keyphrases = extractor.get_n_best(n=15)
        

        for val in keyphrases:
            out.append(val[0])
    except:
        out = []

    return out

def get_keywords(originaltext):
  """
  This function takes the original text and the summary text and generates keywords from both which ever are more relevant
  This is done by checking the keywords generated from the original text to those generated from the summary, so that we get important ones 
  """
  keywords = get_nouns_multipartite(originaltext)
  return keywords

# Generate Questions
def get_question(context,answer,model,tokenizer):
  """
  This function takes the input context text, pretrained model along with the tokenizer and the keyword and the answer and then generates the question from the large paragraph
  """
  text = "context: {} answer: {}".format(context,answer)
  encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question

# Distractors Generator
def filter_same_sense_words(original,wordlist):
  
  """
  This is used to filter the words which are of same sense, where it takes the wordlist which has the sense of the word attached as the string along with the word itself.
  """
  filtered_words=[]
  base_sense =original.split('|')[1] 
  for eachword in wordlist:
    if eachword[0].split('|')[1] == base_sense:
      filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
  return filtered_words

def get_highest_similarity_score(wordlist,wrd):
  """
  This function takes the given word along with the wordlist and then gives out the max-score which is the levenshtein distance for the wrong answers
  because we need the options which are very different from one another but relating to the same context.
  """
  score=[]
  for each in wordlist:
    score.append(textdistance.levenshtein.normalized_similarity(each.lower(),wrd.lower()))
  return max(score)

def sense2vec_get_words(word,s2v,topn,question):
    """
    This function takes the input word, sentence to vector model and top similar words and also the question
    Then it computes the sense of the given word
    then it gets the words which are of same sense but are most similar to the given word
    after that we we return the list of words which satisfy the above mentioned criteria
    """
    output = []
    try:
      sense = s2v.get_best_sense(word, senses= ["NOUN", "PERSON","PRODUCT","LOC","ORG","EVENT","NORP","WORK OF ART","FAC","GPE","NUM","FACILITY"])
      most_similar = s2v.most_similar(sense, n=topn)
      output = filter_same_sense_words(sense,most_similar)
    except:
      output =[]

    threshold = 0.6
    final=[word]
    checklist =question.split()
    for x in output:
      if get_highest_similarity_score(final,x)<threshold and x not in final and x not in checklist:
        final.append(x)
    
    return final[1:]

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
    """
    The mmr function takes document and word embeddings, along with other parameters, and uses the Maximal Marginal Relevance (MMR) algorithm to extract a specified number of keywords/keyphrases from the document. The MMR algorithm balances the relevance of keywords with their diversity, helping to select keywords that are both informative and distinct from each other.
    """

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphrase
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def get_distractors_wordnet(word):
    """
    the get_distractors_wordnet function uses WordNet to find a relevant synset for the input word and then generates distractor words by looking at hyponyms of the hypernym associated with the input word. These distractors are alternative words related to the input word and can be used, for example, in educational or language-related applications to provide choices for a given word.
    """
    distractors=[]
    try:
      syn = wn.synsets(word,'n')[0]
      
      word= word.lower()
      orig_word = word
      if len(word.split())>0:
          word = word.replace(" ","_")
      hypernym = syn.hypernyms()
      if len(hypernym) == 0: 
          return distractors
      for item in hypernym[0].hyponyms():
          name = item.lemmas()[0].name()
          if name == orig_word:
              continue
          name = name.replace("_"," ")
          name = " ".join(w.capitalize() for w in name.split())
          if name is not None and name not in distractors:
              distractors.append(name)
    except:
      print ("Wordnet distractors not found")
    return distractors

def get_distractors (word,origsentence,sense2vecmodel,sentencemodel,top_n,lambdaval):
  """
  this function generates distractor words (answer choices) for a given target word in the context of a provided sentence. It selects distractors based on their similarity to the target word's context and ensures that the target word itself is not included among the distractors. This function is useful for creating multiple-choice questions or answer options in natural language processing tasks.
  """
  distractors = sense2vec_get_words(word,sense2vecmodel,top_n,origsentence)
  if len(distractors) ==0:
    return distractors
  distractors_new = [word.capitalize()]
  distractors_new.extend(distractors)


  embedding_sentence = origsentence+ " "+word.capitalize()
  keyword_embedding = sentencemodel.encode([embedding_sentence])
  distractor_embeddings = sentencemodel.encode(distractors_new)

  max_keywords = min(len(distractors_new),5)
  filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors_new,max_keywords,lambdaval)
  final = [word.capitalize()]
  for wrd in filtered_keywords:
    if wrd.lower() !=word.lower():
      final.append(wrd.capitalize())
  final = final[1:]
  return final

def get_mcq(context: str, summarized_text=None):
    """
    This function generates multiple-choice questions based on a given context.
    It summarizes the context, extracts important keywords, generates questions
    related to those keywords, and provides randomized answer choices,
    including the correct answer, for each question.
    """
    if summarized_text is None:
        summarized_text = summarizer(context, summary_model, summary_tokenizer)

    # Extract important keywords
    imp_keywords = get_keywords(context)
    
    mcqs = []
    for answer in imp_keywords:
        try:
            ques = get_question(summarized_text, answer, question_model, question_tokenizer)

            distractors = get_distractors(answer.capitalize(), ques, s2v, sentence_transformer_model, 40, 0.2)

            if len(distractors) == 0:
               distractors=imp_keywords

            options = []
            if len(distractors) > 0 :
                options = distractors[:3] + [answer]
                random.shuffle(options) 
                  

            mcqs.append({
                "question": ques,
                "answer": answer,
                "options": options
            })
        except Exception:
            traceback.print_exc()

    return mcqs

# Example usage
# if __name__ == "__main__":
#     input_text = (
#         "Automobili Lamborghini, the illustrious Italian manufacturer of luxury sports cars and SUVs, is headquartered in the picturesque Sant'Agata Bolognese. This renowned automotive institution boasts a storied legacy, and its contemporary success is firmly underpinned by a fascinating history that has seen it evolve through ownership changes, economic downturns, and groundbreaking innovations.\
# Ferruccio Lamborghini, a prominent Italian industrialist with a passion for automobiles, laid the foundation for this iconic marque in 1963. His vision was audacious - to challenge the supremacy of Ferrari, the undisputed titan of Italian sports cars. Under Ferruccio's guidance, Automobili Ferruccio Lamborghini S.p.A. was established, and it immediately began making waves in the automotive world.\
# One of the hallmarks of Lamborghini's early years was its distinctive rear mid-engine, rear-wheel-drive layout. This design philosophy became synonymous with Lamborghini's commitment to creating high-performance vehicles. The company's inaugural models, such as the 350 GT, arrived in the mid-1960s and showcased Lamborghini's dedication to precision engineering and uncompromising quality.\
# Lamborghini's ascendancy was nothing short of meteoric during its formative decade. It consistently pushed the boundaries of automotive technology and design. However, the heady days of growth were met with a sudden downturn when the world faced the harsh realities of the 1973 global financial crisis and the subsequent oil embargo. Lamborghini, like many other automakers, grappled with plummeting sales and financial instability.\
# Ownership of Lamborghini underwent multiple transitions in the wake of these challenges. The company faced bankruptcy in 1978, marking a turbulent chapter in its history. The ownership baton changed hands several times, with different entities attempting to steer the storied brand to calmer waters.\
# In 1987, American automaker Chrysler Corporation took the helm at Lamborghini. The Chrysler era saw Lamborghini continue to produce remarkable vehicles like the Diablo while operating under the umbrella of a global conglomerate. However, it was not a permanent arrangement.\
# In 1994, Malaysian investment group Mycom Setdco and Indonesian group V'Power Corporation acquired Lamborghini, signaling another phase of transformation for the company. These new custodians brought fresh perspectives and investment to the brand, fueling its resurgence.\
# A significant turning point occurred in 1998 when Mycom Setdco and V'Power sold Lamborghini to the Volkswagen Group, which placed the Italian marque under the stewardship of its Audi division. This move brought newfound stability and resources, ensuring Lamborghini's enduring presence in the luxury sports car arena.\
# Over the ensuing years, Lamborghini witnessed remarkable expansions in its product portfolio. The V10-powered Huracán captured the hearts of sports car enthusiasts with its exquisite design and formidable performance. Simultaneously, Lamborghini ventured into the SUV market with the Urus, a groundbreaking vehicle powered by a potent twin-turbo V8 engine. This diversification allowed Lamborghini to cater to a broader range of customers without compromising on its commitment to luxury and performance.\
# While these successes were noteworthy, Lamborghini was not immune to the challenges posed by global economic fluctuations. In the late 2000s, during the worldwide financial crisis and the subsequent economic downturn, Lamborghini's sales experienced a significant decline, illustrating the brand's vulnerability to external economic factors.\
# Despite these challenges, Lamborghini maintained its relentless pursuit of automotive excellence. The company's flagship model, the V12-powered Aventador, reached the pinnacle of automotive engineering and design before concluding its production run in 2022. However, the story does not end here. Lamborghini is set to introduce the Revuelto, a V12/electric hybrid model, in 2024, exemplifying its commitment to embracing cutting-edge technologies and pushing the boundaries of performance.\
# In addition to its road car production, Lamborghini has made notable contributions to other industries. The company manufactures potent V12 engines for offshore powerboat racing, further underscoring its prowess in high-performance engineering.\
# Interestingly, Lamborghini's legacy extends beyond the realm of automobiles. Ferruccio Lamborghini founded Lamborghini Trattori in 1948, a separate entity from the automobile manufacturer, which continues to produce tractors to this day.\
# Lamborghini's rich history is also intertwined with the world of motorsport. In a stark contrast to his rival Enzo Ferrari, Ferruccio Lamborghini decided early on not to engage in factory-supported racing, considering it too expensive and resource-intensive. Nonetheless, Lamborghini's engineers, many of whom were passionate about racing, embarked on ambitious projects, including the development of the iconic Miura sports coupe, which possessed racing potential while being road-friendly. This project marked a pivotal moment in Lamborghini's history, showcasing its ability to create vehicles that could excel on both the track and the road.Despite Ferruccio's reluctance, Lamborghini did make some forays into motorsport. In the mid-1970s, while under the management of Georges-Henri Rossetti, Lamborghini collaborated with BMW to develop and manufacture 400 cars for BMW, a venture intended to meet Group 4 homologation requirements. However, due to financial instability and delays in development, BMW eventually took control of the project, finishing it without Lamborghini's involvement.\
# Lamborghini also briefly supplied engines to Formula One teams from 1989 to 1993. Teams like Larrousse, Lotus, Ligier, Minardi, and Modena utilized Lamborghini power units during this period. Lamborghini's best result in Formula One was achieved when Aguri Suzuki finished third at the 1990 Japanese Grand Prix.\
# In addition to Formula One, Lamborghini was involved in other racing series. Notably, racing versions of the Diablo were developed for the Diablo Supertrophy, a single-model racing series that ran from 1996 to 1999. The Murciélago R-GT, a production racing car, was created to compete in events like the FIA GT Championship and the American Le Mans Series in 2004, achieving notable results in its racing endeavors.\
# Lamborghini's connection with motorsport reflects the brand's commitment to engineering excellence, even though it shied away from factory-backed racing for much of its history.\
# Beyond the realms of automotive engineering, Lamborghini has carved a distinct niche in the world of branding. The company licenses its prestigious brand to manufacturers who produce a wide array of Lamborghini-branded consumer goods, including scale models, clothing, accessories, bags, electronics, and even laptop computers. This strategic approach has enabled Lamborghini to extend its brand reach beyond the confines of the automotive industry.\
# One fascinating aspect of Lamborghini's identity is its deep connection with the world of bullfighting. In 1962, Ferruccio Lamborghini visited the ranch of Don Eduardo Miura, a renowned breeder of Spanish fighting bulls. Impressed by the majestic Miura animals, Ferruccio decided to adopt a raging bull as the emblem for his burgeoning automaker. This emblem, now iconic, symbolizes Lamborghini's passion for performance, power, and the thrill of the chase.\
# Lamborghini's vehicle nomenclature also reflects this bullfighting heritage, with many models bearing the names of famous fighting bulls or bull-related themes. The Miura, named after the Miura bulls, set the precedent, and subsequent models like the Murciélago, Gallardo, and Aventador continued this tradition.\
# Furthermore, Lamborghini has enthusiastically embraced emerging automotive technologies, responding to environmental concerns and changing consumer preferences. The Sian, introduced as the company's first hybrid model, showcases Lamborghini's commitment to sustainable performance. With its innovative hybrid powertrain, the Sian combines electric propulsion with a naturally aspirated V12 engine to deliver breathtaking performance while minimizing emissions.\
# Looking ahead, Lamborghini has ambitious plans to produce an all-electric vehicle, aligning with the broader industry trend towards electrification. While traditionalists may lament the absence of roaring V12 engines, Lamborghini recognizes the importance of evolving with the times, ensuring that future generations of enthusiasts can experience the thrill of a Lamborghini while contributing to a more sustainable future.\
# In summary, Automobili Lamborghini stands as a testament to the enduring allure of Italian craftsmanship and automotive excellence. From its audacious beginnings as a challenger to Ferrari, Lamborghini has weathered storms, embraced innovation, and left an indelible mark on the world of sports cars. Its legacy is one of design brilliance, relentless pursuit of power, and a commitment to pushing the boundaries of what's possible in the realm of high-performance automobiles. Whether through its iconic V12-powered supercars, groundbreaking hybrids, or electrifying visions of the future, Lamborghini continues to captivate the hearts of automotive enthusiasts worldwide, cementing its status as a legendary and iconic brand."
#     )

#     mcq_list = get_mcq(input_text)

#     for i, mcq in enumerate(mcq_list, start=1):
#         print(f"Q{i}: {mcq['question']}")
#         for opt in mcq['options']:
#             print(f" - {opt}")
#         print(f"Answer: {mcq['answer']}\n{'-'*50}")