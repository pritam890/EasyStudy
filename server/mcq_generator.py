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
from text import text

warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('brown')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pickle_model(path, loader_fn):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    model = loader_fn()
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return model

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

def postprocesstext(content):
    final=""
    for sent in sent_tokenize(content):
        sent = sent.capitalize()
        final = final +" "+sent
    return final

def summarizer(text, model=summary_model, tokenizer=summary_tokenizer):
    text = text.strip().replace("\n", " ")
    text = "summarize: "+text
    max_len = 512
    encoding = tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=3,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          min_length=75,
                          max_length=300)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    summary = postprocesstext(dec[0].strip())
    return summary

def get_nouns_multipartite(content):
    out=[]
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content, language='en')
        pos = {'PROPN', 'NOUN', 'ADJ', 'VERB'}
        stoplist = list(string.punctuation) + stopwords.words('english')
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyphrases = extractor.get_n_best(n=15)
        out = [val[0] for val in keyphrases]
    except:
        out = []
    return out

def get_keywords(text, top_n=20):
    raw_keywords = get_nouns_multipartite(text)
    if not raw_keywords:
        return []
    doc_embedding = sentence_transformer_model.encode([text])
    keyword_embeddings = sentence_transformer_model.encode(raw_keywords)
    distances = cosine_similarity(keyword_embeddings, doc_embedding)
    keyword_score_pairs = list(zip(raw_keywords, distances))
    keyword_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in keyword_score_pairs[:top_n]]

def get_question(context, answer, model, tokenizer):
    text = f"context: {context} answer: {answer}"
    encoding = tokenizer.encode_plus(text, max_length=384, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=5,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          max_length=72)
    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    return dec[0].replace("question:", "").strip().capitalize()

def filter_same_sense_words(original, wordlist):
    filtered_words = []
    base_sense = original.split('|')[1] 
    for eachword in wordlist:
        if eachword[0].split('|')[1] == base_sense:
            filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
    return filtered_words

def get_highest_similarity_score(wordlist, wrd):
    return max(textdistance.levenshtein.normalized_similarity(each.lower(), wrd.lower()) for each in wordlist)

def sense2vec_get_words(word, s2v, topn, question):
    output = []
    try:
        sense = s2v.get_best_sense(word)
        most_similar = s2v.most_similar(sense, n=topn)
        output = filter_same_sense_words(sense, most_similar)
    except:
        output = []

    threshold = 0.6
    final = [word]
    checklist = question.split()
    for x in output:
        if get_highest_similarity_score(final, x) < threshold and x not in final and x not in checklist:
            final.append(x)
    return final[1:]

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)
        mmr = lambda_param * candidate_similarities - (1 - lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def get_distractors_wordnet(word):
    distractors = []
    try:
        syn = wn.synsets(word, 'n')[0]
        hypernym = syn.hypernyms()
        if not hypernym: return distractors
        for item in hypernym[0].hyponyms():
            name = item.lemmas()[0].name()
            if name.lower() == word.lower(): continue
            distractors.append(" ".join(w.capitalize() for w in name.replace("_", " ").split()))
    except:
        print("Wordnet distractors not found")
    return distractors

def get_distractors(word, origsentence, sense2vecmodel, sentencemodel, top_n, lambdaval):
    distractors = sense2vec_get_words(word, sense2vecmodel, top_n, origsentence)
    if len(distractors) == 0:
        return distractors

    distractors_new = [word.capitalize()] + distractors
    embedding_sentence = origsentence + " " + word.capitalize()
    keyword_embedding = sentencemodel.encode([embedding_sentence])
    distractor_embeddings = sentencemodel.encode(distractors_new)
    max_keywords = min(len(distractors_new), 5)
    filtered_keywords = mmr(keyword_embedding, distractor_embeddings, distractors_new, max_keywords, lambdaval)
    final = [word.capitalize()] + [w.capitalize() for w in filtered_keywords if w.lower() != word.lower()]
    return final[1:]

def get_mcq(summarized_text):
    imp_keywords = get_keywords(summarized_text)
    used_answers = set()
    mcqs = []

    for answer in imp_keywords:
        if answer.lower() in used_answers:
            continue
        try:
            ques = get_question(summarized_text, answer, question_model, question_tokenizer)
            if len(ques.split()) < 5 or answer.lower() in ques.lower():
                continue
            distractors = get_distractors(answer.capitalize(), ques, s2v, sentence_transformer_model, 40, 0.2)
            if len(distractors) == 0:
                distractors = get_distractors_wordnet(answer)
            if len(distractors) == 0:
                fallback_keywords = [kw for kw in imp_keywords if kw.lower() != answer.lower()]
                distractors = random.sample(fallback_keywords, min(3, len(fallback_keywords)))
            if len(distractors) > 0:
                options = distractors[:3] + [answer]
                random.shuffle(options)
                options = [opt.strip().capitalize() for opt in options]
                mcqs.append({
                    "question": ques.strip().capitalize(),
                    "answer": answer.strip().capitalize(),
                    "options": options
                })
                used_answers.add(answer.lower())
        except Exception:
            traceback.print_exc()

    return mcqs

if __name__ == "__main__":
    summary = summarizer(text)
    print(f"Summarization: {summary} \n")
    mcq_list = get_mcq(summary)
    for i, mcq in enumerate(mcq_list, start=1):
        print(f"Q{i}: {mcq['question']}")
        for opt in mcq['options']:
            print(f" - {opt}")
        print(f"Answer: {mcq['answer']}\n{'-'*50}")
