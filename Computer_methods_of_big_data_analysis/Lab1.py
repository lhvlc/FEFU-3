import re
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')

#функция для предварительной обработки текста
def preprocess_text(text):
    #перевод текста в нижний регистр
    text = text.lower()

    #удаление знаков препинания
    text = text.translate(str.maketrans('', '', string.punctuation))

    #удаление стоп-слов
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    #удаление лишних символов
    text = re.sub(r'', '', text)

    return text

#функция для преобразования текста в мешок слов и N-граммы
def create_bow_and_ngrams(text, n_grams):
    vectorizer = CountVectorizer(ngram_range=(n_grams, n_grams))
    ngram_matrix = vectorizer.fit_transform([text])

    return vectorizer.get_feature_names_out(), ngram_matrix.toarray()

text = "London is the capital and most populous city of England and the United Kingdom. Standing on the River Thames in the south east of the island of Great Britain, London has been a major settlement for two millennia. It was founded by the Romans, who named it Londinium."

#предварительная обработка текста
processed_text = preprocess_text(text)
print("Обработанный текст:\n", processed_text)

#создание мешка слов
#N = 1
bow_features, bow_matrix = create_bow_and_ngrams(processed_text, 1)
print("Мешок слов:\n", bow_features)
print("Матрица мешка слов:\n", bow_matrix)

#cоздание N-грамм (N = 2, 4)
for n in [2, 4]:
    ngram_features, ngram_matrix = create_bow_and_ngrams(processed_text, n)
    print(f"N-граммы (N={n}):\n", ngram_features)
    print(f"Матрица N-грамм (N={n}):\n", ngram_matrix)