import numpy as np
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score
import itertools

# Załadowanie i przygotowanie danych
spam_dataset = pd.read_csv('data/spam.csv', encoding="ISO-8859-1", usecols=[0, 1], names=['Spam', 'Text'],
                           skiprows=1)
spam_dataset['Spam'] = spam_dataset['Spam'].replace(['ham', 'spam'], [0, 1])

# Usuwanie znaków specjalnych
def remove_punctuation(text):
    cleaned = ''.join([word for word in text if word not in string.punctuation])
    return cleaned

spam_dataset['Cleaned_Text'] = spam_dataset['Text'].apply(lambda x: remove_punctuation(x))

# Tokenizacja
def tokenize(text):
    clean_text = text.lower()
    tokenized_text = word_tokenize(clean_text)
    return tokenized_text  # Zwracamy listę słów zamiast stringa

spam_dataset['Tokenized_Text'] = spam_dataset['Cleaned_Text'].apply(tokenize)

# Usuwanie stopwords
stopwords = set(stopwords.words("english"))
def without_stopwords(text):
    return [word for word in text if word not in stopwords]

spam_dataset['WithoutStop_Text'] = spam_dataset['Tokenized_Text'].apply(lambda x: without_stopwords(x))

# Stemming
stemmer = PorterStemmer()
def stemming(text):
    stemmed_words = [stemmer.stem(word) for word in text]
    return stemmed_words

spam_dataset['Stemmed_Text'] = spam_dataset['WithoutStop_Text'].apply(stemming)

# Lematyzacja
lemmater = WordNetLemmatizer()
def lemmatizing(text):
    lemmatized_words = [lemmater.lemmatize(word) for word in text]
    return lemmatized_words

spam_dataset['Lemmatized_Text'] = spam_dataset['WithoutStop_Text'].apply(lemmatizing)

# Wektorizacja tekstów
vectorizer = TfidfVectorizer(max_features=5000)  # Zmniejszono liczbę cech
X_train = vectorizer.fit_transform(spam_dataset['Lemmatized_Text'].apply(lambda x: ' '.join(x)))
y_train = spam_dataset['Spam']

# Podział zbioru danych na treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Parametry do sprawdzenia
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Nauczanie modelu Random Forest z wyborem hiperparametrów
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

# Ocena modelu na zbiorze testowym
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
print(f"AUC Score: {auc}")

feature_importance = best_rf.feature_importances_
feature_names = vectorizer.get_feature_names_out()
feature_importance_df_best = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df_best = feature_importance_df_best.sort_values('importance', ascending=False)
print(feature_importance_df_best)

# Wygenerowanie chmury słów dla najlepszego modelu
words_spam = list(spam_dataset.loc[spam_dataset['Spam']==1, 'Lemmatized_Text'].values)
words_spam = list(itertools.chain.from_iterable(words_spam))
words_spam = ' '.join(words_spam)
words_notspam = list(spam_dataset.loc[spam_dataset['Spam']==0, 'Lemmatized_Text'].values)
words_notspam = list(itertools.chain.from_iterable(words_notspam))
words_notspam = ' '.join(words_notspam)

wordcloud = WordCloud().generate(words_spam)
plt.figure(figsize = (12, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.title('Spam')
plt.show()