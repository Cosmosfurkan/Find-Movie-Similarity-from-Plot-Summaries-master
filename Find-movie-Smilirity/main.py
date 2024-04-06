from matplotlib.pyplot import stem
import library
import numpy as np

# Set seed for reproducibility
library.np.random.seed(0)

# Load dataset
movie = library.pd.read_csv("C:/Users/furkan/Desktop/Yapay zeka/Veri setleri/movies.csv")
#print(movie.head())

# Combine wiki_plot and imdb_plot into a single column
movie["plot"] = movie["wiki_plot"].astype(str) + "\n" + movie["imdb_plot"].astype(str)
#print(movie.head())

library.nl.download('punkt')

# Tokenize a paragraph into sentences and store in sent_tokenized
sent_tokenized = [sent for sent in library.sent_tokenize("""
                        Today (May 19, 2016) is his only daughter's wedding. 
                        Vito Corleone is the Godfather.
                        """)]

# Word Tokenize first sentence from sent_tokenized, save as words_tokenized
words_tokenized = [word for word in library.word_tokenize(sent_tokenized[0])]

# Remove tokens that do not contain any letters from words_tokenized
filtered = [word for word in words_tokenized if library.search('[a-zA-Z]', word)]

# Display filtered words to observe words after tokenization
#print(filtered)

# Create an English language SnowballStemmer object
stemmer = library.SnowballStemmer("english")

# Print filtered to observe words without stemming
#print("Without stemming: ", filtered)

# Stem the words from filtered and store in stemmed_words
stemmed_words = [stemmer.stem(word) for word in filtered]

# Print the stemmed_words to observe words after stemming
#print("After stemming:   ", stemmed_words)

# Define a function to perform both stemming and tokenization
def tokenize_stemmer(text):
    tokens = [result for t in library.sent_tokenize(text)
                          for result in library.word_tokenize(t)]

    # Filter out raw tokens to remove noise
    filtered_tokens = [token for token in tokens if any(c.isalpha() for c in token)]

    # Stem the filtered_tokens
    stems = [library.SnowballStemmer("english").stem(t) for t in filtered_tokens]
    return stems


words_stemmed = tokenize_stemmer("Today (May 19, 2016) is his only daughter's wedding.")
#print(words_stemmed)

# Reshape words_stemmed to match the shape of stemmed_words
words_stemmed = np.reshape(words_stemmed, (len(stemmed_words),))

# TfidfVectorizer to create TF-IDF vectors
tfid_vecterizer = library.TfidfVectorizer(max_df=0.8,max_features=200000,
                                          min_df=0.2,stop_words="english",
                                          use_idf=True,tokenizer=tokenize_stemmer,
                                          ngram_range=(1,3))

tfidf_matrix = tfid_vecterizer.fit_transform([x for x in movie["plot"]])

#print(tfidf_matrix.shape)

km = library.KMeans(n_clusters=5)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

movie["cluster"] = clusters

movie["cluster"].value_counts()

similarty_distance = 1 - library.cosine_similarity(tfidf_matrix)

mergins = library.linkage(similarty_distance,method="complete")

dendrogram = library.dendrogram(mergins,
                                labels=[x for x in movie["title"]],
                                leaf_font_size=90,
                                leaf_rotation=16)

fig = library.plt.gcf()
_ = [lbl.set_color('r') for lbl in library.plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)

# Show the plotted dendrogram

library.plt.show()

# Which movie is most similar to the movie Braveheart ?
ans = "Gladiator"
print(ans)
