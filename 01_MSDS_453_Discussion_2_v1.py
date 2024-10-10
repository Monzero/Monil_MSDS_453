#Auther         : Monil Shah
#Last Updated   : Oct 10, 2024
#Purpose        : This code helps preparing for the analysis of NLP course and the discussions.
#Main Inputs    : Main Inputs are two review data sets split into 10 reviews each. And I also source the terms extracted from online tools. 
#Analysis       : The analysis is around NLP exercises.


##################################################################################################
##########             Load libraries                       ######################################
##################################################################################################


import pandas as pd
pd.set_option("display.max_rows",50)
import os
from itertools import combinations

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
import unicodedata
import spacy
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')


##################################################################################################
##########             Define Functions                     ######################################
##################################################################################################

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

strip_html_tags('<html><h2>Some important text</h2></html>')

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

remove_accented_chars('Sómě Áccěntěd těxt')

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


remove_special_characters("Well this was fun! What do you think? 123#@!", remove_digits=True)

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

simple_stemmer("My system keeps crashing his crashed yesterday, ours crashes daily")

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

remove_stopwords("The, and, if are stopwords, computer is not")

# Main function to preprocess text (Lemmatization + Stemming)
def preprocess_text(text):
    lowercased_text = text.lower()
    no_stopwords_text = remove_stopwords(lowercased_text, is_lower_case=True)

    lemmatized_text = lemmatize_text(no_stopwords_text)
    #stemmed_text = simple_stemmer(lemmatized_text)

    # If your text had HTML or special characters, apply those functions first
    cleaned_text = strip_html_tags(lemmatized_text)  # Not necessary here since there's no HTML
    cleaned_text = remove_special_characters(cleaned_text, remove_digits=True)
    return(cleaned_text)

def get_tfidf_dataframe(corpus):
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()
    # Fit and transform the corpus
    X = vectorizer.fit_transform(corpus)
    # Get feature names (terms)
    feature_names = vectorizer.get_feature_names_out()
    # Convert the sparse matrix to a dense array (2D array)
    dense_array = X.toarray()
    # Create a DataFrame from the dense array, with feature names as columns
    df = pd.DataFrame(dense_array, columns=feature_names)
    # Round the values to 4 decimal places
    df = df.round(4) 
    return df

import pandas as pd

# Function to return top N TF-IDF words and their scores
def get_top_n_words_from_dataframe(df, n=3):
    
    # Data structure to hold results
    top_n_words_per_doc = []
    
    # Iterate over each row (document) in the DataFrame
    for i, row in df.iterrows():
        # Sort the row values (TF-IDF scores) in descending order and select the top n
        top_n = row.sort_values(ascending=False).head(n)
        
        # Store document index, words, and their TF-IDF scores
        for word, score in top_n.items():
            top_n_words_per_doc.append({
                'Document': f'Document {i+1}',  # Labeling document (1-indexed)
                'Word': word,
                'TF-IDF Score': score
            })
    
    # Convert to DataFrame for a structured output
    result_df = pd.DataFrame(top_n_words_per_doc)
    
    return result_df


##################################################################################################
##########        01. source Data and start script          ######################################
##################################################################################################

hotel_review_dir        =  '/Users/monilshah/Documents/02_NWU/10_MSDS_453_NLP/01_Assignment_01/02_Submission/01_Review Doc Monil Shah/Hotel Review/'
movie_review_dir        =  '/Users/monilshah/Documents/02_NWU/10_MSDS_453_NLP/01_Assignment_01/02_Submission/01_Review Doc Monil Shah/Movie Review/'

hotel_review_files      = os.listdir(hotel_review_dir)
movie_review_files      = os.listdir(movie_review_dir)

# Filter only .txt files
hotel_review_files = [file for file in hotel_review_files if file.endswith('.txt')]
movie_review_files = [file for file in movie_review_files if file.endswith('.txt')]


##################################################################################################
##########         02. Making datafrane of hotel reviews    ######################################
##################################################################################################

df_hotel_reviews        = pd.DataFrame()

with open(hotel_review_dir + h, 'r') as file_wrapper:  # Use 'with' for automatic file closing
        data = file_wrapper.read()


for h in hotel_review_files:
    with open(hotel_review_dir + h, 'r') as file_wrapper:  # Use 'with' for automatic file closing
        data = file_wrapper.read()

    # Initialize an empty dictionary to store key-value pairs
    parsed_data = {}

    # Split the text by newline character
    lines = data.strip().split("\n")

    # Loop through each line and split by the first occurrence of colon
    for line in lines:
        if ":" in line:  # Ensure there is a colon in the line to avoid splitting errors
            key, value = line.split(":", 1)  # Split only at the first colon
            parsed_data[key.strip()] = value.strip()

    # Create a DataFrame from the parsed data
    df = pd.DataFrame([parsed_data])

    # Concatenate the new DataFrame with the main DataFrame
    df_hotel_reviews = pd.concat([df_hotel_reviews, df], ignore_index=True)

# Display the final DataFrame
print(df_hotel_reviews)


##################################################################################################
##########         03. Making datafrane of movie reviews    ######################################
##################################################################################################



df_movie_reviews        = pd.DataFrame()


for m in movie_review_files:
    with open(movie_review_dir + m, 'r') as file_wrapper:  # Use 'with' for automatic file closing
        data = file_wrapper.read()

    # Initialize an empty dictionary to store key-value pairs
    parsed_data = {}

    # Split the text by newline character
    lines = data.strip().split("\n")

    # Loop through each line and split by the first occurrence of colon
    for line in lines:
        if ":" in line:  # Ensure there is a colon in the line to avoid splitting errors
            key, value = line.split(":", 1)  # Split only at the first colon
            parsed_data[key.strip()] = value.strip()

    # Create a DataFrame from the parsed data
    df = pd.DataFrame([parsed_data])

    # Concatenate the new DataFrame with the main DataFrame
    df_movie_reviews = pd.concat([df_movie_reviews, df], ignore_index=True)

# Display the final DataFrame
print(df_movie_reviews)

##################################################################################################
##########        04.  sourcing manual terms and getting frequency    ############################
##################################################################################################

me_terms      = pd.read_csv('/Users/monilshah/Documents/02_NWU/10_MSDS_453_NLP/02_Discussion_2/Extraction_of_terms.csv')

#####################     For Hotels         ####################################################

#hotel_terms  = me_terms[me_terms['Type']=='Hotel']['Keyword']
hotel_terms   = me_terms[me_terms['Type']=='Hotel']
hotel_terms   = pd.DataFrame(hotel_terms)

# Initialize new columns for each review in hotel_terms
for i in range(1, 11):
    hotel_terms[f'count_in_doc{i}'] = 0  # Create columns count_in_doc1, count_in_doc2, ..., count_in_doc10

# Loop through each keyword and count occurrences in each review
for index, keyword in enumerate(hotel_terms['Keyword']):  # Use enumerate to get index and keyword
    for i in range(1, 11):
        review = df_hotel_reviews['Brief summary'][i - 1]  # Access the review (subtract 1 to match zero-indexing)
        keyword_count = review.lower().count(keyword.lower())  # Case-insensitive count
        hotel_terms.at[index, f'count_in_doc{i}'] = keyword_count  # Store count in the appropriate column

# Display the updated hotel_terms DataFrame
print(hotel_terms)
print(f'Total count found for 450 terms is {hotel_terms.loc[:, 'count_in_doc1':'count_in_doc10'].sum(axis=1).sum()}')

print(f'Total count found for 450 terms are {hotel_terms.loc[:, 'count_in_doc1':'count_in_doc10'].sum(axis=1).sum()}')
hotel_terms.to_csv('/Users/monilshah/Documents/02_NWU/10_MSDS_453_NLP/01_Assignment_01/02_Submission/01_Review Doc Monil Shah/Output/hoetl_terms_frequency.csv')
#####################     For movies         ####################################################

#movie_terms  = me_terms[me_terms['Type']=='Movies']['Keyword']
movie_terms   = me_terms[me_terms['Type']=='Movie']
movie_terms   = pd.DataFrame(movie_terms).reset_index(drop=True)

# Initialize new columns for each review in movie_terms
for i in range(1, 11):
    movie_terms[f'count_in_doc{i}'] = 0  # Create columns count_in_doc1, count_in_doc2, ..., count_in_doc10

# Loop through each keyword and count occurrences in each review
for index, keyword in enumerate(movie_terms['Keyword']):  # Use enumerate to get index and keyword
    for i in range(1, 11):
        review = df_movie_reviews['Brief summary'][i - 1]  # Access the review (subtract 1 to match zero-indexing)
        keyword_count = review.lower().count(keyword.lower())  # Case-insensitive count
        movie_terms.at[index, f'count_in_doc{i}'] = keyword_count  # Store count in the appropriate column

# Display the updated hotel_terms DataFrame
print(movie_terms)

print(f'Total count found for 20 terms are {movie_terms.loc[:, 'count_in_doc1':'count_in_doc10'].sum(axis=1).sum()}')

movie_terms.to_csv('/Users/monilshah/Documents/02_NWU/10_MSDS_453_NLP/01_Assignment_01/02_Submission/01_Review Doc Monil Shah/Output/movie_term_frequency.csv')

#df_hotel_reviews['Brief summary'].to_csv('/Users/monilshah/Documents/02_NWU/10_MSDS_453_NLP/01_Assignment_01/02_Submission/01_Review Doc Monil Shah/Hotel Review/all_hotel_reviews.csv')
#df_movie_reviews['Brief summary'].to_csv('/Users/monilshah/Documents/02_NWU/10_MSDS_453_NLP/01_Assignment_01/02_Submission/01_Review Doc Monil Shah/Movie Review/all_movie_reviews.csv')

##################################################################################################
##########       05.  Calculating overlap of words  between docs      ############################
##################################################################################################

###########           overlap table for Hotel                       #############################


# Step 1: Group by 'Method' and create sets of words from 'Keywords'
grouped_terms = hotel_terms.groupby('Method')['Keyword'].apply(lambda x: set(' '.join(x).split()))

# Step 2: Create a DataFrame to hold the metrics (pairwise comparison of common words)
methods = grouped_terms.index
metrics_df = pd.DataFrame(index=methods, columns=methods)

# Step 3: Iterate over each combination of methods and calculate the intersection of their keywords
for method1, method2 in combinations(grouped_terms.index, 2):
    common_words = grouped_terms[method1].intersection(grouped_terms[method2])
    common_count = len(common_words)
    
    # Fill the matrix symmetrically
    metrics_df.loc[method1, method2] = common_count
    metrics_df.loc[method2, method1] = common_count

# Step 4: For each method, calculate the number of all words (including unique ones)
for method in methods:
    total_words = len(grouped_terms[method])
    metrics_df.loc[method, method] = total_words

# Step 5: Fill any remaining NaN values with 0
metrics_df.fillna(0, inplace=True)

# Step 6: Display the metrics DataFrame
print(metrics_df)
metrics_df.to_csv('/Users/monilshah/Documents/02_NWU/10_MSDS_453_NLP/01_Assignment_01/02_Submission/01_Review Doc Monil Shah/Hotel Review/hotel_term_frequency.csv')

###########           overlap table for movies                    #############################

# Step 1: Group by 'Method' and create sets of words from 'Keywords'
grouped_terms = movie_terms.groupby('Method')['Keyword'].apply(lambda x: set(' '.join(x).split()))

# Step 2: Create a DataFrame to hold the metrics (pairwise comparison of common words)
methods = grouped_terms.index
metrics_df = pd.DataFrame(index=methods, columns=methods)

# Step 3: Iterate over each combination of methods and calculate the intersection of their keywords
for method1, method2 in combinations(grouped_terms.index, 2):
    common_words = grouped_terms[method1].intersection(grouped_terms[method2])
    common_count = len(common_words)
    
    # Fill the matrix symmetrically
    metrics_df.loc[method1, method2] = common_count
    metrics_df.loc[method2, method1] = common_count

# Step 4: For each method, calculate the number of all words (including unique ones)
for method in methods:
    total_words = len(grouped_terms[method])
    metrics_df.loc[method, method] = total_words

# Step 5: Fill any remaining NaN values with 0
metrics_df.fillna(0, inplace=True)

# Step 6: Display the metrics DataFrame
print(metrics_df)
metrics_df.to_csv('/Users/monilshah/Documents/02_NWU/10_MSDS_453_NLP/01_Assignment_01/02_Submission/01_Review Doc Monil Shah/Movie Review/movie_term_frequency.csv')
  

##################################################################################################
##########          06. text preprocessing  & TDIDF                   ############################
##################################################################################################

##################    Following code is picked from the lab3 file shared  ########################

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


###################################################################################################
##########################     06_01   Movie Data TF IDF score     ################################
###################################################################################################



mr_list                    =  df_movie_reviews['Brief summary'].to_list()
pp_mr_list                 = [preprocess_text(doc) for doc in mr_list]

movie_tdidf                = get_tfidf_dataframe(pp_mr_list)

# Get top 3 TF-IDF words for each document in a new DataFrame
movie_top_n_tfidf_df      = get_top_n_words_from_dataframe(movie_tdidf, n=10)
movie_top_n_tfidf_df['Type']= "Movie"

###################################################################################################
###########################    06_02   Hotel Data TF IDF score     ################################
###################################################################################################



hr_list                    =  df_hotel_reviews['Brief summary'].to_list()
pp_hr_list                 = [preprocess_text(doc) for doc in hr_list]

hotel_tdidf                = get_tfidf_dataframe(pp_hr_list)

# Get top 3 TF-IDF words for each document in a new DataFrame
hotel_top_n_tfidf_df      = get_top_n_words_from_dataframe(hotel_tdidf, n=10)

hotel_top_n_tfidf_df['Type']="Hotel"

tfidf_df                 = pd.concat([hotel_top_n_tfidf_df, movie_top_n_tfidf_df])

tfidf_df.head(20)


