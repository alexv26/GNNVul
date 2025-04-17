from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from gensim.models import Word2Vec
import pandas as pd
import json
nltk.download('punkt_tab') # as per https://www.nltk.org/install.html
from multiprocessing import cpu_count
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

###* GETTING DATA FROM TRAIN DATASET *###

# assuming you run out of the data directory
def train_w2v(path="./final_data/all_train_data_new.json", save_path=BASE_DIR):
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Load the existing array
    except FileNotFoundError:
        raise Exception("Unable to access dataset")

    train = pd.DataFrame(data)

    ###* SEE TRAIN DATA *###
    #print(train)
    print(train.loc[0])
    #print(train.loc[0]["func"])


    ###* Tokenizing *###
    # Credit for the following code: https://www.analyticsvidhya.com/blog/2021/07/word2vec-for-word-embeddings-a-beginners-guide/

    def tokenize_function_text(func_text):
        return [[word.lower() for word in word_tokenize(sentence)] for sentence in sent_tokenize(func_text)]

    # Apply function in parallel using swifter
    # Flatten sentences into a single list per function
    train["tokenized_func"] = train["func"].swifter.apply(
        lambda func_text: [word.lower() for sentence in sent_tokenize(func_text) for word in word_tokenize(sentence)]
    )

    # Convert to a list of tokenized functions
    tokenized_functions_in_train_dataset = train["tokenized_func"].tolist()

    # Testing that tokenization was correct
    print(f"(number of tokenized functions, entries in db) = ({len(tokenized_functions_in_train_dataset), len(train)})")
    print(f"First tokenized function: \n {tokenized_functions_in_train_dataset[0]}")


    word2vec_model = Word2Vec(
        sentences=tokenized_functions_in_train_dataset,  
        vector_size=100,   # Size of word embeddings
        window=5,          # Context window size
        min_count=1,       # Minimum occurrences of a word
        workers=4,         # Use multi-threading
        sg=1              # Use Skip-Gram (1) instead of CBOW (0)
    )

    # Save the trained model
    word2vec_model.save(os.path.join(save_path, "word2vec_code.model"))