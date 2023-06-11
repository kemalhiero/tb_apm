from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle

# Load model from file
filename = "pickle/model_svm.sav"
model = pickle.load(open(filename, "rb"))

# Load vectorizer from file
vectorizer = pickle.load(open('pickle/vectorizer.sav', 'rb'))

# Transformer label
genre = pickle.load(open('pickle/genre.sav', 'rb'))

# preprocessing
def preprocess(data):
    # Lowercase all text
    data = data.lower()

    # Remove punctuations and special characters
    data = re.sub('[^a-zA-Z0-9\s]', '', data)

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    data = ' '.join([stemmer.stem(word) for word in data.split() if word not in stop_words])

    return data


# Create a function to predict the class of new data points
def predict_class(new_data):
    new_data = vectorizer.transform(new_data)
    pred = model.predict(new_data)
    return pred

def cari_genre(target_value):

    # Iterasi melalui setiap baris dalam kolom yang diberikan
    for value in genre['no']:
        # Cek apakah nilai pada baris saat ini sama dengan nilai target
        if value == target_value:
            result = genre.loc[genre['no'] == value]
            return result['genre'].values[0]



# Create a Streamlit app
def main():
    st.title('Movie genre classifier')

    # st.write(genre)
    
    # Create input field for plot
    plot = st.text_area('Enter the plot of the film:' )

    # Predict the genre of the film
    if st.button('Submit'):
        user_input = preprocess(plot)
        pred = predict_class([user_input])

        tampil = cari_genre(pred[0])
        st.write(f'The predicted genre is {tampil}')


if __name__ == '__main__':
    main()
