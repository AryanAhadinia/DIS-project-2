import requests
import pandas as pd
from tqdm import tqdm
import re
from langdetect import detect_langs
from deep_translator import GoogleTranslator



############ FONCTIONS TO GET METADATA FROM DIFFERENT API ############

def get_metadata_openlib(isbn):
    url = f"https://openlibrary.org/isbn/{isbn}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "works" in data.keys():
            work = data["works"][0]["key"]
            url = f"https://openlibrary.org{work}.json"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if "description" in data.keys():
                    if type(data["description"])== dict:
                        description = data["description"]["value"]
                    else:
                        description = data["description"]
                else:
                    description = None
                if "title" in data.keys():
                    title = data["title"]
                else:    
                    title = None
                if "authors" in data.keys():
                    author_key = data["authors"][0]['author']['key']
                    url = f"https://openlibrary.org{author_key}.json"
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        author = data["name"]
                    else:
                        author = None
                else:
                    author = None
            else : 
                author,title,description =None,None,None
        else:
            author,title,description =None,None,None
    else:
        description = None
        title = None
        author = None
    return  title, author, description

def get_metadata_gbook(isbn,key=False):
    api_key = 'AIzaSyCPEokGU1fZxT9VT5LOjV8bfZ9_VAAn5mY'
    if key:
        api_key = 'AIzaSyCPEokGU1fZxT9VT5LOjV8bfZ9_VAAn5mY'
        url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}&key={api_key}"
    else:
        url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "items" in data:
            book = data["items"][0]["volumeInfo"]
            title = book.get('title', 'Unavailable title')
            author = book.get('authors', ['Unavailable author'])[0]
            description = book.get('description', 'Unavailable description')
        else:
            title = None
            author = None
            description = None
    else:
        description = None
        title = None
        author = None
    return  title, author, description

def get_book_metadata_isbndb(isbn):
    api_key = "57496_b393f08cd79d5595ba1bc5c4b8f028ef"    
    base_url = f"https://api.isbndb.com/book/{isbn}"
    headers = {
        "Authorization": api_key
    }
    response = requests.get(base_url, headers=headers)    
    if response.status_code == 200:
        data = response.json()
        print(data)
        title = data.get('book', {}).get('title', "Title not available.")
        author = data.get('book', {}).get('authors', ["Author not available."])
        if len(author) > 0:
            author = author[0]
        else:
            author = "Author not available."
        description = data.get('book', {}).get('synopsis', "Description not available.")
    elif response.status_code == 404:
        title = "Book not found."
        author = "Book not found."
        description = "Book not found."
    
    else:
        description = None
        title = None
        author = None
    return  title, author, description


############ FONCTIONS TO CLEAN DATA ############

def process_ISBN(isbn):
    if pd.isna(isbn):
        return None
    if len(isbn) == 10:
        return isbn
    if len(isbn)<10:
        return '0'*(10-len(isbn)) + isbn
    return isbn

def get_book_metadata(isbn, books,get_fct = get_metadata_gbook):
    if not pd.isnull(isbn):
        if books.loc[books['ISBN'] == isbn,"title"].isnull().any() or books.loc[books['ISBN'] == isbn,"author"].isnull().any() or books.loc[books['ISBN'] == isbn,"description"].isnull().any():
            return get_fct(isbn)
        else:
            return books.loc[books['ISBN'] == isbn,["title", "author", "description"]].values[0]
    else :
        return None, None, None
    

def drop_books(books):
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    ids_train_test = set(train['book_id']).union(set(test['book_id']))
    ids_books = set(books['book_id'])
    return books[~books['book_id'].isin(ids_books - ids_train_test)]


def clean_text(text):
    """
    Nettoie un texte en supprimant les sauts de ligne, les liens et les caractères spéciaux.

    Args:
        text (str): Texte brut à nettoyer.

    Returns:
        str: Texte nettoyé.
    """
    if pd.isnull(text):
        return None
    
    # Remove newlines and carriage returns
    text = text.replace('\n', ' ').replace('\r', '')

    # Remove web links (http, https, ftp, etc.)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove non-alphanumeric characters except common punctuation
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove leading and trailing whitespaces
    text = text.strip()

    return text


def detect_lang(text):

    if type(text)== str and len(text) > 3:
        return detect_langs(text)[0].lang
    else: 
        return 'unknown'
    
def translate(books):
    translator = GoogleTranslator(source='auto', target='en')  

    for lang in tqdm(books['lang'].unique()):
        if lang =='unknown' or lang == 'en':
            continue

        books.loc[books['language']==lang, 'description'] = books[books['language']==lang]['description'].apply(
            lambda x: translator.translate(x)
        )
        books.loc[books['language']==lang, 'title'] = books[books['language']==lang]['title'].apply(
            lambda x: translator.translate(x)
        )

if __name__ == "__main__":

    books = pd.read_csv("data/books.csv")
    books = books[:10].copy()
    books['ISBN'] = books['ISBN'].apply(process_ISBN)
    books['title'] = None
    books['author'] = None
    books['description'] = None

    print("Drop books not in train or test dataset")
    books = drop_books(books)

    print("Get book metada from OpenLibrary")
    tqdm.pandas()
    books[['title', 'author', 'description']] = books['ISBN'].progress_apply(
         lambda isbn: pd.Series(get_book_metadata(isbn,books,get_metadata_openlib)))
    print("Get book metada from Google Books")
    tqdm.pandas()
    books[['title', 'author', 'description']] = books['ISBN'].progress_apply(
         lambda isbn: pd.Series(get_book_metadata(isbn,books,get_metadata_gbook)))
    print("Get book metada from ISBNDB")
    tqdm.pandas()
    books[['title', 'author', 'description']] = books['ISBN'].progress_apply(
         lambda isbn: pd.Series(get_book_metadata(isbn,books,get_book_metadata_isbndb)))
    
    print("Clean descriptions")
    books['description'] = books['description'].apply(clean_text)

    print("Detect language of descriptions")
    books['lang'] = books['description'].apply(detect_lang)

    print("Translate tilte and description to english")
    translate(books)

    books.to_csv("data/clean_data/metadata_clean_test.csv", index=False)
    print("Metadata saved in data/clean_data/metadata_clean.csv")

    

    
    
