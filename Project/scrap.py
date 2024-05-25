import streamlit as st
import pandas as pd
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from selenium import webdriver
from bs4 import BeautifulSoup as soup
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import spacy
import os

# Load the trained models
loaded_model_scrapping = joblib.load('kha90.joblib')
loaded_model_detecting = joblib.load('lejone99.joblib')
nlp = spacy.load('en_core_web_sm')

# Function to preprocess text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    if pd.isna(text):
        return ''
    else:
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        cleaned_text = ' '.join(tokens)
        return cleaned_text

# Function to scrape Facebook comments with login
def scrape_comments_with_login(url, email, password, scroll_count=3, scroll_delay=2):
    executable_path = 'C:/Users/lejone/chromedriver.exe'
    service = Service(executable_path)
    
    chrome_options = webdriver.ChromeOptions()
    # Run Chrome in headless mode
    chrome_options.add_argument('--headless') 
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Navigate to the Facebook login page
    driver.get('https://www.facebook.com/')
    
    # Locate username and password input fields
    username = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='email']")))
    password = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='pass']")))

    # Enter username and password
    username.clear()
    username.send_keys(email)  # Read from input
    password.clear()
    password.send_keys(password)  # Read from input
    time.sleep(1)

    # Locate and click the login button
    button = WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))).click()
    print("Logged In")
    
    # Wait for the login to complete
    WebDriverWait(driver, 10).until(EC.url_changes('https://www.facebook.com/'))
    
    # Now navigate to the desired URL
    driver.get(url)

    for _ in range(scroll_count):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_delay)

    html = driver.page_source
    page_soup = soup(html, 'html.parser')
    comment_box = page_soup.find_all('div', class_="x1lliihq xjkvuk6 x1iorvi4")
    comment_list = [comment.text.strip() for comment in comment_box]
    driver.quit()
    return comment_list

# Streamlit app with Bootstrap styling and icons
st.set_page_config(page_title="Comment Scraper & Bullying Detector", page_icon="🔍", layout="wide")
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://kit.fontawesome.com/a076d05399.js" crossorigin="anonymous"></script>
    <style>
        .icon { font-size: 24px; }
        .navbar { margin-bottom: 20px; }
        .container { max-width: 80%; }
    </style>
""", unsafe_allow_html=True)

def main():
    st.sidebar.markdown("""
        <nav class="navbar navbar-expand-lg navbar-light bg-light">
            <a class="navbar-brand" href="#"><i class="fas fa-comments icon"></i> App</a>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="#">Scrapping</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#">Detecting</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#">Close</a>
                    </li>
                </ul>
            </div>
        </nav>
    """, unsafe_allow_html=True)

    app_mode = st.sidebar.selectbox("Choose an option", ["Scrapping", "Detecting", "Close"])

    if app_mode == "Scrapping":
        st.title('Facebook Comment Scraper')
        email = st.text_input('Enter your Facebook email:')
        password = st.text_input('Enter your Facebook password:', type='password')
        url = st.text_input('Enter the Facebook page URL:', 'https://web.facebook.com/VodacomLesotho')

        if st.button('Scrape and Classify Comments'):
            if email and password and url:
                with st.spinner('Scraping comments...'):
                    comments = scrape_comments_with_login(url, email, password)
                    st.success('Comments scraped successfully!')

                if comments:
                    df = pd.DataFrame(comments, columns=['Comment'])
                    df['Cleaned_Comment'] = df['Comment'].apply(preprocess_text)
                    df['bullying_type'] = df['Cleaned_Comment'].apply(lambda x: loaded_model_scrapping.predict([x])[0])

                    st.write('Scraped and classified comments:')
                    st.dataframe(df)

                    df.to_csv('output_with_predictions.csv', index=False)
                    st.success('Data saved to output_with_predictions.csv')
                else:
                    st.warning('No comments found.')
            else:
                st.warning('Please enter your email, password, and Facebook page URL.')

    elif app_mode == "Detecting":
        st.title('Bullying Detection App')
        text = st.text_input('Enter text to check for bullying:')
        if st.button('Check'):
            processed_text = preprocess_text(text)
            prediction = loaded_model_detecting.predict([processed_text])
            st.write(f'The text is classified as: {prediction[0]}')

    else:
        st.title('Close the App')
        st.write('Thank you for using the app!')
        st.stop()

if __name__ == '__main__':
    main()
