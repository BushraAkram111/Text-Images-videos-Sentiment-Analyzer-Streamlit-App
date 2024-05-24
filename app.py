import streamlit as st
import streamlit.components.v1 as components
from textblob import TextBlob
from PIL import Image
import text2emotion as te
import plotly.graph_objects as go
import requests
import json
import pandas as pd
import numpy as np
import cv2
from io import StringIO

# Emoji dictionary
getEmoji = {
    "happy": "😊",
    "neutral": "😐",
    "sad": "😔",
    "disgust": "🤢",
    "surprise": "😲",
    "fear": "😨",
    "angry": "😡",
    "positive": "🙂",
    "negative": "☹️",
}

# Sidebar function
def show_sidebar():
    st.sidebar.title("Navigation")
    return st.sidebar.selectbox("Choose a page", ["Text Analysis", "Image Analysis", "IMDB Reviews"])

# Functions from imagePage.py
def showEmotionData(emotion, topEmotion, image, idx):
    x, y, w, h = tuple(emotion["box"])
    cropImage = image[y:y+h, x:x+w]

    cols = st.columns(7)
    keys = list(emotion["emotions"].keys())
    values = list(emotion["emotions"].values())
    emotions = sorted(emotion["emotions"].items(), key=lambda kv: (kv[1], kv[0]))

    st.components.v1.html(f"<h3 style='color: #ef4444; font-family: Source Sans Pro, sans-serif; font-size: 20px; margin-bottom: 0px; margin-top: 0px;'>Person detected {idx}</h3>", height=30)
    col1, col2, col3 = st.columns([3, 1, 2])

    with col1:
        st.image(cropImage, width=200)
    with col2:
        for i in range(4):
            st.metric(keys[i].capitalize() + " " + getEmoji[keys[i]], round(values[i], 2), None)
    with col3:
        for i in range(4, 7):
            st.metric(keys[i].capitalize() + " " + getEmoji[keys[i]], round(values[i], 2), None)
        st.metric("Top Emotion", emotions[-1][0].capitalize() + " " + getEmoji[topEmotion[0]], None)

    st.components.v1.html("<hr>", height=5)

def printResultHead():
    st.write("")
    st.write("")
    st.components.v1.html("<h3 style='color: #0ea5e9; font-family: Source Sans Pro, sans-serif; font-size: 26px; margin-bottom: 10px; margin-top: 60px;'>Result</h3><p style='color: #57534e; font-family: Source Sans Pro, sans-serif; font-size: 16px;'>Find below the sentiments we found in your given image. What do you think about our results?</p>", height=150)

def printImageInfoHead():
    st.write("")
    st.write("")
    st.components.v1.html("<h3 style='color: #ef4444; font-family: Source Sans Pro, sans-serif; font-size: 22px; margin-bottom: 0px; margin-top: 40px;'>Image information</h3><p style='color: #57534e; font-family: Source Sans Pro, sans-serif; font-size: 14px;'>Expand below to see the information associated with the uploaded image</p>", height=100)

def load_image(image_file):
    image = Image.open(image_file, 'r')
    return image

def uploadFile():
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        content = Image.open(uploaded_file)
        content = np.array(content)  # pil to cv
        shape = np.shape(content)
        if len(shape) < 3:
            st.error('Your image has a bit-depth less than 24. Please upload an image with a bit-depth of 24.')
            return

        emotions, topEmotion, image = modals.imageEmotion(content)

        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
        printImageInfoHead()
        with st.expander("See JSON Object"):
            st.json(json.dumps(file_details))
            st.image(load_image(uploaded_file), caption=uploaded_file.name, width=250)

        if emotions and not emotions:
            st.text("No faces found!!")
        if emotions:
            printResultHead()
            with st.expander("Expand to see individual result"):
                contentcopy = Image.open(uploaded_file)
                contentcopy = np.array(contentcopy)
                for i in range(len(emotions)):
                    showEmotionData(emotions[i], topEmotion, contentcopy, i+1)

            st.write("")
            st.write("")
            col1, col2 = st.columns([4, 2])

            with col1:
                st.image(image, width=300)
            with col2:
                st.metric("Top Emotion", topEmotion[0].capitalize() + " " + getEmoji[topEmotion[0]], None)
                st.metric("Emotion Percentage", str(round(topEmotion[1]*100, 2)), None)

def show_image_page():
    st.title("Sentiment Analyzer 😊😐😕😡")
    components.html("<hr style='height:3px;border:none;color:#333;background-color:#333; margin-bottom: 10px' /> ")
    st.subheader("Image Analyzer")
    st.text("Enter your image and let's find sentiments in there.")
    st.text("")
    uploadFile()

# Functions from imdbReviewsPage.py
def plotPie(labels, values):
    fig = go.Figure(go.Pie(labels=labels, values=[value*100 for value in values], hoverinfo="label+percent", textinfo="value"))
    st.plotly_chart(fig, use_container_width=True)

def getMovies(movieName):
    response = requests.get(f'{baseURL}/SearchMovie/{apiKey}/{movieName}')
    response = response.json()
    if isinstance(response["results"], list):
        movies = [{"id": result['id'], "title": result['title'], "image": result["image"], "description": result["description"]} for result in response["results"]]
        return movies
    else:
        st.error(response["errorMessage"])
        return []

def getFirst200Words(string):
    if len(string) > 200:
        return string[:200]
    return string

def getReviews(id):
    res = requests.get(f'{baseURL}/Reviews/{apiKey}/{id}')
    res = res.json()
    if res["errorMessage"]:
        st.error(res["errorMessage"])
        return []
    items = res["items"]
    if len(items) > 20:
        items = items[0:20]
    reviews = [getFirst200Words(item["title"] + " " + item["content"]) for item in items]
    return reviews

def getData(movieName):
    movies = getMovies(movieName)
    data = []
    for movie in movies:
        reviews = getReviews(movie["id"])
        data.append({"title": movie["title"], "image": movie["image"], "description": movie["description"], "reviews": reviews})
    return json.dumps({"userSearch": movieName, "result": data})

def displayMovieContent(movie):
    col1, col2 = st.columns([2, 3])
    with col1:
        st.image(movie["image"], width=200)
    with col2:
        st.components.v1.html(f"<h3 style='color: #1e293b; font-family: Source Sans Pro, sans-serif; font-size: 20px; margin-bottom: 10px; margin-top: 60px;'>{movie['title']}</h3><p style='color: #64748b; font-family: Source Sans Pro, sans-serif; font-size: 14px;'>{movie['description']}</p>", height=150)

def getEmojiString(head):
    emojiHead = ""
    emotions = head.split("-")
    for emotion in emotions:
        emo = emotion.strip()
        emojiHead += getEmoji[emo.lower()]
    return head + " " + emojiHead

def applyModal(movie, packageName):
    if(packageName == "Flair"):
        predictionList = [modals.flair(review) for review in movie["reviews"]]
        valueCounts = dict(pd.Series(predictionList).value_counts())
        print(valueCounts)
        return valueCounts
    elif(packageName == "TextBlob"):
        predictionList = [modals.textBlob(review) for review in movie["reviews"]]
        valueCounts = dict(pd.Series(predictionList).value_counts())
        print(valueCounts)
        return valueCounts
    elif(packageName == "Vader"):
        predictionList = [modals.vader(review) for review in movie["reviews"]]
        valueCounts = dict(pd.Series(predictionList).value_counts())
        print(valueCounts)
        return valueCounts
    elif(packageName == "Text2emotion"):
        predictionList = [modals.text2emotion(review) for review in movie["reviews"]]
        valueCounts = dict(pd.Series(predictionList).value_counts())
        print(valueCounts)
        return valueCounts
    else:
        return ""
    

def renderPage():
    st.title("Sentiment Analyzer")
    components.html("""<hr style="height:3px;border:none;color:#333;background-color:#333; margin-bottom: 10px" /> """)
    # st.markdown("### User Input Text Analysis")
    st.subheader("IMDb movie review analyer")
    st.text("Analyze movie reviews from IMDb API for sentiments.")
    st.text("")
    movieName = st.text_input('Movie Name', placeholder='Enter Name here')
    packageName = st.selectbox(
     'Select Package',
     ('Flair', 'Vader', 'TextBlob', 'Text2emotion'))
    if st.button('Search'):
        if movieName:
            process(movieName, packageName)
        else:
            st.warning("Please enter a movie name")
from flair.models import TextClassifier
from flair.data import Sentence
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import text2emotion as te
from fer import FER
import matplotlib.pyplot as plt
import cv2
import numpy as np

"""
    Argument:
        Single Text(String) 

    Returns:
        Returns emotion(String)
"""

sia = TextClassifier.load('en-sentiment')
emo_detector = FER(mtcnn=True)

# For Text data
def flair(text):
    sentence = Sentence(text)
    sia.predict(sentence)
    score = str(sentence.labels[0])
    startIdx = int(score.rfind("("))
    endIdx = int(score.rfind(")"))
    percentage = float(score[startIdx+1:endIdx])
    if percentage < 0.60:
        return "NEUTRAL"
    elif "POSITIVE" in str(score):
        return "POSITIVE"
    elif "NEGATIVE" in str(score):
        return "NEGATIVE"
    
    
# For Text data
def textBlob(text):
    tb = TextBlob(text)
    polarity = round(tb.polarity, 2)
    if polarity>0:
        return "Positive"
    elif polarity==0:
        return "Neutral"
    else:
        return "Negative"
    
    
# For Text data
def vader(text):
    #analyze the sentiment for the text
    scores = SentimentIntensityAnalyzer().polarity_scores(text)
    if scores['compound'] >= 0.05 :
        return "Positive"
 
    elif scores['compound'] <= - 0.05 :
        return "Negative"
 
    else :
        return "Neutral"
        

# For Text data
def text2emotion(text):
    emotion = dict(te.get_emotion(text))
    emotion = sorted(emotion.items(), key =
             lambda kv:(kv[1], kv[0]), reverse=True)
    emotionStr = list(emotion)[0][0]
    if(list(emotion)[1][1]>=0.5 or list(emotion)[1][1] == list(emotion)[0][1]):
        emotionStr+=" - {}".format(list(emotion)[1][0])
    print(emotion, emotionStr)
    return emotionStr
    
    
def imageEmotion(image):
    captured_emotions = emo_detector.detect_emotions(image)
    topEmotion = emo_detector.top_emotion(image)
    print(captured_emotions, topEmotion)
    img = image
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
  
    # fontScale
    fontScale = 1.2
   
    # Blue color in BGR
    color = (255, 0, 0)
  
    # Line thickness of 2 px
    thickness = 2
    for emotion in captured_emotions:
        x, y, w, h = tuple(emotion["box"])
        org = (x+w+4, y+5)
        emotions = emotion["emotions"]
        emotions = sorted(emotions.items(), key =
             lambda kv:(kv[1], kv[0]))
        cv2.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 2)
        cv2.putText(img, emotions[len(emotions)-1][0], org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    return captured_emotions, topEmotion, img
    ################################################################
from pickle import FALSE
import streamlit as st
from streamlit_option_menu import option_menu

def show():
    with st.sidebar:
        st.markdown("""
                    # Applications
                    """, unsafe_allow_html = False)
        selected = option_menu(
            menu_title = None, #required
            # options = ["Text", "IMDb movie reviews", "Image", "Audio", "Video", "Twitter Data", "Web Scraping"], #required
            # icons = ["card-text", "film", "image", "mic", "camera-video", "twitter", "globe"], #optional
            
            options = ["Text", "IMDb movie reviews", "Image"], #required
            icons = ["card-text", "film", "image"], #optional
            
            # menu_icon="cast", #optional
            default_index = 0, #optional
        )
        return selected
    ####################################
import streamlit as st
import streamlit.components.v1 as components
from textblob import TextBlob
from PIL import Image
import text2emotion as te
import plotly.graph_objects as go

def plotPie(labels, values):
    fig = go.Figure(
        go.Pie(
        labels = labels,
        values = values,
        hoverinfo = "label+percent",
        textinfo = "value"
    ))
    st.plotly_chart(fig)

    
def getPolarity(userText):
    tb = TextBlob(userText)
    polarity = round(tb.polarity, 2)
    subjectivity = round(tb.subjectivity, 2)
    if polarity>0:
        return polarity, subjectivity, "Positive"
    elif polarity==0:
        return polarity, subjectivity, "Neutral"
    else:
        return polarity, subjectivity, "Negative"

def getSentiments(userText, type):
    if(type == 'Positive/Negative/Neutral - TextBlob'):
        polarity, subjectivity, status = getPolarity(userText)
        if(status=="Positive"):
            image = Image.open('./images/positive.PNG')
        elif(status == "Negative"):
            image = Image.open('./images/negative.PNG')
        else:
            image = Image.open('./images/neutral.PNG')
        col1, col2, col3 = st.columns(3)
        col1.metric("Polarity", polarity, None)
        col2.metric("Subjectivity", subjectivity, None)
        col3.metric("Result", status, None)
        st.image(image, caption=status)
    elif(type == 'Happy/Sad/Angry/Fear/Surprise - text2emotion'):
        emotion = dict(te.get_emotion(userText))
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Happy 😊", emotion['Happy'], None)
        col2.metric("Sad 😔", emotion['Sad'], None)
        col3.metric("Angry 😠", emotion['Angry'], None)
        col4.metric("Fear 😨", emotion['Fear'], None)
        col5.metric("Surprise 😲", emotion['Surprise'], None)
        plotPie(list(emotion.keys()), list(emotion.values()))
        

def renderPage():
    st.title("Sentiment Analyzer")
    components.html("""<hr style="height:3px;border:none;color:#333;background-color:#333; margin-bottom: 10px" /> """)
    # st.markdown("### User Input Text Analysis")
    st.subheader("Text Analysis")
    st.text("The objective is to analyze textual user input and identify the sentiment expressed within.")
    st.text("")
    userText = st.text_input('User Input', placeholder='Input text HERE')
    st.text("")
    type = st.selectbox(
     'Type of analysis',
     ('Positive/Negative/Neutral - TextBlob', 'Happy/Sad/Angry/Fear/Surprise - text2emotion'))
    st.text("")
    if st.button('Predict'):
        if(userText!="" and type!=None):
            st.text("")
            st.components.v1.html("""
                                <h3 style="color: #0284c7; font-family: Source Sans Pro, sans-serif; font-size: 28px; margin-bottom: 10px; margin-top: 50px;">Result</h3>
                                """, height=100)
            getSentiments(userText, type)

