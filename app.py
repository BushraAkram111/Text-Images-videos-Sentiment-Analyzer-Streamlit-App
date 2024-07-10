import streamlit as st
import streamlit.components.v1 as components
from textblob import TextBlob
from PIL import Image
import text2emotion as te
import plotly.graph_objects as go
import numpy as np
import cv2
from flair.models import TextClassifier
from flair.data import Sentence
from fer import FER
from moviepy.editor import VideoFileClip
import pandas as pd

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

# Function to load image
def load_image(image_file):
    image = Image.open(image_file)
    return image

# Function to print result header
def print_result_header():
    st.write("")
    st.write("")
    components.html(
        "<h3 style='color: #0ea5e9; font-family: Source Sans Pro, sans-serif; "
        "font-size: 26px; margin-bottom: 10px; margin-top: 60px;'>Result</h3>"
        "<p style='color: #57534e; font-family: Source Sans Pro, sans-serif; "
        "font-size: 16px;'>Find below the sentiments we found in your given image. "
        "What do you think about our results?</p>", height=150
    )

# Function to upload file
def upload_file(allowed_types):
    uploaded_file = st.file_uploader("Upload a file", type=allowed_types)
    return uploaded_file

# Main function to render image analysis page
def show_image_page():
    st.title("Sentiment Analyzer 😊😐😕😡")
    st.subheader("Image Analyzer 😊📷")
    st.image("https://logicpursuits.com/wp-content/uploads/2022/07/Why-Captemo-Emotion-Sentiment-Analyzer.jpg", use_column_width=True)
    
    st.text("""In the Image Analyzer section, you can upload an image to analyze the emotions 
conveyed by the individuals depicted within it. Utilizing the Facial Emotion 
Recognition (FER) technique, the system detects facial expressions and identifies 
predominant emotions such as happiness, sadness, anger, disgust, surprise, fear,  
or neutrality. These emotions are then visualized alongside the image, providing 
users with an insight into the emotional context captured within the uploaded 
image. Through this analysis, users can gain a deeper understanding of the
emotional nuances portrayed by the individuals in the image.""")

    st.text("")
    image = upload_file(["png", "jpg", "jpeg"])
    if image is not None:
        img = load_image(image)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        analyze_image(img)

# Function to analyze image using FER
def analyze_image(image):
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Initialize FER detector
    detector = FER(mtcnn=True)
    
    # Detect emotions
    emotion_results = detector.detect_emotions(img_array)
    
    if emotion_results:
        for result in emotion_results:
            bounding_box = result["box"]
            emotions = result["emotions"]
            max_emotion = max(emotions, key=emotions.get)
            st.write(f"Detected Emotion: {max_emotion} {getEmoji[max_emotion]}")
            st.write(emotions)
            # Draw bounding box and emotion label on the image
            img_array = cv2.rectangle(img_array, (bounding_box[0], bounding_box[1]), 
                                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), 
                                      (255, 0, 0), 2)
            img_array = cv2.putText(img_array, f"{max_emotion} {getEmoji[max_emotion]}", 
                                    (bounding_box[0], bounding_box[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Convert back to PIL image and display
        result_image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        st.image(result_image, caption="Analyzed Image", use_column_width=True)
    else:
        st.write("No face detected in the image or unable to detect emotions.")

# Function to render video analysis page
def show_video_page():
    st.title("Sentiment Analyzer 😊😐😕😡")
    st.subheader("Video Analyzer  🎥✨")
    st.image("https://cdn.analyticsvidhya.com/wp-content/uploads/2021/06/Learn-How-to-Implement-Face-Recognition-using-OpenCV-with-Python-80.jpg", use_column_width=True)
    
    st.text("""Upload your video and let's find sentiments in it. The Video Analyzer section 
processes uploaded videos to detect facial expressions and extract emotional cues. 
It visualizes dominant emotions such as happiness,sadness,anger,and more,providing 
users with insights into the emotional dynamics depicted in the video content.""")
    st.text("")
    video_file = upload_file(["mp4", "avi", "mov", "mkv"])
    if video_file is not None:
        st.video(video_file)
        video_path = save_uploaded_file(video_file)
        analyze_video(video_path)

# Function to save uploaded file to a temporary location
def save_uploaded_file(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_file.name

# Function to analyze video using FER
def analyze_video(video_path):
    detector = FER(mtcnn=True)
    clip = VideoFileClip(video_path)
    emotions_list = []

    # Process video frames
    for frame in clip.iter_frames(fps=1):  # Analyze one frame per second
        emotion_results = detector.detect_emotions(frame)
        if emotion_results:
            emotions_list.extend([result["emotions"] for result in emotion_results])

    if emotions_list:
        # Aggregate emotions over all frames
        avg_emotions = pd.DataFrame(emotions_list).mean().to_dict()
        max_emotion = max(avg_emotions, key=avg_emotions.get)
        st.write(f"Dominant Emotion in Video: {max_emotion} {getEmoji[max_emotion]}")

        # Plot the emotions
        fig = go.Figure([go.Bar(x=list(avg_emotions.keys()), y=list(avg_emotions.values()))])
        fig.update_layout(title='Average Emotions in Video', xaxis_title='Emotion', yaxis_title='Frequency')
        st.plotly_chart(fig)
    else:
        st.write("No faces detected in the video or unable to detect emotions.")

# Function to render text analysis page
def render_text_analysis_page():
    st.title("Sentiment Analyzer😊😐😕😡")
    st.subheader("Text Analysis 📝")
    st.image("https://miro.medium.com/v2/resize:fit:1358/0*V00v1_1CWWcQ3G6G", use_column_width=True)
    
    st.text("""In the Text Analysis section, you can input text to analyze its sentiment using 
two distinct methods. The first method, TextBlob, evaluates the sentiment polarity 
of the text, categorizing it as Positive, Negative, or Neutral based on the overal
sentiment score. On the other hand, the text2emotion method detects emotions 
expressed within the text, identifying the primary emotion as Happy, Sad, Angry, 
Fearful, or Surprised. Through these analyses, the section provides insights into
the emotional tone and sentiment conveyed by the input text, facilitating a deeper 
understanding of its underlying sentiments and emotions.""")

    st.text("")
    user_text = st.text_input('User Input', placeholder='Enter Your Text')
    st.text("")
    analysis_type = st.selectbox(
        'Type of analysis',
        ('Positive/Negative/Neutral - TextBlob', 'Happy/Sad/Angry/Fear/Surprise - text2emotion'))
    st.text("")
    if st.button('Predict'):
        if user_text != "" and analysis_type is not None:
            st.text("")
            components.html("""
                                <h3 style="color: #0284c7; font-family: Source Sans Pro, sans-serif; font-size: 28px; margin-bottom: 10px; margin-top: 50px;">Result</h3>
                                """, height=100)
            get_sentiments(user_text, analysis_type)

# Function to show sidebar and select page
def show_sidebar():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Text", "Image", "Video"])
    return selected_page

# Function to get sentiments
def get_sentiments(user_text, analysis_type):
    if analysis_type == 'Positive/Negative/Neutral - TextBlob':
        analysis = TextBlob(user_text)
        sentiment = analysis.sentiment.polarity
        if sentiment > 0:
            sentiment_label = "Positive"
            frequencies = {"Positive": 1, "Neutral": 0, "Negative": 0}
        elif sentiment < 0:
            sentiment_label = "Negative"
            frequencies = {"Positive": 0, "Neutral": 0, "Negative": 1}
        else:
            sentiment_label = "Neutral"
            frequencies = {"Positive": 0, "Neutral": 1, "Negative": 0}
        st.write(f"Sentiment: {sentiment_label}")
        
        # Display the sentiment bar chart
        fig = go.Figure([go.Bar(x=list(frequencies.keys()), y=list(frequencies.values()))])
        fig.update_layout(title='Text Sentiment Analysis', xaxis_title='Sentiment', yaxis_title='Frequency')
        st.plotly_chart(fig)
        
    elif analysis_type == 'Happy/Sad/Angry/Fear/Surprise - text2emotion':
        emotions = te.get_emotion(user_text)
        emotion = max(emotions, key=emotions.get)
        st.write(f"Emotion: {emotion} {getEmoji[emotion]}")
        
        # Display the emotion bar chart
        fig = go.Figure([go.Bar(x=list(emotions.keys()), y=list(emotions.values()))])
        fig.update_layout(title='Text Emotion Analysis', xaxis_title='Emotion', yaxis_title='Frequency')
        st.plotly_chart(fig)

# Function to render the selected page
def render_page(selected_page):
    if selected_page == "Image":
        show_image_page()
    elif selected_page == "Video":
        show_video_page()
    elif selected_page == "Text":
        render_text_analysis_page()

def main():
    selected_page = show_sidebar()
    render_page(selected_page)

if __name__ == "__main__":
    main()
