import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Streamlit App Title
st.title('Text Summarizer App')

# User Input for the text
st.write('Enter the text you want to summarize below:')
text_input = st.text_area('Text', '')

# Summarize Button
if st.button('Summarize'):
    if text_input:
        parser = PlaintextParser.from_string(text_input, Tokenizer('english'))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, 2)  # Number of sentences in summary
        summary_text = ' '.join([str(sentence) for sentence in summary])
        if summary_text:
            st.subheader('Summary:')
            st.write(summary_text)
        else:
            st.write('Text is too short to summarize.')
    else:
        st.write('Please enter some text to summarize.')
