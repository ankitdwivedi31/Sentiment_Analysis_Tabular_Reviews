# Use a pipeline as a high-level helper
from transformers import pipeline
import torch
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

model_path = ("C:/Users/ankitdwivedi/OneDrive - Adobe/Desktop/NLP Projects/Video to Text Summarization/Model/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/snapshots/714eb0fa89d2f80546fda750413ed43d93601a13")

# pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

analyzer = pipeline("text-classification", model=model_path)

# print(analyzer(["This product is good", "This product is expenive"]))

def sentiment_analyzer(review):
    sentiment = analyzer(review)
    return sentiment[0]['label']

def sentiment_bar_chart(df):
    sentiment_counts = df['Sentiment'].value_counts()
    #create a bar chart
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', color=['green', 'red'])
    ax.set_title('Sentiment Counts')
    # ax.set_xlabel('Sentiment')
    # ax.set_ylabel('Count')
    return fig

def Read_Analyze(file_object):
    df = pd.read_csv(file_object, encoding='latin1')

    if 'Review' not in df.columns:
        raise ValueError("Review column not found")
    df['Sentiment'] = df['Review'].apply(sentiment_analyzer)
    chart_object = sentiment_bar_chart(df)
    return df, chart_object


# result = sentiment_analyzer("C:/Users/ankitdwivedi/OneDrive - Adobe/Desktop/NLP Projects/Video to Text Summarization/Files/all-data.csv")
# print (result)

gr.close_all()

demo = gr.Interface(fn=Read_Analyze, 
                    inputs=[gr.File(file_types = ["csv"], 
                    label="Upload your review file")],
                    outputs=[gr.Dataframe(label="Reviews"), gr.Plot(label="Sentiment Analysis")],
                    title="Project 3: Sentiment Analyzer",
                    description="""This is a Sentiment Analysis Model.""")
demo.launch()