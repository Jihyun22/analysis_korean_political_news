# 2021.09.15
# @Jihyun22

import pandas as pd
from pandas import DataFrame  as df
import numpy as np
from collections import Counter
from konlpy.tag import Okt
okt = Okt()
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import re 
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
from wordcloud import WordCloud
from datetime import datetime


Data= pd.read_csv('./KOREA_Tweet_Data(12-04).csv', encoding="utf-16")




#keyword = '코로나'
start_year = 2020
start_month = 4
start_day = 1
finish_year = 2020
finish_month = 4
finish_day = 30


#Data = Data.loc[Data['text'].str.contains(keyword)]
#Data = Data.loc[Data['text'].str.contains('a|b')]


Data['timestamp']= pd.to_datetime(Data["timestamp"])
Data = Data[(Data.timestamp >= datetime(start_year,start_month , start_day)) & (Data.timestamp <= datetime(finish_year, finish_month, finish_day))]
Data.reset_index(drop=True, inplace=True)



element_count = {}

for item in Data['timestamp']:
    element_count.setdefault(item,0)
    element_count[item] += 1
    
tweet_count = pd.DataFrame.from_dict(element_count, orient = 'index',columns=["tweet_count"])
tweet_count.to_csv("./Result/tweet_count.csv", index = None)
tweet_count



Data.text = Data.text.astype(str)
clean_Data =Data
clean_Data ['text'] = clean_Data ['text'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣]',' ',regex=True)
clean_Data  = clean_Data .replace({'': np.nan})
clean_Data  = clean_Data .replace(r'^\s*$', None, regex=True)
#clean_Data.dropna(how='any', inplace=True)
clean_Data = clean_Data.reset_index (drop = True)
print(clean_Data.isnull().values.any()) 




Data_list=clean_Data.text.values.tolist()
data_word=[]
for i in range(len(Data_list)):
    try:
        data_word.append(okt.nouns(Data_list[i]))
    except Exception as e:
        continue
Data['clean'] = data_word




id2word=corpora.Dictionary(data_word)
id2word.filter_extremes(no_below = 20)
texts = data_word
corpus=[id2word.doc2bow(text) for text in texts]

mallet_path = 'mallet-2.0.8/bin/mallet' 
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=10, id2word=id2word)


coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()


def compute_coherence_values(dictionary, corpus, texts, limit, start=4, step=2):

    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=data_word, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values



# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=4, limit=21, step=2)






limit=21; start=4; step=2;
x = range(start, limit, step)
topic_num = 0
count = 0
max_coherence = 0
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", cv)
    coherence = cv
    if coherence >= max_coherence:
        max_coherence = coherence
        topic_num = m
        model_list_num = count   
    count = count+1

        
# Select the model and print the topics
optimal_model = model_list[model_list_num]
model_topics = optimal_model.show_topics(formatted=False)
#print(optimal_model.print_topics(num_words=10))



def format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num,topn=10)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    print(type(sent_topics_df))

    # Add original text to the end of the output
    #contents = pd.Series(texts)
    #sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    sent_topics_df = pd.concat([sent_topics_df, Data['text'],Data['timestamp'],Data['tweet_url'],Data['screen_name'],Data['label'],Data['clean']], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=Data_list)

# Format
df_topic_tweet = df_topic_sents_keywords.reset_index()
df_topic_tweet.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text','Timestamp', 'Tweet_url','Screen_name','label','Clean']

#df_dominant_topic=df_dominant_topic.sort_values(by=['Dominant_Topic'])
#df_topic_tweet


# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)


topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
topic_counts.sort_index(inplace=True)

topic_contribution = round(topic_counts/topic_counts.sum(), 4)
topic_contribution

lda_inform = pd.concat([sent_topics_sorteddf_mallet, topic_counts, topic_contribution], axis=1)
lda_inform.columns=["Topic_Num", "Topic_Perc_Contrib", "Keywords", "Text", "timestamp","Clean", "tweet_url","screen_name","label","Num_Documents", "Perc_Documents"]
lda_inform = lda_inform[["Topic_Num","Keywords","Num_Documents","Perc_Documents"]]
lda_inform
#lda_inform.Topic_Num = lda_inform.Topic_Num.astype(int)
lda_inform['Topic_Num'] =lda_inform['Topic_Num'] +1
lda_inform.Topic_Num = lda_inform.Topic_Num.astype(str)
lda_inform['Topic_Num'] =lda_inform['Topic_Num'].str.split('.').str[0]
df_topic_tweet['Dominant_Topic'] =df_topic_tweet['Dominant_Topic'] +1
df_topic_tweet.Dominant_Topic = df_topic_tweet.Dominant_Topic.astype(str)
df_topic_tweet['Dominant_Topic'] =df_topic_tweet['Dominant_Topic'].str.split('.').str[0]



lda_inform.to_csv ("./Result/lda_inform.csv", index = None)
lda_inform


#df_topic_tweet.to_csv ("./Result/df_topic_tweet.csv", index = None)
df_topic_tweet


for i in range(1,topic_num+1):
    globals()['df_{}'.format(i)]=df_topic_tweet.loc[df_topic_tweet.Dominant_Topic==str(i)]
    globals()['df_{}'.format(i)].sort_values('Topic_Perc_Contrib',ascending=False,inplace = True)
    globals()['df_{}'.format(i)].to_csv ("./Result/topic("+str(i)+")_tweet.csv", index = None)
    
df_1



for i in range(1,topic_num+1):
    #data_list = globals()['df_{}'.format(i)].Text.values.tolist()
    long_string = sum(globals()['df_{}'.format(i)].Clean.values,[])
    str(long_string)
    
    
    
    #data_word=[x for x in data_word if not x.isdigit()]
    
    freq=pd.Series(long_string).value_counts().head(50)
    freq=dict(freq)
    # Create a WordCloud object
    freq=dict(freq)
    
    wordcloud = WordCloud(font_path="./Font/BMHANNA_11yrs_ttf.ttf",
             relative_scaling = 0.2,
             background_color = 'white',
            ).generate_from_frequencies(freq)
    
    # Visualize the word cloud
    wordcloud.to_image()
    plt.figure(figsize=(16,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

    plt.savefig("./Result/topic("+str(i)+")wordcloud.png")

for i in range(1,topic_num+1):
    globals()['df_{}_pn'.format(i)]=globals()['df_{}'.format(i)].label.value_counts(normalize=True) * 100
    globals()['df_{}_pn'.format(i)]
  #  globals()['df_{}_pn'.format(i)].to_csv ("./Result/topic_tweet(topic"+str(i)+")posneg.csv", index = None)
    

df_1_pn

