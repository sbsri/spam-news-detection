#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


True_news = pd.read_excel('True.xlsx')
Fake_news = pd.read_excel('Fake.xlsx')


# In[4]:


True_news


# In[5]:


Fake_news


# In[6]:


True_news['label'] = 0
Fake_news['label'] = 1


# In[7]:


True_news


# In[8]:


Fake_news


# In[9]:


dataset1 = True_news[['text','label']]
dataset2 = Fake_news[['text','label']]


# In[10]:


dataset1


# In[11]:


dataset2


# In[12]:


dataset = pd.concat([dataset1,dataset2])


# In[13]:


dataset


# In[14]:


dataset.shape


# In[15]:


## check for null values
dataset.isnull().sum()


# In[16]:


dataset['label'].value_counts()


# In[17]:


dataset = dataset.sample(frac=1)


# In[18]:


dataset


# # nlp part

# In[19]:


# nlp part to encode the data and to clean the data 


# In[20]:


import nltk 
import re 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[21]:


import nltk
nltk.download('wordnet')


# In[22]:


import nltk
nltk.download('stopwords')


# In[23]:


ps = WordNetLemmatizer()


# In[937]:


stopwords = set(stopwords.words('english'))


# In[24]:


nltk.download('wordnet')


# In[25]:


from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

def clean_row(row):
    row = re.sub('[^-a-zA-Z]', '', row)  # removes numbers and special symbols
    token = row.split()
    news = [pslemmatize(word) for word in token if not word in stopwords]
    cleanned_news = ' '.join(news)
    return cleanned_news


# In[26]:


#my name is bhavya and my field is AI
#my is
#my name is bhavya and field AI


# In[27]:


dataset['text']


# In[28]:


import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Specify the NLTK data directory where you have downloaded the English stopwords
nltk.data.path.append(r"C:\Users\BHAVYA SRI\AppData\Roaming\nltk_data")

# Download stopwords if you haven't already
nltk.download('stopwords')

stopwords = set(stopwords.words('english'))

def clean_row(row):
    
    row = re.sub('[^-a-zA-Z]', '', row)  # removes numbers and special symbols
    token = row.split()
    # Initialize a simple lemmatizer
    simple_lemmatizer = WordNetLemmatizer()
    news = [simple_lemmatizer.lemmatize(word) for word in token if not word in stopwords]
    cleanned_news = ' '.join(news)
    return cleanned_news


# In[29]:


import nltk
nltk.download('omw-1.4')


# In[30]:


dataset['text'] = dataset['text'].apply(lambda x : clean_row(x))


# In[31]:


dataset['text']


# In[32]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[33]:


vectorizer = TfidfVectorizer(max_features = 50,lowercase = False, ngram_range=(1,2))


# In[34]:


X = dataset.iloc[:30,0]
Y = dataset.iloc[:30,1]


# In[35]:


X


# In[36]:


Y


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


train_data,test_data,train_label,test_label = train_test_split(X,Y,test_size=0.2,random_state=2)


# In[39]:


train_data


# In[40]:


vec_train_data = vectorizer.fit_transform(train_data)


# In[41]:


train_data


# In[42]:


vec_train_data = vectorizer.fit_transform(train_data)


# In[43]:


vec_train_data = vec_train_data.toarray()


# In[44]:


type(vec_train_data)


# In[45]:


vec_test_data = vectorizer.fit_transform(test_data)


# In[46]:


vec_test_data = vec_test_data.toarray()


# In[47]:


vec_train_data.shape,vec_test_data.shape


# In[48]:


vec_train_data


# In[49]:


vec_test_data


# In[50]:


training_data = pd.DataFrame(vec_train_data)
testing_data = pd.DataFrame(vec_test_data)



# In[51]:


training_data


# In[52]:


testing_data


# # Model

# In[53]:


from sklearn.naive_bayes import MultinomialNB


# In[54]:


clf = MultinomialNB


# In[58]:


from sklearn.metrics import accuracy_score


# In[60]:


import numpy as np
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder()
train_label_array = np.array(train_label).reshape(1,-1)
                                                 
train_label_encoded = one_hot_encoder.fit_transform(train_label_array)


# In[61]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Vectorize the text data
train_data_vectorized = vectorizer.fit_transform(training_data)

# Create the Multinomial Naive Bayes classifier
clf = MultinomialNB()

# Fit the classifier to the vectorized data
clf.fit(train_data_vectorized, train_label)


# In[63]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Create the TF-IDF vectorizer and fit it on the training data
vectorizer = TfidfVectorizer()
train_data_vectorized = vectorizer.fit_transform(train_data)

# Create the Multinomial Naive Bayes classifier and fit it on the vectorized training data
clf = MultinomialNB()
clf.fit(train_data_vectorized, train_label)

# Vectorize the testing data using the same vectorizer
testing_data_vectorized = vectorizer.transform(testing_data)
# Convert non-text elements to strings
train_data_vectorized = [str(item) for item in train_data_vectorized]
testing_data_vectorized = [str(item) for item in testing_data_vectorized]

# Make predictions
y_pred = clf.predict(testing_data_vectorized)

# Calculate accuracy
accuracy = accuracy_score(test_label, y_pred)
print("Accuracy:", accuracy)


# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score
# 
# # Vectorize the training data
# vectorizer = TfidfVectorizer()
# train_data_vectorized = vectorizer.fit_transform(train_data)
# # Assuming train_data is a list of mixed data types, including integers and text
# 
# # You don't need to convert non-text elements to strings here. The TfidfVectorizer can handle mixed data types.
# 
# # Create the Multinomial Naive Bayes classifier
# clf = MultinomialNB()
# # You should use the vectorized data, not the original data.
# clf.fit(train_data_vectorized, train_label)
# 
# # Vectorize the testing data using the same vectorizer
# testing_data_vectorized = vectorizer.transform(testing_data)
# 
# # Make predictions
# y_pred = clf.predict(testing_data_vectorized)
# 
# # Calculate accuracy
# accuracy = accuracy_score(test_label, y_pred)
# print("Accuracy:", accuracy)
# 

# In[64]:


y_pred = clf.predict(testing_data_vectorized)



# In[65]:


test_label


# In[68]:


y


# In[1393]:


from sklearn.metrics import accuracy_score


# In[67]:


# Ensure y has the same length as test_label
y = Y[:len(test_label)]


# In[69]:


print("Length of y:", len(y))
print("Length of test_label:", len(test_label))


# In[70]:


accuracy_score(test_label, y)


# In[71]:


y1 = Y[:len(train_label)]


# In[72]:


print("Length of y1:", len(y1))
print("Length of train_label:", len(train_label))


# In[73]:


accuracy_score(train_label,y1)


# In[ ]:





# In[74]:


txt ='Watching telugu movie..wat abt u?'


# In[75]:


news = clean_row(txt)


# In[76]:


news


# In[77]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Create the TF-IDF vectorizer and fit it on the training data
vectorizer = TfidfVectorizer()
train_data_vectorized = vectorizer.fit_transform(train_data)

# Create the Multinomial Naive Bayes classifier and fit it on the vectorized training data
clf = MultinomialNB()
clf.fit(train_data_vectorized, train_label)

# Now, you can use this clf object for predictions
# Make sure you pass a list to vectorizer.transform even if you have a single document
news = ["Watchingtelugumoviewatabtu"]
test_data_vectorized = vectorizer.transform(news)

# Make predictions
pred = clf.predict(test_data_vectorized)

# The 'pred' variable will contain the predicted class for the given news.


# In[78]:


pred


# In[79]:


txt = input("Enter News")
news = clean_row(str(txt))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Create the TF-IDF vectorizer and fit it on the training data
vectorizer = TfidfVectorizer()
train_data_vectorized = vectorizer.fit_transform(train_data)

# Create the Multinomial Naive Bayes classifier and fit it on the vectorized training data
clf = MultinomialNB()
clf.fit(train_data_vectorized, train_label)

# Now, you can use this clf object for predictions
# Make sure you pass a list to vectorizer.transform even if you have a single document
news = ["Watchingtelugumoviewatabtu"]
test_data_vectorized = vectorizer.transform(news)

# Make predict
pred = clf.predict(test_data_vectorized)

# The 'pred' variable will contain the predicted class for the given news.



if pred == 0:
    print('News is correct')
else:
    print('News is fake')


# In[ ]:




