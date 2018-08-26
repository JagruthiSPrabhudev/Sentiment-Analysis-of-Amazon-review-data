# Sentiment-Analysis-of-Amazon-review-data

using the dataset  from the Amazon Reviews Kaggle competition. The goal is to perform
sentiment analysis to determine whether a review is positive or negative using a classifier in python for sentiment analysis on Amazon reviews. 

Sentiment analysis is often used to derive the emotion / opinion expressed in a text

The goal of this project is to conduct sentiment analysis on Amazon product reviews using machine learning techniques.

 The trained model was to be used to predict users’ sentiment based on their online reviews.



Sentiment Analysis on Amazon Product Reviews data
#### Part 1. Data Exploration
The dataset consisted of 400 thousand reviews of products from amazon.com.

The data set has the following fields :
Text – The review data 
Label – Binary label (positive/negative)

Below are some summary statistics about the data:
Total number of reviews: 399939
Number of positive reviews:199968
Number of negative reviews:199971

#### Part 2. Data Preparation
The dataset was separated into test and training data as follows: every 5th sample belongs to test data, the remaining samples belong to training data.

Text pre-processing is needed to convert raw reviews into cleaned review. Necessary steps include conversion to lowercase, removal of non-characters, removal of stop words, removal of html tags.

The first main step involved in text classification is to find a word embedding to convert text into numerical representations. I have used frequency based embedding model for the same.
I have implemented CountVectorizer in sklearn to compute occurrence counting of words
I have also implemented TfidfVectorizer in sklearn to compute tf-idf weighted counting. 

Once we have numerical representations of the text data, we are ready to fit the feature vectors to supervised learning algorithms
#### Part 4. Word2Vec
Word2vec is a two-layer neural net that processes text. Its input is a text corpus and its output is a set of vectors: feature vectors for words in that corpus.
#### Part 5. Machine Learning algorithm
I have constructed the following models for evaluation :


Multinomial Naïve bias
Neural Networks
Decision Tree
##### Implementation :
Fit feature vectors to supervised learning algorithm using Multinomial Naïve bias, Neural Networks and Decision Tree in sklearn
Load pre-trained model and predict the sentiment of the new data.

#### multinomial naïve bayes algorithm .
Naive Bayes is a simple and due to its simplicity, this algorithm might outperform more complex models when the data set isn’t large enough and the categories are kept simple.
Given estimates of parameters calculated from the training documents, classification is performed on test documents by calculating the posterior probability of each class given the evidence of the test document, and selecting the class with the highest probability. 
We formulate this by applying Bayes’ rule:
 P(cj |di; ˆθ) = P(cj |ˆθ)P(di|cj; ˆθj) P(di|ˆθ).
 
##### Classification report 
				 precision  	  recall 	Accuracy
	Naïve bayes            	 0.85   	   0.85    	 0.8483

	Neural Network	         0.90     	   0.90	  	 0.9002

	Decision Tree	         0.75      	   0.74  	 0.7424
              

#### Part 6.visualisation

used plotpy for visualising the results and some analysis of the data.
