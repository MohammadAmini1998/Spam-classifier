
from itertools import count
from numpy import vectorize
import pandas as pd 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer



dataset=pd.read_csv("SMSSpamCollection"
                    ,sep='\t',header=None
                    ,names=['Label', 'SMS'])


dataset=dataset.sample(frac=1,random_state=1)

train_dataset=dataset[:round(len(dataset)*.8)]
test_dataset=dataset[round(len(dataset)*.8):]

vectorizer=CountVectorizer()
counts=vectorizer.fit_transform(train_dataset['SMS'])

classifier=MultinomialNB()
targets=train_dataset['Label'].values
classifier.fit(counts,targets)


test_counts=vectorizer.transform(test_dataset['SMS'])
test_targets=test_dataset['Label'].values


print(classifier.score(test_counts,test_targets))


example=['Its the simplest of truths and the easiest to put into action.…']
example_count=vectorizer.transform(example)
prediction=classifier.predict(example_count)
print(prediction)