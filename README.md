## Imports


```python
import pprint
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import treebank
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
```

## Hyperparams


```python
TRAIN_SPLIT = 0.8

# Train samples (Training on all dataset is time-consuming)
N_TRAIN = 10000
```

## Datasets download


```python
nltk.download('treebank')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

    [nltk_data] Downloading package treebank to
    [nltk_data]     C:\Users\alise\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping corpora\treebank.zip.
    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\alise\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping tokenizers\punkt.zip.
    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     C:\Users\alise\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping taggers\averaged_perceptron_tagger.zip.
    




    True



## Tagged sentences


```python
tagged_sentences = treebank.tagged_sents()
```


```python
print(f"Tagged sentences: {len(tagged_sentences)}")
print(f"Tagged words: {len(nltk.corpus.treebank.tagged_words())}")
```

    Tagged sentences: 3914
    Tagged words: 100676
    


```python
pprint.pprint(tagged_sentences[0])
```

    [('Pierre', 'NNP'),
     ('Vinken', 'NNP'),
     (',', ','),
     ('61', 'CD'),
     ('years', 'NNS'),
     ('old', 'JJ'),
     (',', ','),
     ('will', 'MD'),
     ('join', 'VB'),
     ('the', 'DT'),
     ('board', 'NN'),
     ('as', 'IN'),
     ('a', 'DT'),
     ('nonexecutive', 'JJ'),
     ('director', 'NN'),
     ('Nov.', 'NNP'),
     ('29', 'CD'),
     ('.', '.')]
    

## Sample for output of our PoS tagger


```python
pprint.pprint(pos_tag(word_tokenize('This is my friend, John.')))
```

    [('This', 'DT'),
     ('is', 'VBZ'),
     ('my', 'PRP$'),
     ('friend', 'NN'),
     (',', ','),
     ('John', 'NNP'),
     ('.', '.')]
    

## Feature extractor


```python
def feature_extractor(sentence, index):
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }
```


```python
pprint.pprint(feature_extractor(['This', 'is', 'a', 'sentence'], 2))
```

    {'capitals_inside': False,
     'has_hyphen': False,
     'is_all_caps': False,
     'is_all_lower': True,
     'is_capitalized': False,
     'is_first': False,
     'is_last': False,
     'is_numeric': False,
     'next_word': 'sentence',
     'prefix-1': 'a',
     'prefix-2': 'a',
     'prefix-3': 'a',
     'prev_word': 'is',
     'suffix-1': 'a',
     'suffix-2': 'a',
     'suffix-3': 'a',
     'word': 'a'}
    

## Dataset


```python
def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]
```


```python
def transform_to_dataset(tagged_sentences):
    X = []
    y = []
    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(feature_extractor(untag(tagged), index))
            y.append(tagged[index][1])
    return X, y
```


```python
part = int(TRAIN_SPLIT * len(tagged_sentences))
training_sentences = tagged_sentences[:part]
test_sentences = tagged_sentences[part:]
```


```python
print(f"Number of train sentences: {len(training_sentences)}")
print(f"Number of test sentences: {len(test_sentences)}")
```

    Number of train sentences: 3131
    Number of test sentences: 783
    


```python
X_train, y_train = transform_to_dataset(training_sentences)
X_test, y_test = transform_to_dataset(test_sentences)
```


```python
print(f"Length X_train: {len(X_train)}")
print(f"Length X_test: {len(X_test)}")
```

    Length X_train: 80637
    Length X_test: 20039
    


```python
dict_vectorizer = DictVectorizer(sparse=False)
dict_vectorizer.fit(X_train, y_train)
```



## Model and training



```python
classifier = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])
```



```python
classifier.fit(X_train[:N_TRAIN], y_train[:N_TRAIN])
```



## Evaluation



```python
train_accuracy = classifier.score(X_train, y_train)
print(f"Train Accuracy: {train_accuracy}")
```

    Train Accuracy: 0.8958790629611717
    


```python
test_accuracy = classifier.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
```

    Test Accuracy: 0.8934078546833675
    

## Model output


```python
def my_pos_tag(sentence, clf):
    tags = clf.predict([feature_extractor(sentence, index) for index in range(len(sentence))])
    return list(zip(sentence, tags))
```


```python
my_sentence = 'This is my friend, John.'
my_tagged_sentence = my_pos_tag(word_tokenize(my_sentence), classifier)
```


```python
pprint.pprint(my_tagged_sentence)
```

    [('This', 'DT'),
     ('is', 'VBZ'),
     ('my', 'NN'),
     ('friend', 'NN'),
     (',', ','),
     ('John', 'NNP'),
     ('.', '.')]
    
