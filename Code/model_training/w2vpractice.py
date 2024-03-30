import io
import re
import string
import tqdm 

import numpy as np

import tensorflow as tf 
from tensorflow.keras import layers

    
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

vocab, index = {}, 1 #start indexing from 1
vocab['<pad>'] = 0 #padding token

#test vectorization
sentence = "The wide road shimmered in the hot sun"
tokens = list(sentence.lower().split())
print(len(tokens))

#vocab mapping
for token in tokens:
    if token not in vocab:
        vocab[token] = index
        index += 1
    vocab_size = len(vocab)
    print(vocab)
    
#inverse vocab to save mappings fomr integer indicies to tokens
inv_vocab = {index: token for token, index in vocab.items()}
print(inv_vocab)

#vectorize sentence
example_sequence = [vocab[word] for word in tokens]
print(example_sequence)

#generating skip-grams from one sentence
window_size = 2
positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
    example_sequence,
    vocabulary_size = vocab_size,
    window_size = window_size,
    negative_samples = 0
)
print(len(positive_skip_grams))

for target, context in positive_skip_grams[:5]:
  print(f"({target}, {context}): ({inv_vocab[target]}, {inv_vocab[context]})")
  
 
#target and context words for one positive skip-gram
target_word, context_word = positive_skip_grams[0]

#set number of negative samples per positive context
num_ns = 4

context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))
negative_sampleing_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
    true_classes = context_class, #class that should be samples as 'positive'
    num_true = 1, #each positive skip-gram has 1 positive context class
    num_sampled = num_ns,
    unique = True, #all the negative samples should be unique
    range_max = vocab_size, #pick index of samples from [0, vocab_size]
    seed = SEED, #seed for reproducibility
    name = "negative_sampling"
)
print(negative_sampleing_candidates)
print([inv_vocab[index.numpy()] for index in negative_sampleing_candidates])