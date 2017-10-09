
# Abstractive Summarization

Loading pre-trained GloVe embeddings.
Source of Data: https://nlp.stanford.edu/projects/glove/

Another interesting embedding to look into:
https://github.com/commonsense/conceptnet-numberbatch


```python
import numpy as np
from __future__ import division

filename = 'glove.6B.50d.txt'
def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd
vocab,embd = loadGloVe(filename)

embedding = np.asarray(embd)
embedding = embedding.astype(np.float32)

word_vec_dim = len(embedding[0])
#Pre-trained GloVe embedding
```

    Loaded GloVe!





Here I will define functions for converting words to its vector representations and vice versa. 

<b>word2vec():</b> converts words to its vector representations.
If the word is not present in the vocabulary, and thus if it doesn't have any vector representation,
the word will be considered as 'unk' (denotes unknown) and the vector representation of unk will be
returned instead. 

<b>np_nearest_neighbour():</b> returns the word vector in the vocabularity that is most similar
to word vector given as an argument. The similarity is evaluated based on the formula of cosine
similarity. 

<b>vec2word():</b> converts vectors to words. If the vector representation is unknown, and no corresponding word
is known, then it returns the word representation of a known vector representation which is most similar 
to the vector given as argument (the np_nearest_neighbour() function is used for that).





```python
def np_nearest_neighbour(x):
    #returns array in embedding that's most similar (in terms of cosine similarity) to x
        
    xdoty = np.multiply(embedding,x)
    xdoty = np.sum(xdoty,1)
    xlen = np.square(x)
    xlen = np.sum(xlen,0)
    xlen = np.sqrt(xlen)
    ylen = np.square(embedding)
    ylen = np.sum(ylen,1)
    ylen = np.sqrt(ylen)
    xlenylen = np.multiply(xlen,ylen)
    cosine_similarities = np.divide(xdoty,xlenylen)

    return embedding[np.argmax(cosine_similarities)]


def word2vec(word):  # converts a given word into its vector representation
    if word in vocab:
        return embedding[vocab.index(word)]
    else:
        return embedding[vocab.index('unk')]

def vec2word(vec):   # converts a given vector representation into the represented word 
    for x in xrange(0, len(embedding)):
        if np.array_equal(embedding[x],np.asarray(vec)):
            return vocab[x]
    return vec2word(np_nearest_neighbour(np.asarray(vec)))
```

Loading processed dataset from file.
The Data is preprocessed in [Data_Pre-processing.ipynb](https://github.com/JRC1995/Abstractive-Summarization/blob/master/Data%20Pre-processing.ipynb)


Dataset source: https://www.kaggle.com/snap/amazon-fine-food-reviews


```python
import pickle

with open ('vec_summaries', 'rb') as fp:
    vec_summaries = pickle.load(fp)

with open ('vec_texts', 'rb') as fp:
    vec_texts = pickle.load(fp)
    
```

Here, I am Loading vocab_limit and embd_limit (though I may not ever use embd_limit).
Vocab_limit contains only vocabularies that are present in the dataset and 
some special words representing markers 'eos', '<PAD>' etc.

The network should output the probability distribution over the words in 
vocab_limit. So using limited vocabulary (vocab_limit) will mean requiring
less parameters for calculating the probability distribution.


```python
with open ('vocab_limit', 'rb') as fp:
    vocab_limit = pickle.load(fp)

with open ('embd_limit', 'rb') as fp:
    embd_limit = pickle.load(fp)
    
```

Here I am planning to reduce the size of the datasets, and remove some data that can be troublesome.

<b>REMOVING DATA WITH SUMMARIES WHICH ARE TOO LONG</b>

I will not be training the model in batches. I will train the model one sample at a time, because my laptop
will probably not be able to handle batch training (the kernel crashes now and then even with SGD ).

However, if I was training in batches I had to choose a fixed maximum length for output.
Each target output is marked with the word 'eos' at the end. After that each target output can be padded with
'<PAD>' to fit the maximum output length. The network can be taught to produce an output in the form
"word1 word2 eos <PAD> <PAD>". The batch training can be handled better if all target outputs are transformed
to a fixed length. 

But, the fixed length should be less than or equal to the length of the longest target output so as to
not discard any word from any target-output sample.

But there may be a few very long target outputs\summaries (say, 50+) whereas most summaries are near about
length 10. So to fix the length, lots of padding has to be done to most of the summaries just because there
are a few long summaries. 

Better to just remove the data whose summaries are bigger than a specified threshold (MAX_SUMMARY_LEN).
In this cell I will diagnose how many percentage of data will be removed for a given threshold length,
and in the next cell I will remove them.

Note: I am comparing len(summary_vec)-1, instead of len(summary_vec). The reason is that I am ignoring 
the last word vector which is the representation of the 'eos' marker. I will explain why later on this
notebook. 

<b>REMOVING DATA WITH ITS TEXTS WHOSE LENGTH IS SMALLER THAN THE WINDOW SIZE</b>

In this model I will try to implement <b>local attention</b> with standard encoder-decoder architecture.

Where global attention looks at all the hidden states of the encoder to determine where to attend to,
local attention looks only at the hidden states under the range pt-D to pt+D where D is empirically selected
and pt is a position determined by the program.
The range of pt-D to pt+D can be said to be the window where attention takes place.  Pt is the center of the
window.

I am treating D as a hyperparameter. The window size will be (pt-D)-(pt+D)+1 = 2D+1.

Now, obviously, the window needs to be smaller than or equal to the no. of the encoded hidden states themselves.
We will encode one hidden state for each words in the input text, so size of the hidden states will be equivalent
to the size of the input text.

So we must choose D such that 2D+1 is not bigger than the length of any text in the dataset.

To ensure that, I will first diagnose how many data will be removed for a given D, and in the next cell,
I will remove all input texts whose length is less than 2D+1.

<b>REMOVING DATA WITH TEXTS(REVIEWS) WHICH ARE TOO LONG</b>

The RNN encoders will encode one word at a time. No. of words in the text data or in other words,
the length of the text size will also be the no. of timesteps for the encoder RNN. To make the training less intensive 
(so that it doesn't burden my laptop too much), I will be removing
all data with whose review size exceeds a given threshold (MAX_TEXT_LEN).


```python
#DIAGNOSIS

count = 0

LEN = 7

for summary in vec_summaries:
    if len(summary)-1>LEN:
        count = count + 1
print "Percentage of dataset with summary length beyond "+str(LEN)+": "+str((count/len(vec_summaries))*100)+"% "

count = 0

D = 10 

window_size = 2*D+1

for text in vec_texts:
    if len(text)<window_size+1:
        count = count + 1
print "Percentage of dataset with text length less that window size: "+str((count/len(vec_texts))*100)+"% "

count = 0

LEN = 80

for text in vec_texts:
    if len(text)>LEN:
        count = count + 1
print "Percentage of dataset with text length more than "+str(LEN)+": "+str((count/len(vec_texts))*100)+"% "
```

    Percentage of dataset with summary length beyond 7: 16.146% 
    Percentage of dataset with text length less that window size: 2.258% 
    Percentage of dataset with text length more than 80: 40.412% 


Here I will start the aformentioned removal process.
vec_summary_reduced and vec_texts_reduced will contain the remaining data after the removal.

<b>Note: an important hyperparameter D is initialized here.</b>

D determines the window size of local attention as mentioned before. 


```python
MAX_SUMMARY_LEN = 7
MAX_TEXT_LEN = 80

#D is a major hyperparameters. Windows size for local attention will be 2*D+1
D = 10

window_size = 2*D+1

#REMOVE DATA WHOSE SUMMARIES ARE TOO BIG
#OR WHOSE TEXT LENGTH IS TOO BIG
#OR WHOSE TEXT LENGTH IS SMALLED THAN WINDOW SIZE

vec_summaries_reduced = []
vec_texts_reduced = []

i = 0
for summary in vec_summaries:
    if len(summary)-1<=MAX_SUMMARY_LEN and len(vec_texts[i])>=window_size and len(vec_texts[i])<=MAX_TEXT_LEN:
        vec_summaries_reduced.append(summary)
        vec_texts_reduced.append(vec_texts[i])
    i=i+1
```

Here, I am simply splitting the dataset into training, validation and test sets.

Note: I am not likely to be using the validation and test set. I will just check only a couple 
of training iterations. 

But no harm to keep them prepared here.


```python
train_len = int((.7)*len(vec_summaries_reduced))

train_texts = vec_texts_reduced[0:train_len]
train_summaries = vec_summaries_reduced[0:train_len]

val_len = int((.15)*len(vec_summaries_reduced))

val_texts = vec_texts_reduced[train_len:train_len+val_len]
val_summaries = vec_summaries_reduced[train_len:train_len+val_len]

test_texts = vec_texts_reduced[train_len+val_len:len(vec_summaries_reduced)]
test_summaries = vec_summaries_reduced[train_len+val_len:len(vec_summaries_reduced)]
```


```python
print train_len
```

    18293




The function transform_out() will convert the target output sample so that 
it can be in a format which can be used by tensorflow's 
sparse_softmax_cross_entropy_with_logits() for loss calculation.

Think of one hot encoding. This transformation is kind of like that.
All the words in the vocab_limit are like classes in this context.

However, instead of being precisely one hot encoded the output will be transformed
such that it will contain the list of indexes which would have been 'one' if it was one hot encoded.



```python
def transform_out(output_text):
    output_len = len(output_text)
    transformed_output = np.zeros([output_len],dtype=np.int32)
    for i in xrange(0,output_len):
        transformed_output[i] = vocab_limit.index(vec2word(output_text[i]))
    #transformed_output[output_len:MAX_LEN] = vocab_limit.index('<PAD>')
    return transformed_output   
```

Here I am simply setting up some of the rest of the hyperparameters.
K, here, is a special hyperparameter. It denotes the no. of previous hidden states
to consider for residual connections. More on that later. 


```python
#Some MORE hyperparameters and other stuffs

hidden_size = 250
learning_rate = 0.003
K = 5
vocab_len = len(vocab_limit)
training_iters = 1 
```

Setting up tensorflow placeholders.
The purpose of the placeholders are pretty much self explanatory from the name.

Note: tf_seq_len, and tf_output_len aren't really necessary. They can be derived 
from tf_text and tf_summary respectively, but I ended up making them anyway.


```python
import tensorflow as tf

#placeholders
tf_text = tf.placeholder(tf.float32, [None,word_vec_dim])
tf_seq_len = tf.placeholder(tf.int32)
tf_summary = tf.placeholder(tf.int32,[None])
tf_output_len = tf.placeholder(tf.int32)
```

I will be using the encoder-decoder architecture.
For the encoder I will be using a bi-directional Recurrent Neural Net.
Below is the function of the forward encoder (the RNN in the forward direction
that starts from the first word and encodes a word in the context of previous words)

The RNN used here, is a plain RNN with some tricks.

I am using hidden wieght matrix which are initialized with an identity.
This idea is borrowed from: https://arxiv.org/abs/1504.00941
        
The intuition behind it is that if one uses a plain RNN with ReLu activation,
and hidden weight matrix as an identity, then if the input is null, and the bias
is also initialized as with zeros, the hidden state won't change 
no matter how many timesteps - which is how it should be. Intuitively,
If the input has near zero values, then the context shouldn't be influenced as much.

I am aslo including residual recurrent attention connections.

Remember, the hyperparameter K?

The model will compute the weighted sum (weighted based on some trainable parameters
in the attention weight matrix) of the PREVIOUS K hidden states - the weighted sum
is denoted as RRA in this function.

hidden_residuals will contain the last K hidden states.

The RRA will then be added to the current hidden state. 

The idea of RRA is borrowed from:https://arxiv.org/abs/1709.03714

(The attention weight matrix is to be normalized by dividing each elements by the sum of all 
the elements as said in the paper. But, here, I am normalizing it by softmax)

The purpose for this is to created connections between hidden states of different timesteps,
to establish long term dependencies.

Note: Adding RRA may somewhat oppose the purpose behind identity initialization. But, anyway,
identity initialization (identity being orthonological) can still be a better initialization
that random_normal or truncated_normal:
https://smerity.com/articles/2016/orthogonal_init.html

For activation, I am using ELU (Exponential Linear Units): https://arxiv.org/pdf/1511.07289.pdf

The identity initialization was supposed to be used with ReLu, however. But, at this point
I am not intending to make a pure duplicate of IRNN. 

This is more of a mix mash of various ideas. 


```python
def forward_encoder(inp,hidden,Wxh,Whh,Wattention,B,seq_len,inp_dim):

    Wattention = tf.nn.softmax(Wattention,0)
    hidden_forward = tf.TensorArray(size=seq_len,dtype=tf.float32)
    
    hidden_residuals = tf.TensorArray(size=K,dynamic_size=True,dtype=tf.float32,clear_after_read=False)
    hidden_residuals = hidden_residuals.unstack(tf.zeros([K,hidden_size],dtype=tf.float32))
    
    i=0
    j=K
    
    def cond(i,j,hidden,hidden_forward,hidden_residuals):
        return i < seq_len
    
    def body(i,j,hidden,hidden_forward,hidden_residuals):
        
        x = tf.reshape(inp[i],[1,inp_dim])
        
        hidden_residuals_stack = hidden_residuals.stack()
        
        RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-K:j],Wattention),0)
        RRA = tf.reshape(RRA,[1,hidden_size])
        
        hidden_next = tf.nn.elu(tf.matmul(x,Wxh) + tf.matmul(hidden,Whh) + B + RRA)
        
        hidden = hidden_next
        
        hidden_residuals = tf.cond(tf.equal(j,seq_len-1+K),
                                   lambda: hidden_residuals,
                                   lambda: hidden_residuals.write(j,tf.reshape(hidden,[hidden_size])))

        hidden_forward = hidden_forward.write(i,tf.reshape(hidden,[hidden_size]))
        
        return i+1,j+1,hidden,hidden_forward,hidden_residuals
    
    _,_,_,hidden_forward,hidden_residuals = tf.while_loop(cond,body,[i,j,hidden,hidden_forward,hidden_residuals])
    
    hidden_residuals.close().mark_used()
    
    return hidden_forward.stack()
        
```

This is the function for the backward encoder.
It starts from the last word in the input sequence, and encodes
a word in the context of the next word.

It similarly uses a RNN with RRA, identity initalization and elu.  


```python
def backward_encoder(inp,hidden,Wxh,Whh,Wattention,B,seq_len,inp_dim):
    Wattention = tf.nn.softmax(Wattention,0)
    hidden_backward = tf.TensorArray(size=seq_len,dtype=tf.float32)
    
    hidden_residuals = tf.TensorArray(size=K,dynamic_size=True,dtype=tf.float32,clear_after_read=False)
    hidden_residuals = hidden_residuals.unstack(tf.zeros([K,hidden_size],dtype=tf.float32))
    
    i=seq_len-1
    j=K
    
    def cond(i,j,hidden,hidden_backward,hidden_residuals):
        return i > -1
    
    def body(i,j,hidden,hidden_backward,hidden_residuals):
        
        x = tf.reshape(inp[i],[1,inp_dim])
        
        hidden_residuals_stack = hidden_residuals.stack()
        
        RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-K:j],Wattention),0)
        RRA = tf.reshape(RRA,[1,hidden_size])
        
        hidden_next = tf.nn.elu(tf.matmul(x,Wxh) + tf.matmul(hidden,Whh) + B + RRA)
        
        hidden = hidden_next
        hidden_residuals = tf.cond(tf.equal(j,seq_len-1+K),
                                   lambda: hidden_residuals,
                                   lambda: hidden_residuals.write(j,tf.reshape(hidden,[hidden_size])))
        
        hidden_backward = hidden_backward.write(i,tf.reshape(hidden,[hidden_size]))
        
        return i-1,j+1,hidden,hidden_backward,hidden_residuals
    
    _,_,_,hidden_backward,hidden_residuals = tf.while_loop(cond,body,[i,j,hidden,hidden_backward,hidden_residuals])

    hidden_residuals.close().mark_used()
    
    return hidden_backward.stack()
        
```


Pretty self explanatory - this is the function for the decoder.
It similarly uses a RNN with RRA, identity initalization and elu.  


```python
def decoder(inp,hidden,Wxh,Whh,B,RRA):
    hidden_next = tf.nn.elu(tf.matmul(inp,Wxh) + tf.matmul(hidden,Whh) + B + RRA)
    
    return hidden_next
```

The cell below includes some major functions for the attention mechanism.

The attention mechanism is usually implemented to compute an attention score 
for each of the encoded hidden state in the context of a particular
decoder hidden state in each timestep - all to determine which encoded hidden
states to attend to for a particular decoder hidden state context.

More specifically, I am here implementing local attention as opposed to global attention.

I already mentioned local attention before. Local attention mechanism involves focusing on
a subset of encoded hidden states, whereas a gloabl attention mechanism invovles focusing on all
the encoded hidden states.

This is the paper on which this implementation is based on:
https://nlp.stanford.edu/pubs/emnlp15_attn.pdf
    
Following the formulas presented in the paper, first, I am computing
the position pt (the center of the window of attention).

pt is simply a position in the sequence.
For a given pt, the model will only consider the hidden state starting from the position
pt-D to the hidden state at the position pt+D. 

To say a hidden state is at position p, I mean to say that the hidden state is the encoded
representation of a word at position p in the sequence.

The paper formulates the equation for calculating pt like this:
pt = sequence_length x sigmoid(..some linear algebras and activations...)

But, I didn't used the sequence_length of the whole text which is tf_seq_len but 'positions' which
is = tf_seq_len-1-2D or tf_seq_len-(2D+1)

if pt = tf_seq_len x sigmoid(tensor)

Then pt will be in the range 0 to tf_seq_len

But, we can't have that. There is no tf_seq_len position. Since the length is tf_seq_len,
the available positions are 0 to (tf_seq_len-1). Which is why I subtracted 1 from it.

Next, we must have the value of pt to be such that it represents the CENTER of the window.
If pt is too close to 0, pt-D will be negative - a non-existent position.
If pt is too close to tf_seq_len, pt+D will be a non-existent position.

So pt can't occupy the first D positions (0 to D-1) and it can't occupy the last D positions
((tf_seq_len-D) to (tf_seq_len-1)) in order to keep pt-D and pt+D as legal positions.
So a total 2D positions should be restricted to pt.

Which is why I further subtracted 2D from tf_seq_len.

Still, after calculating pt = positions x sigmoid(tensor)
where positions = tf_seq_len-(2D+1), 
pt will merely range between 0 to tf_seq_len-(2D+1)

We can't still accept pt to be 0 since pt-D will be negative. But the length of the range 
of integer positions pt can occupy is now perfect.

So at this point, we can simply center pt at the window by adding a D.

After that, pt will range from D to (tf_seq_len-1)-D

Now, it can be checked that pt+D, or pt-D will never become negative or exceed
the total sequence length.

After calculating pt, we can use the formulas presented in the paper to calculate
the G score which signifies the weight (or attention) that should be given to a hidden state.

G scores is calculated for each of hidden states in the local window. This is equivalent to
a(s) used in the paper.

The function returns the G scores and the position pt, so that the model can create the 
context vector. 


```python
def score(hs,ht,Wa,seq_len):
    return tf.reshape(tf.matmul(tf.matmul(hs,Wa),tf.transpose(ht)),[seq_len])

def align(hs,ht,Wp,Vp,Wa,tf_seq_len):
   
    pd = tf.TensorArray(size=(2*D+1),dtype=tf.float32)
    
    sequence_length = tf_seq_len-tf.constant((2*D+1),dtype=tf.int32)
    sequence_length = tf.cast(sequence_length,dtype=tf.float32)
    
    pt_float = tf.multiply(sequence_length,tf.nn.sigmoid(tf.matmul(tf.tanh(tf.matmul(ht,Wp)),Vp)))
    pt_float = tf.add(pt_float,tf.cast(D,tf.float32))
    
    pt_float = tf.reshape(pt_float,[])
    
    pt = tf.cast(pt_float,tf.int32)
    
    sigma = tf.constant(D/2,dtype=tf.float32)
    
    i = 0
    pos = pt - D
    
    def cond(i,pos,pd):
        
        return i < (2*D+1)
                      
    def body(i,pos,pd):
        
        comp_1 = tf.cast(tf.square(tf.cast(pos,tf.float32)-pt_float),tf.float32)
        comp_2 = tf.cast(2*tf.square(sigma),tf.float32)
            
        pd = pd.write(i,tf.exp(-(comp_1/comp_2)))
            
        return i+1,pos+1,pd
                      
    i,pos,pd = tf.while_loop(cond,body,[i,pos,pd])
    
    local_hs = hs[(pt-D):(pt+D+1)]
    
    normalized_scores = tf.nn.softmax(score(local_hs,ht,Wa,2*D+1))
    
    pd=pd.stack()
    
    G = tf.multiply(normalized_scores,pd)
    G = tf.reshape(G,[2*D+1,1])
    
    return G,pt

```
This is the model definition.

First is the <b>bi-directional encoder</b>.

h_forward is the tensorarray of all the hidden states from the 
forward encoder whereas h_backward is the tensorarray of all the hidden states
from the backward encoder.

The final list of encoder hidden states are usually calculated by combining 
the equivalents of h_forward and h_backward by some means.

There are many means of combining them, like: concatenation, summation, average etc.
    
I will be using concatenation.

hidden_encoder is the final list of encoded hidden state

Now, there is the question about initializing the decoder hidden state.
I was a bit confused about it. In the end, I am using the first encoded_hidden_state 
as the initial decoder state. The first encoded_hidden_state may have the least 
past context (none actually) but, it will have the most future context.

Next the <b>attention function</b> is called, to compute the G score.
The context vector is created by the weight (weighted in terms of G scores) summation
of hidden states in the local attention window.

I used the formulas mentioned here: https://nlp.stanford.edu/pubs/emnlp15_attn.pdf

to calculate the first y (output) from the context vector and decoder hidden state.

Note: y is of the same size as the no. of vocabs in vocab_limit. Y is supposed to be 
a probability distribution. The value of index i of Y denotes the probability for Y 
to be the word that is located in the index i of vacab_limit.

('beam search' is another approach to look into, for not predicting simply the next word,
but the next k words.)

The first y is the input for the <b>decoder RNN</b>. In the context of the first y and 
the initial hidden state the RNN produces the next decoder hidden state and the loop
continues. 

Here, I used the attended_hidden_state as the initial hidden state for the
decoder RNN. Though I have some confusion about this, it seemed to produce 
results with better variety.

Since I will be training sample to sample, I can dynamically send the output length 
of the current sample, and the decoder loops for the given 'output length' times.

NOTE: I am saving y without softmax in the tensorarray output. Why? Because
I will be using tensorflow cost functions that requires the logits to be without
softmax (the function will internally apply Softmax).



```python
def model(tf_text,tf_seq_len,tf_output_len):
    
    #PARAMETERS
    
    #1. GENERAL ENCODER PARAMETERS
    
    #Whf = tf.Variable(tf.truncated_normal(shape=[],stddev=0.01))
    
    #1.1 FORWARD ENCODER PARAMETERS
    
    initial_hidden_f = tf.zeros([1,hidden_size],dtype=tf.float32)
    Wxh_f = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    Whh_f = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    B_f = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    Wattention_f = tf.Variable(tf.zeros([K,1]),dtype=tf.float32)
                               
    #1.2 BACKWARD ENCODER PARAMETERS
    
    initial_hidden_b = tf.zeros([1,hidden_size],dtype=tf.float32)
    Wxh_b = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    Whh_b = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    B_b = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    Wattention_b = tf.Variable(tf.zeros([K,1]),dtype=tf.float32)
    
    #2 ATTENTION PARAMETERS
    
    Wp = tf.Variable(tf.truncated_normal(shape=[2*hidden_size,50],stddev=0.01))
    Vp = tf.Variable(tf.truncated_normal(shape=[50,1],stddev=0.01))
    Wa = tf.Variable(tf.truncated_normal(shape=[2*hidden_size,2*hidden_size],stddev=0.01))
    Wc = tf.Variable(tf.truncated_normal(shape=[4*hidden_size,2*hidden_size],stddev=0.01))
    
    #3 DECODER PARAMETERS
    
    Ws = tf.Variable(tf.truncated_normal(shape=[2*hidden_size,vocab_len],stddev=0.01))
    
    Wxh_d = tf.Variable(tf.truncated_normal(shape=[vocab_len,2*hidden_size],stddev=0.01))
    Whh_d = tf.Variable(np.eye(2*hidden_size),dtype=tf.float32)
    B_d = tf.Variable(tf.zeros([1,2*hidden_size]),dtype=tf.float32)
    
    hidden_residuals = tf.TensorArray(size=K,dynamic_size=True,dtype=tf.float32,clear_after_read=False)
    hidden_residuals = hidden_residuals.unstack(tf.zeros([K,2*hidden_size],dtype=tf.float32))
    
    Wattention_d = tf.Variable(tf.zeros([K,1]),dtype=tf.float32)
    
    output = tf.TensorArray(size=tf_output_len,dtype=tf.float32)
                               
    #BI-DIRECTIONAL ENCODER
                               
    hidden_forward = forward_encoder(tf_text,
                                     initial_hidden_f,
                                     Wxh_f,Whh_f,Wattention_f,B_f,
                                     tf_seq_len,
                                     word_vec_dim)
    
    hidden_backward = backward_encoder(tf_text,
                                       initial_hidden_b,
                                       Wxh_b,Whh_b,Wattention_b,B_b,
                                       tf_seq_len,
                                       word_vec_dim)
    
    #Whf = tf.nn.sigmoid(Whf)
    
    #encoded_hidden = tf.multiply(hidden_forward,Whf) + tf.multiply(hidden_backward,(1-Whf))
    
    encoded_hidden = tf.concat([hidden_forward,hidden_backward],1)
    
    #ATTENTION MECHANISM AND DECODER
    
    decoded_hidden = encoded_hidden[0]
    decoded_hidden = tf.reshape(decoded_hidden,[1,2*hidden_size])
    Wattention_d_normalized = tf.nn.softmax(Wattention_d)
                               
    i=0
    j=K
    
    def attention_decoder_cond(i,j,decoded_hidden,hidden_residuals,output):
        return i < tf_output_len
    
    def attention_decoder_body(i,j,decoded_hidden,hidden_residuals,output):
        
        #LOCAL ATTENTION
        
        G,pt = align(encoded_hidden,decoded_hidden,Wp,Vp,Wa,tf_seq_len)
        local_encoded_hidden = encoded_hidden[pt-D:pt+D+1]
        weighted_encoded_hidden = tf.multiply(local_encoded_hidden,G)
        context_vector = tf.reduce_sum(weighted_encoded_hidden,0)
        context_vector = tf.reshape(context_vector,[1,2*hidden_size])
        
        attended_hidden = tf.tanh(tf.matmul(tf.concat([context_vector,decoded_hidden],1),Wc))
        
        #DECODER
        
        y = tf.matmul(attended_hidden,Ws)
        
        output = output.write(i,tf.reshape(y,[vocab_len]))
        
        y = tf.nn.softmax(y)
        
        hidden_residuals_stack = hidden_residuals.stack()
        
        RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-K:j],Wattention_d_normalized),0)
        RRA = tf.reshape(RRA,[1,2*hidden_size])
        
        decoded_hidden_next = decoder(y,attended_hidden,Wxh_d,Whh_d,B_d,RRA)
        
        decoded_hidden = decoded_hidden_next

        hidden_residuals = hidden_residuals.write(j,tf.reshape(decoded_hidden,[2*hidden_size]))
        
        return i+1,j+1,decoded_hidden,hidden_residuals,output
    
    i,j,decoded_hidden,hidden_residuals,output = tf.while_loop(attention_decoder_cond,
                                            attention_decoder_body,
                                            [i,j,decoded_hidden,hidden_residuals,output])
    hidden_residuals.close().mark_used()
    
    output = output.stack()
    
    return output
```


The model function is initiated here. The output is
computed. Cost function and optimizer are defined.
I am creating a prediction tensorarray which will 
store the index of maximum element of 
the output probability distributions.
From that index I can find the word in vocab_limit
which is represented by it. So the final visible
predictions will be the words that the model decides to
be most probable.


```python
output = model(tf_text,tf_seq_len,tf_output_len)

#OPTIMIZER

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf_summary))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#PREDICTION

pred = tf.TensorArray(size=tf_output_len,dtype=tf.int32)

i=0

def cond_pred(i,pred):
    return i<tf_output_len
def body_pred(i,pred):
    pred = pred.write(i,tf.cast(tf.argmax(output[i]),tf.int32))
    return i+1,pred

i,pred = tf.while_loop(cond_pred,body_pred,[i,pred]) 

prediction = pred.stack()
```

Finally, this is where training takes place.
It's all pretty self explanatory, but one thing to note is that
I am sending "train_summaries[i][0:len(train_summaries[i])-1]"
to the transform_out() function. That is, I am ignoring the last
word from summary. The last word marks the end of the summary.
It's 'eos'. 

I trained it before without dynamically feeding the output_len.
Ideally the network should determine the output_len by itself.

Which is why I defined (in past) a MAX_LEN, and transformed target outputs in
the form "word1 word2 word3....eos <PAD> <PAD>....until max_length"
I created the model output in the same way.

The model would ideally learn in which context and where to put eos.
And then the only the portion before eos can be shown to the user.

After training, the model can even be modified to run until,
the previous output y denotes an eos. 

That way, we can have variable length output, with the length decided
by the model itself, not the user.

But all the padding and eos, makes the model to come in contact with 
pads and eos in most of the target output. The model learns to consider eos and 
pad to be important. Trying to fit to the data, the early model starts to
spam eos and pad in its predicted output.

That necessarily isn't a problem. The model may learn to fare better
later on, but I planned only to check a couple of early iterations, 
and looking at predictions consisting of only eos and pads
isn't too interesting. I wanted to check what kind of words (other than
eos and pads) the model learns to produce in the early iterations. 

Which is why I am doing what I am doing. Ideally, my past implemention
waould be better. 

As I said before, I will run it for only a few early iterations.
So, it's not likely to see any great predicted summaries here.
As can be seen, the summaries seem more influenced by previous 
output sample than the input context in these early iterations.

Some of the texts contains undesirable words like br tags and so
on. So better preprocessing and tokenization may be desirable.

With more layer depth, larger hidden size, mini-batch training,
and other changes, this model may have potential, or may not.

The same arcitechture should be usable for training on translation data.


```python
import string
from __future__ import print_function

init = tf.global_variables_initializer()


with tf.Session() as sess: # Start Tensorflow Session
    
    saver = tf.train.Saver() 
    # Prepares variable for saving the model
    sess.run(init) #initialize all variables
    step = 0   
    loss_list=[]
    acc_list=[]
    val_loss_list=[]
    val_acc_list=[]
    best_val_acc=0
    display_step = 1
    
    while step < training_iters:
        
        total_loss=0
        total_acc=0
        total_val_loss = 0
        total_val_acc = 0
           
        for i in xrange(0,train_len):
            
            train_out = transform_out(train_summaries[i][0:len(train_summaries[i])-1])
            
            if i%display_step==0:
                print("\nIteration: "+str(i))
                print("Training input sequence length: "+str(len(train_texts[i])))
                print("Training target outputs sequence length: "+str(len(train_out)))
            
                print("\nTEXT:")
                flag = 0
                for vec in train_texts[i]:
                    if vec2word(vec) in string.punctuation or flag==0:
                        print(str(vec2word(vec)),end='')
                    else:
                        print((" "+str(vec2word(vec))),end='')
                    flag=1

                print("\n")


            # Run optimization operation (backpropagation)
            _,loss,pred = sess.run([optimizer,cost,prediction],feed_dict={tf_text: train_texts[i], 
                                                    tf_seq_len: len(train_texts[i]), 
                                                    tf_summary: train_out,
                                                    tf_output_len: len(train_out)})
            
         
            if i%display_step==0:
                print("\nPREDICTED SUMMARY:\n")
                flag = 0
                for index in pred:
                    #if int(index)!=vocab_limit.index('eos'):
                    if vocab_limit[int(index)] in string.punctuation or flag==0:
                        print(str(vocab_limit[int(index)]),end='')
                    else:
                        print(" "+str(vocab_limit[int(index)]),end='')
                    flag=1
                print("\n")
                
                print("ACTUAL SUMMARY:\n")
                flag = 0
                for vec in train_summaries[i]:
                    if vec2word(vec)!='eos':
                        if vec2word(vec) in string.punctuation or flag==0:
                            print(str(vec2word(vec)),end='')
                        else:
                            print((" "+str(vec2word(vec))),end='')
                    flag=1

                print("\n")
            
                #print(hs)
            
                print("loss="+str(loss))
            
            #print(h)
            #print(out)
            #print(ht_s)
            
        step=step+1
    
```

    
    Iteration: 0
    Training input sequence length: 51
    Training target outputs sequence length: 4
    
    TEXT:
    i have bought several of the vitality canned dog food products and have found them all to be of good quality. the product looks more like a stew than a processed meat and it smells better. my labrador is finicky and she appreciates this product better than most.
    
    
    PREDICTED SUMMARY:
    
    nor milks milks originating
    
    ACTUAL SUMMARY:
    
    good quality dog food
    
    loss=10.622
    
    Iteration: 1
    Training input sequence length: 37
    Training target outputs sequence length: 3
    
    TEXT:
    product arrived labeled as jumbo salted peanuts ... the peanuts were actually small sized unsalted. not sure if this was an error or if the vendor intended to represent the product as `` jumbo ''.
    
    
    PREDICTED SUMMARY:
    
    division winn dog
    
    ACTUAL SUMMARY:
    
    not as advertised
    
    loss=10.4359
    
    Iteration: 2
    Training input sequence length: 46
    Training target outputs sequence length: 2
    
    TEXT:
    if you are looking for the secret ingredient in robitussin i believe i have found it. i got this in addition to the root beer extract i ordered( which was good) and made some cherry soda. the flavor is very medicinal.
    
    
    PREDICTED SUMMARY:
    
    good 3:00
    
    ACTUAL SUMMARY:
    
    cough medicine
    
    loss=10.4655
    
    Iteration: 3
    Training input sequence length: 32
    Training target outputs sequence length: 2
    
    TEXT:
    great taffy at a great price. there was a wide assortment of yummy taffy. delivery was very quick. if your a taffy lover, this is a deal.
    
    
    PREDICTED SUMMARY:
    
    good as
    
    ACTUAL SUMMARY:
    
    great taffy
    
    loss=10.5359
    
    Iteration: 4
    Training input sequence length: 30
    Training target outputs sequence length: 4
    
    TEXT:
    this taffy is so good. it is very soft and chewy. the flavors are amazing. i would definitely recommend you buying it. very satisfying!!
    
    
    PREDICTED SUMMARY:
    
    cough medicine medicine medicine
    
    ACTUAL SUMMARY:
    
    wonderful, tasty taffy
    
    loss=10.049
    
    Iteration: 5
    Training input sequence length: 29
    Training target outputs sequence length: 2
    
    TEXT:
    right now i 'm mostly just sprouting this so my cats can eat the grass. they love it. i rotate it around with wheatgrass and rye too
    
    
    PREDICTED SUMMARY:
    
    cough medicine
    
    ACTUAL SUMMARY:
    
    yay barley
    
    loss=10.4939
    
    Iteration: 6
    Training input sequence length: 29
    Training target outputs sequence length: 3
    
    TEXT:
    this is a very healthy dog food. good for their digestion. also good for small puppies. my dog eats her required amount at every feeding.
    
    
    PREDICTED SUMMARY:
    
    cough medicine medicine
    
    ACTUAL SUMMARY:
    
    healthy dog food
    
    loss=9.12638
    
    Iteration: 7
    Training input sequence length: 24
    Training target outputs sequence length: 4
    
    TEXT:
    the strawberry twizzlers are my guilty pleasure- yummy. six pounds will be around for a while with my son and i.
    
    
    PREDICTED SUMMARY:
    
    cough medicine medicine medicine
    
    ACTUAL SUMMARY:
    
    strawberry twizzlers- yummy
    
    loss=10.2866
    
    Iteration: 8
    Training input sequence length: 45
    Training target outputs sequence length: 2
    
    TEXT:
    i love eating them and they are good for watching tv and looking at movies! it is not too sweet. i like to transfer them to a zip lock baggie so they stay fresh so i can take my time eating them.
    
    
    PREDICTED SUMMARY:
    
    cough taffy
    
    ACTUAL SUMMARY:
    
    poor taste
    
    loss=10.3284
    
    Iteration: 9
    Training input sequence length: 28
    Training target outputs sequence length: 3
    
    TEXT:
    i am very satisfied with my unk purchase. i shared these with others and we have all enjoyed them. i will definitely be ordering more.
    
    
    PREDICTED SUMMARY:
    
    cough taffy taffy
    
    ACTUAL SUMMARY:
    
    love it!
    
    loss=10.4985
    
    Iteration: 10
    Training input sequence length: 31
    Training target outputs sequence length: 3
    
    TEXT:
    candy was delivered very fast and was purchased at a reasonable price. i was home bound and unable to get to a store so this was perfect for me.
    
    
    PREDICTED SUMMARY:
    
    cough taffy taffy
    
    ACTUAL SUMMARY:
    
    home delivered unk
    
    loss=10.5958
    
    Iteration: 11
    Training input sequence length: 52
    Training target outputs sequence length: 2
    
    TEXT:
    my husband is a twizzlers addict. we 've bought these many times from amazon because we 're government employees living overseas and ca n't get them in the country we are assigned to. they 've always been fresh and tasty, packed well and arrive in a timely manner.
    
    
    PREDICTED SUMMARY:
    
    cough taffy
    
    ACTUAL SUMMARY:
    
    always fresh
    
    loss=10.6754
    
    Iteration: 12
    Training input sequence length: 68
    Training target outputs sequence length: 1
    
    TEXT:
    i bought these for my husband who is currently overseas. he loves these, and apparently his staff likes them unk< br/> there are generous amounts of twizzlers in each 16-ounce bag, and this was well worth the price.< a unk '' http: unk ''> twizzlers, strawberry, 16-ounce bags( pack of 6)< unk>
    
    
    PREDICTED SUMMARY:
    
    cough
    
    ACTUAL SUMMARY:
    
    twizzlers
    
    loss=9.49611
    
    Iteration: 13
    Training input sequence length: 31
    Training target outputs sequence length: 3
    
    TEXT:
    i can remember buying this candy as a kid and the quality has n't dropped in all these years. still a superb product you wo n't be disappointed with.
    
    
    PREDICTED SUMMARY:
    
    cough taffy taffy
    
    ACTUAL SUMMARY:
    
    delicious product!
    
    loss=9.60363
    
    Iteration: 14
    Training input sequence length: 21
    Training target outputs sequence length: 1
    
    TEXT:
    i love this candy. after weight watchers i had to cut back but still have a craving for it.
    
    
    PREDICTED SUMMARY:
    
    cough
    
    ACTUAL SUMMARY:
    
    twizzlers
    
    loss=7.11544
    
    Iteration: 15
    Training input sequence length: 72
    Training target outputs sequence length: 7
    
    TEXT:
    i have lived out of the us for over 7 yrs now, and i so miss my twizzlers!! when i go back to visit or someone visits me, i always stock up. all i can say is yum!< br/> sell these in mexico and you will have a faithful


```python

```

BLEU and ROUGE metrics can be implemented for scoring. Validation and testing can be easily implemented too, if needed.



