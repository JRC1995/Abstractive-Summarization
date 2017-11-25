
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


Here, I will define functions for converting words to their vector representations, and vice versa. 

### word2vec: 

Converts words to their vector representations.
If a word is not present in the vocabulary, and thus if it doesn't have any vector representation,
the word will be considered as 'unk' (denotes unknown) and the vector representation of unk will be
returned instead. Note: It has nothing to do with 'word2vec' embedding. 

### np_nearest_neighbour:

Returns the word vector in the vocabularity that is most similar
to word vector given as an argument. The similarity is evaluated based on the formula of cosine
similarity. 

### vec2word: 

Converts vectors to words. If the vector representation is unknown, and no corresponding word
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

### Loading pre-processed dataset

The Data is preprocessed in [Data_Pre-processing.ipynb](https://github.com/JRC1995/Abstractive-Summarization/blob/master/Data%20Pre-processing.ipynb)

Dataset source: https://www.kaggle.com/snap/amazon-fine-food-reviews


```python
import pickle

with open ('vec_summaries', 'rb') as fp:
    vec_summaries = pickle.load(fp)

with open ('vec_texts', 'rb') as fp:
    vec_texts = pickle.load(fp)
    
```

Here, I am Loading vocab_limit and embd_limit.
Vocab_limit contains only the words that are present in the dataset, along
some special words representing markers for EOS, PAD etc.

The network should output the probability distribution over the words in 
vocab_limit. So using limited vocabulary (vocab_limit) will mean requiring
less parameters for calculating the probability distribution as compared to using the complete vocabulary from gloVe dataset.


```python
with open ('vocab_limit', 'rb') as fp:
    vocab_limit = pickle.load(fp)

with open ('embd_limit', 'rb') as fp:
    embd_limit = pickle.load(fp)
    
```

Including SOS (signifies 'start of sentence'. It will be used as the initial input token for the decoder) and its vector to the vocbulary list and the embedding matrix. I forgot to do this while pre-processing.


```python
vocab_limit.append('<SOS>')
embd_limit.append(np.zeros((word_vec_dim),dtype=np.float32))

SOS = embd_limit[vocab_limit.index('<SOS>')]

np_embd_limit = np.asarray(embd_limit,dtype=np.float32)

```

### Removing data samples whose summaries are too long

I will not be training the model in mini-batches. I will train the model one sample at a time, because I fear my old laptop
will probably not be able to handle batch training.

Reducing data with high summary lengths will entail less maximum decoder timestep. This step is mainly taken to keep the training light. 

### Removing data samples who text (review) length is less than the window size

In this model I will try to implement <b>local attention</b> with the seq2seq architecture.

Where global attention looks at all the hidden states of the encoder to determine where to attend to,
local attention looks only at the hidden states under the range pt-D to pt+D where D is empirically selected
and pt is a position determined by the program.

The range of pt-D to pt+D can be said to be the window where attention takes place.  Pt is the center of the
window.

I am treating D as a hyperparameter. The window size will be (pt-D)-(pt+D)+1 = 2D+1.

Now, the window needs to be smaller than or equal to the no. of the encoded hidden states themselves.
We will encode one hidden state for each words in the input text, so size of the hidden states will be equivalent
to the size of the input text.

So we must choose D such that 2D+1 is not bigger than the length of any text in the dataset.

To ensure that, I will first diagnose how many data will be removed for a given D, and in the next jupyter cell,
I will remove all input texts whose length is less than 2D+1.

### Removing data samples whose text (review) is too long 

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

Creating train, validation and test batches.


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

This transformation will be like one hot encoding, but somewhat different. 

Instead of being precisely one hot encoded the output will be transformed
such that it will contain the list of indices which would have been 'one' if it was one hot encoded.

Each word in vocab_limit will be considered as different classes here. 


```python
def transform_out(output_text):
    output_len = len(output_text)
    transformed_output = np.zeros([output_len],dtype=np.int32)
    for i in xrange(0,output_len):
        transformed_output[i] = vocab_limit.index(vec2word(output_text[i]))
    return transformed_output   
```

### Hyperparameters

Here I am simply setting up some of the rest of the hyperparameters.
K, here, is a special hyperparameter. It denotes the no. of previous hidden states
to consider for residual connections (elaborated later on)


```python
#Some MORE hyperparameters and other stuffs

hidden_size = 500
learning_rate = 0.003
K = 5
vocab_len = len(vocab_limit)
training_iters = 5 
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

### FORWARD AND BACKWARD LSTM WITH RRA

I will be using the encoder-decoder architecture.
For the encoder I will be using a bi-directional LSTM.
Below is the function of the forward encoder (the LSTM in the forward direction
that starts from the first word and encodes a word in the context of previous words),
and then for the backward encoder (the LSTM in the backward direction
that starts from the last word and encodes a word in the context of later words)

The RNN used here, is a standard LSTM with RRA (Recurrent Residual Attention)

([RRA: Recurrent Residual Attention for Sequence Learning - Cheng Wang 	arXiv:1709.03714 [cs.LG]](https://arxiv.org/abs/1709.03714))

The model will compute the weighted sum (weighted based on some trainable parameters
in the attention weight matrix) of the PREVIOUS K (K is the hyperparameter mentioned before) hidden states - the weighted sum is denoted as RRA in this function.

The last K indices of hidden_residuals will contain the last K hidden states.

The RRA will influence the Hidden State calculation in LSTM.

(The attention weight matrix is to be normalized by dividing each elements by the sum of all 
the elements as said in the paper. But, here, I am normalizing it by softmax)

The purpose for this is to create connections between hidden states of different timesteps,
to establish long term dependencies.

**UPDATE:** A LSTMN may better serve the purpose of RRA. 

See: [Long Short-Term Memory-Networks for Machine Reading by Jianpeng Cheng, Li Dong and Mirella Lapata](https://arxiv.org/pdf/1601.06733.pdf)

I had similar ideas here: https://github.com/JRC1995/INTER-INTRA-attentions


**NOTE:** To implement RRA, I am using a dynamic tensorarray (kind of like a list but for tensorflow).
The tensorarray (named hidden_residuals here) will contain ALL the past hidden states. New hidden states will be dynamically added to this tensorarray. 
Initially the list start with K hidden_state shaped tensors with zeros in their last axis. Zeros, since initially there won't be any past hidden state except one zero-initialized one. After the initial timestep, new hidden state will be appended to the tensorarray, and from the next timestep, the last K indices of the tensorarray will be summed together to create the RRA. 


```python
def forward_encoder(inp,hidden,cell,
                    wf,uf,bf,
                    wi,ui,bi,
                    wo,uo,bo,
                    wc,uc,bc,
                    Wattention,seq_len,inp_dim):

    Wattention = tf.nn.softmax(Wattention,0)
    hidden_forward = tf.TensorArray(size=seq_len,dtype=tf.float32)
    
    hidden_residuals = tf.TensorArray(size=K,dynamic_size=True,dtype=tf.float32,clear_after_read=False)
    hidden_residuals = hidden_residuals.unstack(tf.zeros([K,hidden_size],dtype=tf.float32))
    
    i=0
    j=K
    
    def cond(i,j,hidden,cell,hidden_forward,hidden_residuals):
        return i < seq_len
    
    def body(i,j,hidden,cell,hidden_forward,hidden_residuals):
        
        x = tf.reshape(inp[i],[1,inp_dim])
        
        hidden_residuals_stack = hidden_residuals.stack()
        
        RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-K:j],Wattention),0)
        RRA = tf.reshape(RRA,[1,hidden_size])
        
        # LSTM with RRA
        fg = tf.sigmoid( tf.matmul(x,wf) + tf.matmul(hidden,uf) + bf)
        ig = tf.sigmoid( tf.matmul(x,wi) + tf.matmul(hidden,ui) + bi)
        og = tf.sigmoid( tf.matmul(x,wo) + tf.matmul(hidden,uo) + bo)
        cell = tf.multiply(fg,cell) + tf.multiply(ig,tf.sigmoid( tf.matmul(x,wc) + tf.matmul(hidden,uc) + bc))
        hidden = tf.multiply(og,tf.tanh(cell+RRA))
        
        hidden_residuals = tf.cond(tf.equal(j,seq_len-1+K),
                                   lambda: hidden_residuals,
                                   lambda: hidden_residuals.write(j,tf.reshape(hidden,[hidden_size])))

        hidden_forward = hidden_forward.write(i,tf.reshape(hidden,[hidden_size]))
        
        return i+1,j+1,hidden,cell,hidden_forward,hidden_residuals
    
    _,_,_,_,hidden_forward,hidden_residuals = tf.while_loop(cond,body,[i,j,hidden,cell,hidden_forward,hidden_residuals])
    
    hidden_residuals.close().mark_used()
    
    return hidden_forward.stack()
        
```


```python
def backward_encoder(inp,hidden,cell,
                     wf,uf,bf,
                     wi,ui,bi,
                     wo,uo,bo,
                     wc,uc,bc,
                     Wattention,seq_len,inp_dim):
    
    Wattention = tf.nn.softmax(Wattention,0)
    hidden_backward = tf.TensorArray(size=seq_len,dtype=tf.float32)
    
    hidden_residuals = tf.TensorArray(size=K,dynamic_size=True,dtype=tf.float32,clear_after_read=False)
    hidden_residuals = hidden_residuals.unstack(tf.zeros([K,hidden_size],dtype=tf.float32))
    
    i=seq_len-1
    j=K
    
    def cond(i,j,hidden,cell,hidden_backward,hidden_residuals):
        return i > -1
    
    def body(i,j,hidden,cell,hidden_backward,hidden_residuals):
        
        x = tf.reshape(inp[i],[1,inp_dim])
        
        hidden_residuals_stack = hidden_residuals.stack()
        
        RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-K:j],Wattention),0)
        RRA = tf.reshape(RRA,[1,hidden_size])
        
        # LSTM with RRA
        fg = tf.sigmoid( tf.matmul(x,wf) + tf.matmul(hidden,uf) + bf)
        ig = tf.sigmoid( tf.matmul(x,wi) + tf.matmul(hidden,ui) + bi)
        og = tf.sigmoid( tf.matmul(x,wo) + tf.matmul(hidden,uo) + bo)
        cell = tf.multiply(fg,cell) + tf.multiply(ig,tf.sigmoid( tf.matmul(x,wc) + tf.matmul(hidden,uc) + bc))
        hidden = tf.multiply(og,tf.tanh(cell+RRA))

        hidden_residuals = tf.cond(tf.equal(j,seq_len-1+K),
                                   lambda: hidden_residuals,
                                   lambda: hidden_residuals.write(j,tf.reshape(hidden,[hidden_size])))
        
        hidden_backward = hidden_backward.write(i,tf.reshape(hidden,[hidden_size]))
        
        return i-1,j+1,hidden,cell,hidden_backward,hidden_residuals
    
    _,_,_,_,hidden_backward,hidden_residuals = tf.while_loop(cond,body,[i,j,hidden,cell,hidden_backward,hidden_residuals])

    hidden_residuals.close().mark_used()
    
    return hidden_backward.stack()
        
```

The decoder similarly uses LSTM with RRA


```python
def decoder(x,hidden,cell,
            wf,uf,bf,
            wi,ui,bi,
            wo,uo,bo,
            wc,uc,bc,RRA):
    
    # LSTM with RRA
    fg = tf.sigmoid( tf.matmul(x,wf) + tf.matmul(hidden,uf) + bf)
    ig = tf.sigmoid( tf.matmul(x,wi) + tf.matmul(hidden,ui) + bi)
    og = tf.sigmoid( tf.matmul(x,wo) + tf.matmul(hidden,uo) + bo)
    cell_next = tf.multiply(fg,cell) + tf.multiply(ig,tf.sigmoid( tf.matmul(x,wc) + tf.matmul(hidden,uc) + bc))
    hidden_next = tf.multiply(og,tf.tanh(cell+RRA))
    
    return hidden_next,cell_next
```

### LOCAL ATTENTION:

The cell below includes some major functions for the attention mechanism.

The attention mechanism is usually implemented to compute an attention score 
for each of the encoded hidden state in the context of a particular
decoder hidden state in each timestep - all to determine which encoded hidden
states to attend to, given the context of a particular decoder hidden state.

More specifically, I am here implementing local attention as opposed to global attention.

Local attention mechanism involves focusing on a subset of encoded hidden states, whereas a gloabl attention mechanism invovles focusing on all
the encoded hidden states.

This is the paper on which this implementation is based on:

[Effective Approaches to Attention-based Neural Machine Translation
Minh-Thang Luong Hieu Pham Christopher D. Manning](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)
    
Following the formulas presented in the paper, first, I am computing
the position pt (the center of the window of attention).

pt is simply a position in the sequence.
For a given pt, the model will only consider the hidden state starting from the position
pt-D to the hidden state at the position pt+D. 

To say a hidden state is at position pt, I mean to say that the hidden state is the encoded
representation of a word at position pt in the sequence.

The paper formulates the equation for calculating pt like this:
pt = sequence_length x sigmoid(..some linear algebras and activations...)

But, I didn't used the sequence_length of the whole text which is tf_seq_len but 'positions' which
is = tf_seq_len-1-2D

if pt = tf_seq_len x sigmoid(tensor)

Then pt will be in the range 0 to tf_seq_len

But, there is no 'tf_seq_len' position, since the length is tf_seq_len,
the available positions are 0 to (tf_seq_len-1). This is why I subtracted 1 from it.

Next, we must have the value of pt such that it represents the CENTER of the window of size 2D+1.

So the window begin at pt-D and end at pt+D, if pt has to be the center. 

However, if pt is too close to 0, pt-D will be negative - a non-existent position.

And, If pt is too close to tf_seq_len, pt+D will become a non-existent position beyond the maximum sequence length.

So pt can't occupy the first D positions (0 to D-1) and it can't occupy the last D positions
((tf_seq_len-D) to (tf_seq_len-1)), if pt-D and pt+D has to be considered as legal positions.
So a total 2D positions should be restricted to pt.

Which is why I further subtracted 2D from tf_seq_len.

Still, after calculating pt = positions x sigmoid(tensor)
where positions = tf_seq_len-1-2D or tf_seq_len-(2D+1), 
pt will merely range between 0 to tf_seq_len-(2D+1)

We can't still accept pt to be 0 since pt-D will be negative. But the length of the range 
of integer positions pt can occupy is now accurate.

So at this point, we can simply center pt at the window by adding a D.

After that, pt will range from D to (tf_seq_len-1)-D

Now, it can be checked that pt-D will never become negative, and pt+D will never exceed
the total sequence length.

After calculating pt, we can use the formulas presented in the paper to calculate
G (as written in the code below) constititues the attention or compatibility scores, i.e the weights (or attention) that should be given to a hidden state.

G is calculated for each of hidden states in the local window. This is equivalent to
the function 'a(s)' used in the paper.

The function returns the attention weights in G and the position pt, so that the model can create the 
context vector. 



```python
def score(hs,ht,Wa,seq_len):
    return tf.reshape(tf.matmul(tf.matmul(hs,Wa),tf.transpose(ht)),[seq_len])

def align(hs,ht,Wp,Vp,Wa,tf_seq_len):
   
    pd = tf.TensorArray(size=(2*D+1),dtype=tf.float32)
    
    positions = tf.cast(tf_seq_len-1-2*D,dtype=tf.float32)
    
    sigmoid_multiplier = tf.nn.sigmoid(tf.matmul(tf.tanh(tf.matmul(ht,Wp)),Vp))
    sigmoid_multiplier = tf.reshape(sigmoid_multiplier,[])
    
    pt_float = positions*sigmoid_multiplier
    
    pt = tf.cast(pt_float,tf.int32)
    pt = pt+D #center to window
    
    sigma = tf.constant(D/2,dtype=tf.float32)
    
    i = 0
    pos = pt - D
    
    def cond(i,pos,pd):
        
        return i < (2*D+1)
                      
    def body(i,pos,pd):
        
        comp_1 = tf.cast(tf.square(pos-pt),tf.float32)
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

### MODEL DEFINITION

First is the <b>bi-directional encoder</b>.

h_forward is the tensorarray of all the hidden states from the 
forward encoder whereas h_backward is the tensorarray of all the hidden states
from the backward encoder.

The final list of encoder hidden states are usually calculated by combining 
the equivalents of h_forward and h_backward by some means.

There are many means of combining them, like: concatenation, summation, average etc.
    
I will be using concatenation.

hidden_encoder is the final list of encoded hidden states.

The first decoder input is the word vector representation of <SOS> which siginfies the start of decoding.

I am using the first encoded_hidden_state 
as the initial decoder hidden state. The first encoded_hidden_state may have the least 
past context (none actually) but, it will have the most future context.

The next decoder hidden state is generated from the initial decoder input and the initial decoder state.
Next, I start a loop which iterates for output_len times. 

Next the <b>attention function</b> is called, to compute the G score by scoring the encoder hidden states
in term of current decoder hidden step.

The context vector is created by the weighted (weighted in terms of G scores) summation
of hidden states in the local attention window.

I used the formulas mentioned here: https://nlp.stanford.edu/pubs/emnlp15_attn.pdf

to calculate the probability distribution for the first output token from the context vector and decoder hidden state.

The word vector represention of the output token - the word with maximum the predicted probability in the recently calculated probability distribution, is used as the decoder input token. The output decoder hidden state from that current decoder input token, and the hidden state, is used again in the next loop to calculate the probability distribution of the next output token and so on. 

('beam search' is another strategy that can be added to the model, but for simplicity's sake I am avoiding it.)

The loop continues for 'output_len' no. of iterations. 

Since I will be training sample to sample, I can dynamically send the output length 
of the current sample, and the decoder loops for the given 'output length' (the value will be stored in 'output_len placeholder) times.

NOTE: I am saving only the (non-softmaxed) probability distributions for prediction. Tensorflow cost function will internally apply softmax. 


```python
def model(tf_text,tf_seq_len,tf_output_len):
    
    #PARAMETERS
    
    #1.1 FORWARD ENCODER PARAMETERS
    
    initial_hidden_f = tf.zeros([1,hidden_size],dtype=tf.float32)
    cell_f = tf.zeros([1,hidden_size],dtype=tf.float32)
    wf_f = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    uf_f = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bf_f = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    wi_f = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    ui_f = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bi_f = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    wo_f = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    uo_f = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bo_f = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    wc_f = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    uc_f = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bc_f = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    Wattention_f = tf.Variable(tf.zeros([K,1]),dtype=tf.float32)
                               
    #1.2 BACKWARD ENCODER PARAMETERS
    
    initial_hidden_b = tf.zeros([1,hidden_size],dtype=tf.float32)
    cell_b = tf.zeros([1,hidden_size],dtype=tf.float32)
    wf_b = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    uf_b = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bf_b = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    wi_b = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    ui_b = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bi_b = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    wo_b = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    uo_b = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bo_b = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    wc_b = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,hidden_size],stddev=0.01))
    uc_b = tf.Variable(np.eye(hidden_size),dtype=tf.float32)
    bc_b = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
    Wattention_b = tf.Variable(tf.zeros([K,1]),dtype=tf.float32)
    
    #2 ATTENTION PARAMETERS
    
    Wp = tf.Variable(tf.truncated_normal(shape=[2*hidden_size,50],stddev=0.01))
    Vp = tf.Variable(tf.truncated_normal(shape=[50,1],stddev=0.01))
    Wa = tf.Variable(tf.truncated_normal(shape=[2*hidden_size,2*hidden_size],stddev=0.01))
    Wc = tf.Variable(tf.truncated_normal(shape=[4*hidden_size,2*hidden_size],stddev=0.01))
    
    #3 DECODER PARAMETERS
    
    Ws = tf.Variable(tf.truncated_normal(shape=[2*hidden_size,vocab_len],stddev=0.01))
    
    cell_d = tf.zeros([1,2*hidden_size],dtype=tf.float32)
    wf_d = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,2*hidden_size],stddev=0.01))
    uf_d = tf.Variable(np.eye(2*hidden_size),dtype=tf.float32)
    bf_d = tf.Variable(tf.zeros([1,2*hidden_size]),dtype=tf.float32)
    wi_d = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,2*hidden_size],stddev=0.01))
    ui_d = tf.Variable(np.eye(2*hidden_size),dtype=tf.float32)
    bi_d = tf.Variable(tf.zeros([1,2*hidden_size]),dtype=tf.float32)
    wo_d = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,2*hidden_size],stddev=0.01))
    uo_d = tf.Variable(np.eye(2*hidden_size),dtype=tf.float32)
    bo_d = tf.Variable(tf.zeros([1,2*hidden_size]),dtype=tf.float32)
    wc_d = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,2*hidden_size],stddev=0.01))
    uc_d = tf.Variable(np.eye(2*hidden_size),dtype=tf.float32)
    bc_d = tf.Variable(tf.zeros([1,2*hidden_size]),dtype=tf.float32)
    
    hidden_residuals_d = tf.TensorArray(size=K,dynamic_size=True,dtype=tf.float32,clear_after_read=False)
    hidden_residuals_d = hidden_residuals_d.unstack(tf.zeros([K,2*hidden_size],dtype=tf.float32))
    
    Wattention_d = tf.Variable(tf.zeros([K,1]),dtype=tf.float32)
    
    output = tf.TensorArray(size=tf_output_len,dtype=tf.float32)
                               
    #BI-DIRECTIONAL LSTM
                               
    hidden_forward = forward_encoder(tf_text,
                                     initial_hidden_f,cell_f,
                                     wf_f,uf_f,bf_f,
                                     wi_f,ui_f,bi_f,
                                     wo_f,uo_f,bo_f,
                                     wc_f,uc_f,bc_f,
                                     Wattention_f,
                                     tf_seq_len,
                                     word_vec_dim)
    
    hidden_backward = backward_encoder(tf_text,
                                     initial_hidden_b,cell_b,
                                     wf_b,uf_b,bf_b,
                                     wi_b,ui_b,bi_b,
                                     wo_b,uo_b,bo_b,
                                     wc_b,uc_b,bc_b,
                                     Wattention_b,
                                     tf_seq_len,
                                     word_vec_dim)
    
    encoded_hidden = tf.concat([hidden_forward,hidden_backward],1)
    
    #ATTENTION MECHANISM AND DECODER
    
    decoded_hidden = encoded_hidden[0]
    decoded_hidden = tf.reshape(decoded_hidden,[1,2*hidden_size])
    Wattention_d_normalized = tf.nn.softmax(Wattention_d)
    tf_embd_limit = tf.convert_to_tensor(np_embd_limit)
    
    y = tf.convert_to_tensor(SOS) #inital decoder token <SOS> vector
    y = tf.reshape(y,[1,word_vec_dim])
    
    j=K
    
    hidden_residuals_stack = hidden_residuals_d.stack()
    
    RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-K:j],Wattention_d_normalized),0)
    RRA = tf.reshape(RRA,[1,2*hidden_size])
    
    decoded_hidden_next,cell_d = decoder(y,decoded_hidden,cell_d,
                                  wf_d,uf_d,bf_d,
                                  wi_d,ui_d,bf_d,
                                  wo_d,uo_d,bf_d,
                                  wc_d,uc_d,bc_d,
                                  RRA)
    decoded_hidden = decoded_hidden_next
    
    hidden_residuals_d = hidden_residuals_d.write(j,tf.reshape(decoded_hidden,[2*hidden_size]))
    
    j=j+1
                           
    i=0
    
    def attention_decoder_cond(i,j,decoded_hidden,cell_d,hidden_residuals_d,output):
        return i < tf_output_len
    
    def attention_decoder_body(i,j,decoded_hidden,cell_d,hidden_residuals_d,output):
        
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
        #Save probability distribution as output
        
        y = tf.nn.softmax(y)
        
        y_index = tf.cast(tf.argmax(tf.reshape(y,[vocab_len])),tf.int32)
        y = tf_embd_limit[y_index]
        y = tf.reshape(y,[1,word_vec_dim])
        
        #setting next decoder input token as the word_vector of maximum probability 
        #as found from previous attention-decoder output.
        
        hidden_residuals_stack = hidden_residuals_d.stack()
        
        RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-K:j],Wattention_d_normalized),0)
        RRA = tf.reshape(RRA,[1,2*hidden_size])
        
        decoded_hidden_next,cell_d = decoder(y,decoded_hidden,cell_d,
                                  wf_d,uf_d,bf_d,
                                  wi_d,ui_d,bf_d,
                                  wo_d,uo_d,bf_d,
                                  wc_d,uc_d,bc_d,
                                  RRA)
        
        decoded_hidden = decoded_hidden_next
        
        hidden_residuals_d = tf.cond(tf.equal(j,tf_output_len-1+K+1), #(+1 for <SOS>)
                                   lambda: hidden_residuals_d,
                                   lambda: hidden_residuals_d.write(j,tf.reshape(decoded_hidden,[2*hidden_size])))
        
        return i+1,j+1,decoded_hidden,cell_d,hidden_residuals_d,output
    
    i,j,decoded_hidden,cell_d,hidden_residuals_d,output = tf.while_loop(attention_decoder_cond,
                                            attention_decoder_body,
                                            [i,j,decoded_hidden,cell_d,hidden_residuals_d,output])
    hidden_residuals_d.close().mark_used()
    
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

### TRAINING

Finally, this is where training takes place.
It's all pretty self explanatory, but one thing to note is that
I am sending "train_summaries[i][0:len(train_summaries[i])-1]"
to the transform_out() function. That is, I am ignoring the last
word from summary. The last word marks the end of the summary.
It's 'eos'. 

As I said before, I will run it for only a few early iterations.
So, it's not likely to see any great predicted summaries here.
As can be seen, the summaries seem more influenced by previous 
output sample than the input context in these early iterations.

Some of the texts contains undesirable words like br tags and so
on. So better preprocessing and tokenization may be desirable.

With more layer depth, larger hidden size, mini-batch training,
and other changes, this model may have potential.

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
                print("loss="+str(loss))
            
        step=step+1
    
```

    
    Iteration: 0
    Training input sequence length: 51
    Training target outputs sequence length: 4
    
    TEXT:
    i have bought several of the vitality canned dog food products and have found them all to be of good quality. the product looks more like a stew than a processed meat and it smells better. my labrador is finicky and she appreciates this product better than most.
    
    
    PREDICTED SUMMARY:
    
    swipe swipe weiner weiner
    
    ACTUAL SUMMARY:
    
    good quality dog food
    
    loss=10.3719
    
    Iteration: 1
    Training input sequence length: 37
    Training target outputs sequence length: 3
    
    TEXT:
    product arrived labeled as jumbo salted peanuts ... the peanuts were actually small sized unsalted. not sure if this was an error or if the vendor intended to represent the product as `` jumbo ''.
    
    
    PREDICTED SUMMARY:
    
    good food food
    
    ACTUAL SUMMARY:
    
    not as advertised
    
    loss=10.5117
    
    Iteration: 2
    Training input sequence length: 46
    Training target outputs sequence length: 2
    
    TEXT:
    if you are looking for the secret ingredient in robitussin i believe i have found it. i got this in addition to the root beer extract i ordered( which was good) and made some cherry soda. the flavor is very medicinal.
    
    
    PREDICTED SUMMARY:
    
    quality food
    
    ACTUAL SUMMARY:
    
    cough medicine
    
    loss=10.497
    
    Iteration: 3
    Training input sequence length: 32
    Training target outputs sequence length: 2
    
    TEXT:
    great taffy at a great price. there was a wide assortment of yummy taffy. delivery was very quick. if your a taffy lover, this is a deal.
    
    
    PREDICTED SUMMARY:
    
    not advertised
    
    ACTUAL SUMMARY:
    
    great taffy
    
    loss=10.4166
    
    Iteration: 4
    Training input sequence length: 30
    Training target outputs sequence length: 4
    
    TEXT:
    this taffy is so good. it is very soft and chewy. the flavors are amazing. i would definitely recommend you buying it. very satisfying!!
    
    
    PREDICTED SUMMARY:
    
    not advertised advertised advertised
    
    ACTUAL SUMMARY:
    
    wonderful, tasty taffy
    
    loss=10.62
    
    Iteration: 5
    Training input sequence length: 29
    Training target outputs sequence length: 2
    
    TEXT:
    right now i 'm mostly just sprouting this so my cats can eat the grass. they love it. i rotate it around with wheatgrass and rye too
    
    
    PREDICTED SUMMARY:
    
    not advertised
    
    ACTUAL SUMMARY:
    
    yay barley
    
    loss=12.5289
    
    Iteration: 6
    Training input sequence length: 29
    Training target outputs sequence length: 3
    
    TEXT:
    this is a very healthy dog food. good for their digestion. also good for small puppies. my dog eats her required amount at every feeding.
    
    
    PREDICTED SUMMARY:
    
    not advertised advertised
    
    ACTUAL SUMMARY:
    
    healthy dog food
    
    loss=7.77178
    
    Iteration: 7
    Training input sequence length: 24
    Training target outputs sequence length: 4
    
    TEXT:
    the strawberry twizzlers are my guilty pleasure- yummy. six pounds will be around for a while with my son and i.
    
    
    PREDICTED SUMMARY:
    
    cough taffy taffy taffy
    
    ACTUAL SUMMARY:
    
    strawberry twizzlers- yummy
    
    loss=13.8934
    
    Iteration: 8
    Training input sequence length: 45
    Training target outputs sequence length: 2
    
    TEXT:
    i love eating them and they are good for watching tv and looking at movies! it is not too sweet. i like to transfer them to a zip lock baggie so they stay fresh so i can take my time eating them.
    
    
    PREDICTED SUMMARY:
    
    cough taffy
    
    ACTUAL SUMMARY:
    
    poor taste
    
    loss=13.8444
    
    Iteration: 9
    Training input sequence length: 28
    Training target outputs sequence length: 3
    
    TEXT:
    i am very satisfied with my unk purchase. i shared these with others and we have all enjoyed them. i will definitely be ordering more.
    
    
    PREDICTED SUMMARY:
    
    cough taffy food
    
    ACTUAL SUMMARY:
    
    love it!
    
    loss=13.4374
    
    Iteration: 10
    Training input sequence length: 31
    Training target outputs sequence length: 3
    
    TEXT:
    candy was delivered very fast and was purchased at a reasonable price. i was home bound and unable to get to a store so this was perfect for me.
    
    
    PREDICTED SUMMARY:
    
    great taffy food
    
    ACTUAL SUMMARY:
    
    home delivered unk
    
    loss=12.8076
    
    Iteration: 11
    Training input sequence length: 52
    Training target outputs sequence length: 2
    
    TEXT:
    my husband is a twizzlers addict. we 've bought these many times from amazon because we 're government employees living overseas and ca n't get them in the country we are assigned to. they 've always been fresh and tasty, packed well and arrive in a timely manner.
    
    
    PREDICTED SUMMARY:
    
    great great
    
    ACTUAL SUMMARY:
    
    always fresh
    
    loss=10.7377
    
    Iteration: 12
    Training input sequence length: 68
    Training target outputs sequence length: 1
    
    TEXT:
    i bought these for my husband who is currently overseas. he loves these, and apparently his staff likes them unk< br/> there are generous amounts of twizzlers in each 16-ounce bag, and this was well worth the price.< a unk '' http: unk ''> twizzlers, strawberry, 16-ounce bags( pack of 6)< unk>
    
    
    PREDICTED SUMMARY:
    
    yay
    
    ACTUAL SUMMARY:
    
    twizzlers
    
    loss=6.44585
    
    Iteration: 13
    Training input sequence length: 31
    Training target outputs sequence length: 3
    
    TEXT:
    i can remember buying this candy as a kid and the quality has n't dropped in all these years. still a superb product you wo n't be disappointed with.
    
    
    PREDICTED SUMMARY:
    
    yay barley,
    
    ACTUAL SUMMARY:
    
    delicious product!
    
    loss=9.29425
    
    Iteration: 14
    Training input sequence length: 21
    Training target outputs sequence length: 1
    
    TEXT:
    i love this candy. after weight watchers i had to cut back but still have a craving for it.
    
    
    PREDICTED SUMMARY:
    
    twizzlers
    
    ACTUAL SUMMARY:
    
    twizzlers
    
    loss=3.23818
    
    Iteration: 15
    Training input sequence length: 72
    Training target outputs sequence length: 7
    
    TEXT:
    i have lived out of the us for over 7 yrs now

### To Try\ To Do\ To keep in mind: (It's more like a note to self)

* Beam Search
* Pointer Mechanisms
* Heirarchical attention
* Try some of these tricks: http://ruder.io/deep-learning-nlp-best-practices/index.html#hyperparameteroptimization
* [Intra-input-attention](https://arxiv.org/pdf/1705.04304.pdf)
* Better pre-processing
* Switch to PyTorch or DyNet or something more suitable for dynamic models. 
* Mini-Batch Training
* Better Datasets.
* Train for different tasks (eg. Translation) using different datasets.
* Intra-layer attention for both encoder and decoder together with everything else.
* Adopt a more object oriented approach
* Regularization
* Validation
* Testing
* Implement Evaluation Metrics (ROUGE\BLEU\Something else)


