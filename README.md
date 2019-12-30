# Abstractive Summarization

Based on [Seq2seq learning](https://arxiv.org/abs/1409.3215)
with [attention mechanism](https://arxiv.org/abs/1409.0473), specifically [local attention](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf).

### Loading Pre-processed Dataset

The Data is preprocessed in [Data_Pre-Processing.ipynb](https://github.com/JRC1995/Abstractive-Summarization/blob/master/Data_Pre-Processing.ipynb)

Dataset source: https://www.kaggle.com/snap/amazon-fine-food-reviews


```python
import json

with open('Processed_Data/Amazon_Reviews_Processed.json') as file:

    for json_data in file:
        saved_data = json.loads(json_data)

        vocab2idx = saved_data["vocab"]
        embd = saved_data["embd"]
        train_batches_text = saved_data["train_batches_text"]
        test_batches_text = saved_data["test_batches_text"]
        val_batches_text = saved_data["val_batches_text"]
        train_batches_summary = saved_data["train_batches_summary"]
        test_batches_summary = saved_data["test_batches_summary"]
        val_batches_summary = saved_data["val_batches_summary"]
        train_batches_true_text_len = saved_data["train_batches_true_text_len"]
        val_batches_true_text_len = saved_data["val_batches_true_text_len"]
        test_batches_true_text_len = saved_data["test_batches_true_text_len"]
        train_batches_true_summary_len = saved_data["train_batches_true_summary_len"]
        val_batches_true_summary_len = saved_data["val_batches_true_summary_len"]
        test_batches_true_summary_len = saved_data["test_batches_true_summary_len"]

        break
        
idx2vocab = {v:k for k,v in vocab2idx.items()}
```

## Hyperparameters


```python
hidden_size = 300
learning_rate = 0.001
epochs = 5
max_summary_len = 31 # should be summary_max_len as used in data_preprocessing with +1 (+1 for <EOS>) 
D = 5 # D determines local attention window size
window_len = 2*D+1
l2=1e-6
```

## Tensorflow Placeholders


```python
import tensorflow.compat.v1 as tf 

tf.disable_v2_behavior()
tf.disable_eager_execution()

embd_dim = len(embd[0])

tf_text = tf.placeholder(tf.int32, [None, None])
tf_embd = tf.placeholder(tf.float32, [len(vocab2idx),embd_dim])
tf_true_summary_len = tf.placeholder(tf.int32, [None])
tf_summary = tf.placeholder(tf.int32,[None, None])
tf_train = tf.placeholder(tf.bool)
```

    WARNING:tensorflow:From /home/jishnu/miniconda3/envs/ML/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term


## Dropout Function


```python
def dropout(x,rate,training):
    return tf.cond(tf_train,
                    lambda: tf.nn.dropout(x,rate=0.3),
                    lambda: x)

    
```

## Embed vectorized text

Dropout used for regularization 
(https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)


```python
embd_text = tf.nn.embedding_lookup(tf_embd, tf_text)

embd_text = dropout(embd_text,rate=0.3,training=tf_train)
```

## LSTM function

More info: 
<br>
https://dl.acm.org/citation.cfm?id=1246450, 
<br>
https://www.bioinf.jku.at/publications/older/2604.pdf,
<br>
https://en.wikipedia.org/wiki/Long_short-term_memory


```python
def LSTM(x,hidden_state,cell,input_dim,hidden_size,scope):
    
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        
        w = tf.get_variable("w", shape=[4,input_dim,hidden_size],
                                    dtype=tf.float32,
                                    trainable=True,
                                    initializer=tf.glorot_uniform_initializer())
        
        u = tf.get_variable("u", shape=[4,hidden_size,hidden_size],
                            dtype=tf.float32,
                            trainable=True,
                            initializer=tf.glorot_uniform_initializer())
        
        b = tf.get_variable("bias", shape=[4,1,hidden_size],
                    dtype=tf.float32,
                    trainable=True,
                    initializer=tf.zeros_initializer())
        
    input_gate = tf.nn.sigmoid( tf.matmul(x,w[0]) + tf.matmul(hidden_state,u[0]) + b[0])
    forget_gate = tf.nn.sigmoid( tf.matmul(x,w[1]) + tf.matmul(hidden_state,u[1]) + b[1])
    output_gate = tf.nn.sigmoid( tf.matmul(x,w[2]) + tf.matmul(hidden_state,u[2]) + b[2])
    cell_ = tf.nn.tanh( tf.matmul(x,w[3]) + tf.matmul(hidden_state,u[3]) + b[3])
    cell = forget_gate*cell + input_gate*cell_
    hidden_state = output_gate*tf.tanh(cell)
    
    return hidden_state, cell
      
```

## Bi-Directional LSTM Encoder

(https://maxwell.ict.griffith.edu.au/spl/publications/papers/ieeesp97_schuster.pdf)

More Info: https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/

Bi-directional LSTM encoder has a forward encoder and a backward encoder. The forward encoder encodes a text sequence from start to end, and the backward encoder encodes the text sequence from end to start.
The final output is a combination (in this case, a concatenation) of the forward encoded text and the backward encoded text
    


## Forward Encoding


```python
S = tf.shape(embd_text)[1] #text sequence length
N = tf.shape(embd_text)[0] #batch_size

i=0
hidden=tf.zeros([N, hidden_size], dtype=tf.float32)
cell=tf.zeros([N, hidden_size], dtype=tf.float32)
hidden_forward=tf.TensorArray(size=S, dtype=tf.float32)

#shape of embd_text: [N,S,embd_dim]
embd_text_t = tf.transpose(embd_text,[1,0,2]) 
#current shape of embd_text: [S,N,embd_dim]

def cond(i, hidden, cell, hidden_forward):
    return i < S

def body(i, hidden, cell, hidden_forward):
    x = embd_text_t[i]
    
    hidden,cell = LSTM(x,hidden,cell,embd_dim,hidden_size,scope="forward_encoder")
    hidden_forward = hidden_forward.write(i, hidden)

    return i+1, hidden, cell, hidden_forward

_, _, _, hidden_forward = tf.while_loop(cond, body, [i, hidden, cell, hidden_forward])
```

## Backward Encoding


```python
i=S-1
hidden=tf.zeros([N, hidden_size], dtype=tf.float32)
cell=tf.zeros([N, hidden_size], dtype=tf.float32)
hidden_backward=tf.TensorArray(size=S, dtype=tf.float32)

def cond(i, hidden, cell, hidden_backward):
    return i >= 0

def body(i, hidden, cell, hidden_backward):
    x = embd_text_t[i]
    hidden,cell = LSTM(x,hidden,cell,embd_dim,hidden_size,scope="backward_encoder")
    hidden_backward = hidden_backward.write(i, hidden)

    return i-1, hidden, cell, hidden_backward

_, _, _, hidden_backward = tf.while_loop(cond, body, [i, hidden, cell, hidden_backward])
```

## Merge Forward and Backward Encoder Hidden States


```python
hidden_forward = hidden_forward.stack()
hidden_backward = hidden_backward.stack()

encoder_states = tf.concat([hidden_forward,hidden_backward],axis=-1)
encoder_states = tf.transpose(encoder_states,[1,0,2])

encoder_states = dropout(encoder_states,rate=0.3,training=tf_train)

final_encoded_state = dropout(tf.concat([hidden_forward[-1],hidden_backward[-1]],axis=-1),rate=0.3,training=tf_train)


```

## Implementation of attention scoring function

Given a sequence of encoder states ($H_s$) and the decoder hidden state ($H_t$) of current timestep $t$, the equation for computing attention score is:

$$Score = (H_s.W_a).H_t^T $$

($W_a$ = trainable parameters)

(https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)


```python
def attention_score(encoder_states,decoder_hidden_state,scope="attention_score"):
    
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        Wa = tf.get_variable("Wa", shape=[2*hidden_size,2*hidden_size],
                                    dtype=tf.float32,
                                    trainable=True,
                                    initializer=tf.glorot_uniform_initializer())
        
    encoder_states = tf.reshape(encoder_states,[N*S,2*hidden_size])
    
    encoder_states = tf.reshape(tf.matmul(encoder_states,Wa),[N,S,2*hidden_size])
    decoder_hidden_state = tf.reshape(decoder_hidden_state,[N,2*hidden_size,1])
    
    return tf.reshape(tf.matmul(encoder_states,decoder_hidden_state),[N,S])

```

## Local Attention Function

Based on: https://nlp.stanford.edu/pubs/emnlp15_attn.pdf


```python

def align(encoder_states, decoder_hidden_state,scope="attention"):
    
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        Wp = tf.get_variable("Wp", shape=[2*hidden_size,128],
                                    dtype=tf.float32,
                                    trainable=True,
                                    initializer=tf.glorot_uniform_initializer())
        
        Vp = tf.get_variable("Vp", shape=[128,1],
                            dtype=tf.float32,
                            trainable=True,
                            initializer=tf.glorot_uniform_initializer())
    
    positions = tf.cast(S-window_len,dtype=tf.float32) # Maximum valid attention window starting position
    
    # Predict attention window starting position 
    ps = positions*tf.nn.sigmoid(tf.matmul(tf.tanh(tf.matmul(decoder_hidden_state,Wp)),Vp))
    # ps = (soft-)predicted starting position of attention window
    pt = ps+D # pt = center of attention window where the whole window length is 2*D+1
    pt = tf.reshape(pt,[N])
    
    i = 0
    gaussian_position_based_scores = tf.TensorArray(size=S,dtype=tf.float32)
    sigma = tf.constant(D/2,dtype=tf.float32)
    
    def cond(i,gaussian_position_based_scores):
        
        return i < S
                      
    def body(i,gaussian_position_based_scores):
        
        score = tf.exp(-((tf.square(tf.cast(i,tf.float32)-pt))/(2*tf.square(sigma)))) 
        # (equation (10) in https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)
        gaussian_position_based_scores = gaussian_position_based_scores.write(i,score)
            
        return i+1,gaussian_position_based_scores
                      
    i,gaussian_position_based_scores = tf.while_loop(cond,body,[i,gaussian_position_based_scores])
    
    gaussian_position_based_scores = gaussian_position_based_scores.stack()
    gaussian_position_based_scores = tf.transpose(gaussian_position_based_scores,[1,0])
    gaussian_position_based_scores = tf.reshape(gaussian_position_based_scores,[N,S])
    
    scores = attention_score(encoder_states,decoder_hidden_state)*gaussian_position_based_scores
    scores = tf.nn.softmax(scores,axis=-1)
    
    return tf.reshape(scores,[N,S,1])
```

## LSTM Decoder With Local Attention


```python
with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
    SOS = tf.get_variable("sos", shape=[1,embd_dim],
                                dtype=tf.float32,
                                trainable=True,
                                initializer=tf.glorot_uniform_initializer())
    
    # SOS represents starting marker 
    # It tells the decoder that it is about to decode the first word of the output
    # I have set SOS as a trainable parameter
    
    Wc = tf.get_variable("Wc", shape=[4*hidden_size,embd_dim],
                            dtype=tf.float32,
                            trainable=True,
                            initializer=tf.glorot_uniform_initializer())
    


SOS = tf.tile(SOS,[N,1]) #now SOS shape: [N,embd_dim]
inp = SOS
hidden=final_encoded_state
cell=tf.zeros([N, 2*hidden_size], dtype=tf.float32)
decoder_outputs=tf.TensorArray(size=max_summary_len, dtype=tf.float32)
outputs=tf.TensorArray(size=max_summary_len, dtype=tf.int32)

attention_scores = align(encoder_states,hidden)
encoder_context_vector = tf.reduce_sum(encoder_states*attention_scores,axis=1)

for i in range(max_summary_len):
    
    inp = dropout(inp,rate=0.3,training=tf_train)
    
    inp = tf.concat([inp,encoder_context_vector],axis=-1)
    
    hidden,cell = LSTM(inp,hidden,cell,embd_dim+2*hidden_size,2*hidden_size,scope="decoder")
    
    hidden = dropout(hidden,rate=0.3,training=tf_train)
    
    attention_scores = align(encoder_states,hidden)
    encoder_context_vector = tf.reduce_sum(encoder_states*attention_scores,axis=1)
    
    concated = tf.concat([hidden,encoder_context_vector],axis=-1)
    
    linear_out = tf.nn.tanh(tf.matmul(concated,Wc))
    decoder_output = tf.matmul(linear_out,tf.transpose(tf_embd,[1,0])) 
    # produce unnormalized probability distribution over vocabulary
    
    
    decoder_outputs = decoder_outputs.write(i,decoder_output)
    
    # Pick out most probable vocab indices based on the unnormalized probability distribution
    
    next_word_vec = tf.cast(tf.argmax(decoder_output,1),tf.int32)

    next_word_vec = tf.reshape(next_word_vec, [N])

    outputs = outputs.write(i,next_word_vec)

    next_word = tf.nn.embedding_lookup(tf_embd, next_word_vec)
    inp = tf.reshape(next_word, [N, embd_dim])
    
    
decoder_outputs = decoder_outputs.stack()
outputs = outputs.stack()

decoder_outputs = tf.transpose(decoder_outputs,[1,0,2])
outputs = tf.transpose(outputs,[1,0])

    
    
```

## Define Cross Entropy Cost Function and L2 Regularization


```python
filtered_trainables = [var for var in tf.trainable_variables() if
                       not("Bias" in var.name or "bias" in var.name
                           or "noreg" in var.name)]

regularization = tf.reduce_sum([tf.nn.l2_loss(var) for var
                                in filtered_trainables])

with tf.variable_scope("loss"):

    epsilon = tf.constant(1e-9, tf.float32)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf_summary, logits=decoder_outputs)

    pad_mask = tf.sequence_mask(tf_true_summary_len,
                                maxlen=max_summary_len,
                                dtype=tf.float32)

    masked_cross_entropy = cross_entropy*pad_mask

    cost = tf.reduce_mean(masked_cross_entropy) + \
        l2*regularization

    cross_entropy = tf.reduce_mean(masked_cross_entropy)
```

## Accuracy


```python
# Comparing predicted sequence with labels
comparison = tf.cast(tf.equal(outputs, tf_summary),
                     tf.float32)

# Masking to ignore the effect of pads while calculating accuracy
pad_mask = tf.sequence_mask(tf_true_summary_len,
                            maxlen=max_summary_len,
                            dtype=tf.bool)

masked_comparison = tf.boolean_mask(comparison, pad_mask)

# Accuracy
accuracy = tf.reduce_mean(masked_comparison)
```

## Define Optimizer


```python
all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

gvs = optimizer.compute_gradients(cost, all_vars)

capped_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs] # Gradient Clipping

train_op = optimizer.apply_gradients(capped_gvs)
```

## Training and Validation


```python
import pickle
import random

with tf.Session() as sess:  # Start Tensorflow Session
    display_step = 100
    patience = 5

    load = input("\nLoad checkpoint? y/n: ")
    print("")
    saver = tf.train.Saver()

    if load.lower() == 'y':

        print('Loading pre-trained weights for the model...')

        saver.restore(sess, 'Model_Backup/Seq2seq_summarization.ckpt')
        sess.run(tf.global_variables())
        sess.run(tf.tables_initializer())

        with open('Model_Backup/Seq2seq_summarization.pkl', 'rb') as fp:
            train_data = pickle.load(fp)

        covered_epochs = train_data['covered_epochs']
        best_loss = train_data['best_loss']
        impatience = 0
        
        print('\nRESTORATION COMPLETE\n')

    else:
        best_loss = 2**30
        impatience = 0
        covered_epochs = 0

        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(tf.tables_initializer())

    epoch=0
    while (epoch+covered_epochs)<epochs:
        
        print("\n\nSTARTING TRAINING\n\n")
        
        batches_indices = [i for i in range(0, len(train_batches_text))]
        random.shuffle(batches_indices)

        total_train_acc = 0
        total_train_loss = 0

        for i in range(0, len(train_batches_text)):
            
            j = int(batches_indices[i])

            cost,prediction,\
                acc, _ = sess.run([cross_entropy,
                                   outputs,
                                   accuracy,
                                   train_op],
                                  feed_dict={tf_text: train_batches_text[j],
                                             tf_embd: embd,
                                             tf_summary: train_batches_summary[j],
                                             tf_true_summary_len: train_batches_true_summary_len[j],
                                             tf_train: True})
            
            total_train_acc += acc
            total_train_loss += cost

            if i % display_step == 0:
                print("Iter "+str(i)+", Cost= " +
                      "{:.3f}".format(cost)+", Acc = " +
                      "{:.2f}%".format(acc*100))
            
            if i % 500 == 0:
                
                idx = random.randint(0,len(train_batches_text[j])-1)
                
                
                
                text = " ".join([idx2vocab.get(vec,"<UNK>") for vec in train_batches_text[j][idx]])
                predicted_summary = [idx2vocab.get(vec,"<UNK>") for vec in prediction[idx]]
                actual_summary = [idx2vocab.get(vec,"<UNK>") for vec in train_batches_summary[j][idx]]
                
                print("\nSample Text\n")
                print(text)
                print("\nSample Predicted Summary\n")
                for word in predicted_summary:
                    if word == '<EOS>':
                        break
                    else:
                        print(word,end=" ")
                print("\n\nSample Actual Summary\n")
                for word in actual_summary:
                    if word == '<EOS>':
                        break
                    else:
                        print(word,end=" ")
                print("\n\n")
                
        print("\n\nSTARTING VALIDATION\n\n")
                
        total_val_loss=0
        total_val_acc=0
                
        for i in range(0, len(val_batches_text)):
            
            if i%100==0:
                print("Validating data # {}".format(i))

            cost, prediction,\
                acc = sess.run([cross_entropy,
                                outputs,
                                accuracy],
                                  feed_dict={tf_text: val_batches_text[i],
                                             tf_embd: embd,
                                             tf_summary: val_batches_summary[i],
                                             tf_true_summary_len: val_batches_true_summary_len[i],
                                             tf_train: False})
            
            total_val_loss += cost
            total_val_acc += acc
            
        avg_val_loss = total_val_loss/len(val_batches_text)
        
        print("\n\nEpoch: {}\n\n".format(epoch+covered_epochs))
        print("Average Training Loss: {:.3f}".format(total_train_loss/len(train_batches_text)))
        print("Average Training Accuracy: {:.2f}".format(100*total_train_acc/len(train_batches_text)))
        print("Average Validation Loss: {:.3f}".format(avg_val_loss))
        print("Average Validation Accuracy: {:.2f}".format(100*total_val_acc/len(val_batches_text)))
              
        if (avg_val_loss < best_loss):
            best_loss = avg_val_loss
            save_data={'best_loss':best_loss,'covered_epochs':covered_epochs+epoch+1}
            impatience=0
            with open('Model_Backup/Seq2seq_summarization.pkl', 'wb') as fp:
                pickle.dump(save_data, fp)
            saver.save(sess, 'Model_Backup/Seq2seq_summarization.ckpt')
            print("\nModel saved\n")
              
        else:
            impatience+=1
              
        if impatience > patience:
              break
              
              
        epoch+=1
            
```

    
    Load checkpoint? y/n:  n


    
    
    
    STARTING TRAINING
    
    
    Iter 0, Cost= 1.493, Acc = 0.00%
    
    Sample Text
    
    i was given these as a gift ... they were so amazing i now order them for all occasions and sometimes just because i had n't had them in a while . a little warning ; they are completely addictive . i like the <UNK> ones ; my girlfriend likes the rocky road . highly recommended ! < br / > < br / > sure to be appreciated by everyone on your gift list .
    
    Sample Predicted Summary
    
    condolence s.e. foodstuff condolence webbed poverty squarely poverty poverty assists foodstuff webbed poverty methodist foodstuff webbed poverty gephardt foodstuff ethier articulos meh rojos cols colombians webbed poverty condolence poverty condolence hourly 
    
    Sample Actual Summary
    
    simply amazing brownies ... 
    
    
    Iter 100, Cost= 0.684, Acc = 26.98%
    Iter 200, Cost= 0.649, Acc = 27.19%
    Iter 300, Cost= 0.744, Acc = 25.93%
    Iter 400, Cost= 0.976, Acc = 19.88%
    Iter 500, Cost= 0.839, Acc = 21.53%
    
    Sample Text
    
    for those looking for a <UNK> water beverage and one with a neutral taste that does n't have <UNK> aftertaste , this one 's for <UNK> < br / > < br / > also , traditional tap water is slightly more acidic ( i believe ph 7-8 ) . <UNK> 's is supposed at 9.5 ph , so if you 're very sensitive to acidic products , this might help you out .
    
    Sample Predicted Summary
    
    good 
    
    Sample Actual Summary
    
    neutral taste , low ph 
    
    
    Iter 600, Cost= 0.697, Acc = 27.82%
    Iter 700, Cost= 0.763, Acc = 24.24%
    Iter 800, Cost= 0.792, Acc = 24.82%
    Iter 900, Cost= 0.866, Acc = 23.13%
    Iter 1000, Cost= 0.838, Acc = 23.03%
    
    Sample Text
    
    i love my starbucks sumatra first thing in the morning . i was not always up early enough to take the detour to starbucks and now i do n't have to ! these <UNK> are perfect and delicious . now i can have my fav coffee even before i take off my slippers ! i love this product ! it 's easy to order - arrived quickly and the price was good .
    
    Sample Predicted Summary
    
    great 
    
    Sample Actual Summary
    
    no drive through at starbucks ? 
    
    
    Iter 1100, Cost= 0.648, Acc = 30.58%
    Iter 1200, Cost= 0.977, Acc = 19.08%
    Iter 1300, Cost= 0.788, Acc = 23.29%
    Iter 1400, Cost= 0.681, Acc = 28.23%
    Iter 1500, Cost= 0.608, Acc = 29.32%
    
    Sample Text
    
    husband loves this tea especially in the <UNK> recommend using the large cup setting on your keurig brewer unless you prefer your tea extra strong .
    
    Sample Predicted Summary
    
    great tea 
    
    Sample Actual Summary
    
    good substitute for coffee . 
    
    
    Iter 1600, Cost= 0.709, Acc = 27.48%
    Iter 1700, Cost= 0.729, Acc = 31.11%
    Iter 1800, Cost= 0.627, Acc = 28.93%
    Iter 1900, Cost= 0.798, Acc = 26.36%
    Iter 2000, Cost= 0.856, Acc = 22.08%
    
    Sample Text
    
    can no longer find this product locally anymore . i purchased it previously at a warehouse club but costco , bj ` s and sam ` s club no longer stock it in my area stores . my two golden retriever ` s love this gravy when added to their mix of both dry and moist dog food . hope it stays on the market ... <UNK> !
    
    Sample Predicted Summary
    
    great 
    
    Sample Actual Summary
    
    best pet food gravy 
    
    
    Iter 2100, Cost= 0.640, Acc = 30.77%
    Iter 2200, Cost= 0.792, Acc = 24.49%
    Iter 2300, Cost= 0.735, Acc = 22.86%
    Iter 2400, Cost= 0.769, Acc = 21.68%
    Iter 2500, Cost= 0.900, Acc = 21.15%
    
    Sample Text
    
    i want to start out by saying that i thought at first that a bag with only 120 calories and 4 grams of fat ( no saturated or trans ) for every 20 chips was going to taste like crap . i must say that not only was i wrong , that this is my favorite bbq chip on the market today . they are light and you can not taste any fat or grease after eating them . that 's because they are n't baked or fried , just popped as their name suggests . these chips are very easy to dip as well . fantastic product !
    
    Sample Predicted Summary
    
    great chips 
    
    Sample Actual Summary
    
    fantastic chips ! ! ! 
    
    
    Iter 2600, Cost= 0.740, Acc = 22.86%
    Iter 2700, Cost= 0.848, Acc = 24.84%
    Iter 2800, Cost= 0.677, Acc = 28.57%
    Iter 2900, Cost= 0.779, Acc = 25.90%
    Iter 3000, Cost= 0.718, Acc = 27.34%
    
    Sample Text
    
    this <UNK> of 7-ounce `` taster 's choice french roast '' canisters , is a good buy . the coffee is flavored differently than original flavor , but the difference is very subtle , and refreshingly good . overall , this taster 's choice coffee is a bargain , and highly recommended .
    
    Sample Predicted Summary
    
    great flavor 
    
    Sample Actual Summary
    
    good buy 


### Future Works

* Beam Search
* Pointer Mechanisms
* BLEU\ROUGE evaluation
* Implement Testing
* Complete Training and Optimize Hyperparameters


```python

```
