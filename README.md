
# Abstractive Summarization

### Loading Pre-processed Dataset

The Data is preprocessed in [Data_Pre-Processing.ipynb](https://github.com/JRC1995/Abstractive-Summarization/blob/master/Data%20Pre-processing.ipynb)

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
max_summary_len = 16
D = 5 # D determines local attention window size
window_len = 2*D+1
l2=1e-6
```

## Tensorflow Placeholders


```python
import tensorflow as tf 

embd_dim = len(embd[0])

tf_text = tf.placeholder(tf.int32, [None, None])
tf_embd = tf.placeholder(tf.float32, [len(vocab2idx),embd_dim])
tf_true_summary_len = tf.placeholder(tf.int32, [None])
tf_summary = tf.placeholder(tf.int32,[None, None])
tf_train = tf.placeholder(tf.bool)
```

## Embed vectorized text

Dropout used for regularization 
(https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)


```python
embd_text = tf.nn.embedding_lookup(tf_embd, tf_text)
embd_text = tf.layers.dropout(embd_text,rate=0.3,training=tf_train)
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
hidden_backward_rev = tf.reverse(hidden_backward,axis=[1])

encoder_states = tf.concat([hidden_forward,hidden_backward_rev],axis=-1)
encoder_states = tf.transpose(encoder_states,[1,0,2])

encoder_states = tf.layers.dropout(encoder_states,rate=0.3,training=tf_train)

final_encoded_state = tf.layers.dropout(tf.concat([hidden_forward[-1],hidden_backward[-1]],axis=-1),rate=0.3,training=tf_train)

```

## Implementation of attention scoring function

Given a sequence of encoder states (H_s) and the decoder hidden state (H_t) of current timestep t, the equation for computing attention score is:

Score = (H_s.W_a).Transpose(H_t)

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
        Wp = tf.get_variable("Wp", shape=[2*hidden_size,125],
                                    dtype=tf.float32,
                                    trainable=True,
                                    initializer=tf.glorot_uniform_initializer())
        
        Vp = tf.get_variable("Vp", shape=[125,1],
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

for i in range(max_summary_len):
    
    inp = tf.layers.dropout(inp,rate=0.3,training=tf_train)
    
    attention_scores = align(encoder_states,hidden)
    encoder_context_vector = tf.reduce_sum(encoder_states*attention_scores,axis=1)
    
    hidden,cell = LSTM(inp,hidden,cell,embd_dim,2*hidden_size,scope="decoder")
    
    hidden_ = tf.layers.dropout(hidden,rate=0.3,training=tf_train)
    
    concated = tf.concat([hidden_,encoder_context_vector],axis=-1)
    
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

optimizer = tf.contrib.opt.NadamOptimizer(
    learning_rate=learning_rate)

gvs = optimizer.compute_gradients(cost, var_list=all_vars)

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

        train_saver.restore(sess, 'Model_Backup/Seq2seq_summarization.ckpt')
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

    
    Load checkpoint? y/n: n
    
    
    
    STARTING TRAINING
    
    
    Iter 0, Cost= 2.086, Acc = 0.00%
    
    Sample Text
    
    i 'm not a big pretzel eater , but i love this little <UNK> nibblers . i like the low fat snack and how it fills you up .
    
    Sample Predicted Summary
    
    municipality jackass municipality mongolian seats han han mongolian hah sus sus wat hah casbah dynasty province 
    
    Sample Actual Summary
    
    great pretzels 
    
    
    Iter 100, Cost= 0.985, Acc = 35.58%
    Iter 200, Cost= 0.914, Acc = 33.33%
    Iter 300, Cost= 0.928, Acc = 36.11%
    Iter 400, Cost= 0.943, Acc = 35.19%
    Iter 500, Cost= 0.676, Acc = 42.71%
    
    Sample Text
    
    we <UNK> this one , but the flavor could have been a tad stronger . very yummy tho , we will totally purchase again !
    
    Sample Predicted Summary
    
    delicious ! 
    
    Sample Actual Summary
    
    very good ! 
    
    
    Iter 600, Cost= 0.878, Acc = 35.24%
    Iter 700, Cost= 0.949, Acc = 33.04%
    Iter 800, Cost= 1.074, Acc = 34.65%
    Iter 900, Cost= 0.831, Acc = 44.21%
    Iter 1000, Cost= 0.911, Acc = 36.36%
    
    Sample Text
    
    tried this hoping for something better than the thick salsa that everyone else makes and it was great ! after making our own it gets time consuming so this is a good alternative .
    
    Sample Predicted Summary
    
    great 
    
    Sample Actual Summary
    
    great salsa 
    
    
    Iter 1100, Cost= 1.081, Acc = 23.33%
    Iter 1200, Cost= 1.018, Acc = 32.73%
    Iter 1300, Cost= 0.902, Acc = 35.87%
    Iter 1400, Cost= 0.946, Acc = 31.07%
    Iter 1500, Cost= 0.798, Acc = 42.31%
    
    Sample Text
    
    i had a coupon for this so it was a good value . otherwise it is to expense for what you get . my box had a couple of opened cereals in it so i did n't get the full value of all ...
    
    Sample Predicted Summary
    
    good 
    
    Sample Actual Summary
    
    good value 
    
    
    Iter 1600, Cost= 0.871, Acc = 33.33%
    Iter 1700, Cost= 0.943, Acc = 40.00%
    Iter 1800, Cost= 0.876, Acc = 40.20%
    Iter 1900, Cost= 0.973, Acc = 37.25%
    Iter 2000, Cost= 0.978, Acc = 29.73%
    
    Sample Text
    
    my 4 dogs all had allergies and are just fine now that i switched to <UNK> the <UNK> one smell abit but <UNK> they still love it <UNK> the dried <UNK> canned r terrific <UNK> nooo grani !
    
    Sample Predicted Summary
    
    <UNK> ! 
    
    Sample Actual Summary
    
    great food 
    
    
    Iter 2100, Cost= 0.907, Acc = 37.04%
    Iter 2200, Cost= 0.928, Acc = 34.31%
    Iter 2300, Cost= 0.906, Acc = 31.25%
    Iter 2400, Cost= 0.903, Acc = 37.00%
    Iter 2500, Cost= 0.811, Acc = 33.01%
    
    Sample Text
    
    the chocolate was a little crumbly , but the taste is very good . my hubby has <UNK> , and it is gluten free , so it is an excellent bar to stock in the pantry for whenever he does n't have time for breakfast .
    
    Sample Predicted Summary
    
    great 
    
    Sample Actual Summary
    
    yum 
    
    
    Iter 2600, Cost= 0.839, Acc = 34.62%
    Iter 2700, Cost= 0.927, Acc = 37.07%
    Iter 2800, Cost= 0.853, Acc = 36.73%
    Iter 2900, Cost= 0.805, Acc = 40.00%
    Iter 3000, Cost= 0.855, Acc = 35.51%
    
    Sample Text
    
    tea came packaged as expected , delivered quickly and with stash you can not go wrong . individually wrapped and stays fresh and very flavorful . highly recommended for the earl gray tea lover .
    
    Sample Predicted Summary
    
    delicious tea 
    
    Sample Actual Summary
    
    great tea 
    
    
    Iter 3100, Cost= 0.854, Acc = 36.63%
    
    
    STARTING VALIDATION
    
    
    Validating data # 0
    Validating data # 100
    Validating data # 200
    Validating data # 300
    
    
    Epoch: 0
    
    
    Average Training Loss: 0.907
    Average Training Accuracy: 35.42
    Average Validation Loss: 0.865
    Average Validation Accuracy: 36.65
    
    Model saved
    
    
    
    STARTING TRAINING
    
    
    Iter 0, Cost= 0.808, Acc = 34.34%
    
    Sample Text
    
    quaker oatmeal squares has been our family favorite for a couple of years now . ca n't get enough of it . just the right sweetness and crunch .
    
    Sample Predicted Summary
    
    great 
    
    Sample Actual Summary
    
    favorite cereal 
    
    
    Iter 100, Cost= 1.036, Acc = 34.26%
    Iter 200, Cost= 0.934, Acc = 33.03%
    Iter 300, Cost= 0.972, Acc = 35.85%
    Iter 400, Cost= 0.926, Acc = 32.35%
    Iter 500, Cost= 0.738, Acc = 41.05%
    
    Sample Text
    
    great taste , nice smell , great <UNK> < br / > if you mix it with fresh ment you will get fantastic <UNK> < br / > i will buy it again .
    
    Sample Predicted Summary
    
    great 
    
    Sample Actual Summary
    
    the best 
    
    
    Iter 600, Cost= 0.858, Acc = 41.24%
    Iter 700, Cost= 0.905, Acc = 36.45%
    Iter 800, Cost= 0.795, Acc = 35.05%
    Iter 900, Cost= 0.806, Acc = 37.50%
    Iter 1000, Cost= 0.795, Acc = 35.64%
    
    Sample Text
    
    i bought about 5 different kinds of <UNK> when i first got my coffee maker , which i love by the way , and i 'd have to say that this was my favorite one out of them all . it has the perfect balance of everything , i was really surprised .
    
    Sample Predicted Summary
    
    great 
    
    Sample Actual Summary
    
    excellent stuff 
    
    
    Iter 1100, Cost= 0.825, Acc = 39.42%
    Iter 1200, Cost= 0.743, Acc = 38.78%
    Iter 1300, Cost= 0.813, Acc = 41.84%
    Iter 1400, Cost= 0.933, Acc = 29.66%
    Iter 1500, Cost= 0.978, Acc = 33.61%
    
    Sample Text
    
    i really wanted to like this , as it was organic , and came in a glass bottle , but there was hardly any flavor at all . i could barely smell it , and even when i poured a generous amount on my dish , it imparts little to no truffle <UNK> . my truffle salt is much more potent .
    
    Sample Predicted Summary
    
    good 
    
    Sample Actual Summary
    
    weak 
    
    
    Iter 1600, Cost= 0.778, Acc = 45.10%
    Iter 1700, Cost= 0.855, Acc = 38.83%
    Iter 1800, Cost= 0.815, Acc = 41.58%
    Iter 1900, Cost= 0.853, Acc = 37.62%
    Iter 2000, Cost= 1.003, Acc = 32.74%
    
    Sample Text
    
    i love milk chocolate and do n't like dark <UNK> . my husband is the opposite , so i always buy him the dark stuff and it 's safe for him , haha ! until i happened to try this one . it 's awesome !
    
    Sample Predicted Summary
    
    <UNK> ! 
    
    Sample Actual Summary
    
    it 's good ! ! 
    
    
    Iter 2100, Cost= 0.817, Acc = 37.74%
    Iter 2200, Cost= 0.977, Acc = 33.33%
    Iter 2300, Cost= 0.840, Acc = 35.96%
    Iter 2400, Cost= 0.749, Acc = 31.58%
    Iter 2500, Cost= 0.885, Acc = 31.73%
    
    Sample Text
    
    the best thing about this coffee is the sweet smell , just like a blueberry muffin . the taste is good , not as sweet as i was expecting but it was good nonetheless . its a nice treat when you 're craving something sweet but it wo n't replace my morning donut shop coffee : )
    
    Sample Predicted Summary
    
    delicious 
    
    Sample Actual Summary
    
    smells yummy : ) 
    
    
    Iter 2600, Cost= 0.887, Acc = 32.73%
    Iter 2700, Cost= 0.780, Acc = 44.94%
    Iter 2800, Cost= 0.899, Acc = 35.71%
    Iter 2900, Cost= 0.797, Acc = 38.24%
    Iter 3000, Cost= 1.061, Acc = 33.33%
    
    Sample Text
    
    this tea is wonderful , one bag will make three cups for most people . i like my tea very strong so these were perfect . i bet they will be good for making a good ice tea .
    
    Sample Predicted Summary
    
    great tea 
    
    Sample Actual Summary
    
    one bag 3 cups 
    
    
    Iter 3100, Cost= 0.769, Acc = 37.86%
    
    
    STARTING VALIDATION
    
    
    Validating data # 0
    Validating data # 100
    Validating data # 200
    Validating data # 300
    
    
    Epoch: 1
    
    
    Average Training Loss: 0.863
    Average Training Accuracy: 36.40
    Average Validation Loss: 0.837
    Average Validation Accuracy: 37.30
    
    Model saved
    
    
    
    STARTING TRAINING
    
    
    Iter 0, Cost= 0.959, Acc = 35.85%
    
    Sample Text
    
    really good bars . you could cut this baby in 1/2 and have 2 snacks out of it ! i bought 1 at the store first to see if i liked them and paid lots more for it . i do n't eat alot of meat so this caught my eye . i now have them on auto delivery ! !
    
    Sample Predicted Summary
    
    great ! 
    
    Sample Actual Summary
    
    great bars ! 
    
    
    Iter 100, Cost= 0.792, Acc = 33.33%
    Iter 200, Cost= 0.781, Acc = 35.29%
    Iter 300, Cost= 0.825, Acc = 40.74%
    Iter 400, Cost= 0.793, Acc = 40.19%
    Iter 500, Cost= 0.860, Acc = 31.07%
    
    Sample Text
    
    i always buy my coffee from amazon as the prices are cheaper and i love all the coffee . best price on line .
    
    Sample Predicted Summary
    
    great coffee 
    
    Sample Actual Summary
    
    peggy 
    
    
    Iter 600, Cost= 0.990, Acc = 28.57%
    Iter 700, Cost= 0.736, Acc = 41.41%
    Iter 800, Cost= 0.826, Acc = 33.68%
    Iter 900, Cost= 0.904, Acc = 35.24%
    Iter 1000, Cost= 0.858, Acc = 35.71%
    
    Sample Text
    
    i am very pleased with this product and the company sent it on a timely basis , well packed to prevent breakage .
    
    Sample Predicted Summary
    
    great 
    
    Sample Actual Summary
    
    good stuff 
    
    
    Iter 1100, Cost= 0.999, Acc = 30.36%
    Iter 1200, Cost= 0.726, Acc = 44.79%
    Iter 1300, Cost= 0.798, Acc = 36.73%
    Iter 1400, Cost= 0.712, Acc = 48.98%
    Iter 1500, Cost= 0.884, Acc = 38.18%
    
    Sample Text
    
    i love this cereal , easy to eat out of the box . sweet but not too sweet and very crunchy . since frosted mini wheats have gelatin in them they are not an option for vegans or vegetarians , these are the best replacement .
    
    Sample Predicted Summary
    
    <UNK> 
    
    Sample Actual Summary
    
    love it . 
    
    
    Iter 1600, Cost= 0.819, Acc = 44.12%
    Iter 1700, Cost= 0.958, Acc = 32.11%
    Iter 1800, Cost= 0.800, Acc = 37.96%
    Iter 1900, Cost= 0.649, Acc = 40.82%
    Iter 2000, Cost= 0.900, Acc = 36.21%
    
    Sample Text
    
    i have been buying these bars ( without chocolate ) for years and have recently found them with the chocolate . our family of six loves them . they are a great bar to hold you over and give you extended fuel , made with great ingredients to boot . i highly recommend you try a box !
    
    Sample Predicted Summary
    
    great ! 
    
    Sample Actual Summary
    
    <UNK> ! 
    
    
    Iter 2100, Cost= 0.767, Acc = 36.19%
    Iter 2200, Cost= 0.676, Acc = 37.62%
    Iter 2300, Cost= 0.871, Acc = 40.00%
    Iter 2400, Cost= 0.687, Acc = 39.60%
    Iter 2500, Cost= 0.898, Acc = 36.27%
    
    Sample Text
    
    this is top notch almond syrup . we put it in lemonade and strawberries . great for many <UNK> < br / > also use in baking recipes .
    
    Sample Predicted Summary
    
    great 
    
    Sample Actual Summary
    
    soooo yummy 
    
    
    Iter 2600, Cost= 0.796, Acc = 40.74%
    Iter 2700, Cost= 0.775, Acc = 46.00%
    Iter 2800, Cost= 0.856, Acc = 40.19%
    Iter 2900, Cost= 0.954, Acc = 35.85%
    Iter 3000, Cost= 0.831, Acc = 35.40%
    
    Sample Text
    
    this is very good coffee at a good price ... it is an old product that has been on the market since i was quite young .
    
    Sample Predicted Summary
    
    coffee 
    
    Sample Actual Summary
    
    good stuff ! 
    
    
    Iter 3100, Cost= 0.754, Acc = 36.27%
    
    
    STARTING VALIDATION
    
    
    Validating data # 0
    Validating data # 100
    Validating data # 200
    Validating data # 300
    
    
    Epoch: 2
    
    
    Average Training Loss: 0.840
    Average Training Accuracy: 37.26
    Average Validation Loss: 0.818
    Average Validation Accuracy: 38.42
    
    Model saved
    
    
    
    STARTING TRAINING
    
    
    Iter 0, Cost= 0.822, Acc = 36.36%
    
    Sample Text
    
    the chocolate covered figs were delicious and presented beautifully in the package . great for a gift for someone who has everything .
    
    Sample Predicted Summary
    
    delicious 
    
    Sample Actual Summary
    
    figs 
    
    
    Iter 100, Cost= 0.734, Acc = 37.86%
    Iter 200, Cost= 0.837, Acc = 41.18%
    Iter 300, Cost= 0.717, Acc = 34.91%
    Iter 400, Cost= 0.797, Acc = 38.61%
    Iter 500, Cost= 0.718, Acc = 32.38%
    
    Sample Text
    
    one of my favorite flavors of <UNK> , it used to be called twisted tornado , now called fruit twist either way i ca n't stop myself from eating its so flavorful = )
    
    Sample Predicted Summary
    
    great ! 
    
    Sample Actual Summary
    
    yum 
    
    
    Iter 600, Cost= 0.846, Acc = 40.59%
    Iter 700, Cost= 0.676, Acc = 43.75%
    Iter 800, Cost= 0.882, Acc = 39.22%
    Iter 900, Cost= 0.803, Acc = 36.54%
    Iter 1000, Cost= 0.718, Acc = 40.40%
    
    Sample Text
    
    i found this product to be a nice tasting pepper blend and would recommend it to all of those who enjoy the fresh flavor of ground pepper .
    
    Sample Predicted Summary
    
    good taste 
    
    Sample Actual Summary
    
    peppercorn mix 
    
    
    Iter 1100, Cost= 0.749, Acc = 41.24%
    Iter 1200, Cost= 0.821, Acc = 38.10%
    Iter 1300, Cost= 0.883, Acc = 39.81%
    Iter 1400, Cost= 0.961, Acc = 29.91%
    Iter 1500, Cost= 1.130, Acc = 33.96%
    
    Sample Text
    
    bought the popper about two years ago and have been enjoying the delicious fresh buttery salty ( as i want ) best popcorn ever . love it and it 's a staple snack in our house . would never <UNK> corn again .
    
    Sample Predicted Summary
    
    great 
    
    Sample Actual Summary
    
    top notch 
    
    
    Iter 1600, Cost= 0.855, Acc = 35.24%
    Iter 1700, Cost= 0.701, Acc = 38.61%
    Iter 1800, Cost= 0.865, Acc = 35.64%
    Iter 1900, Cost= 0.868, Acc = 39.62%
    Iter 2000, Cost= 0.849, Acc = 40.78%
    
    Sample Text
    
    i love sour stuff . this is n't too sour but still gets the job done . good chewy candy . arrived faster than expected too .
    
    Sample Predicted Summary
    
    good 
    
    Sample Actual Summary
    
    mmmmm 
    
    
    Iter 2100, Cost= 0.951, Acc = 32.73%
    Iter 2200, Cost= 0.875, Acc = 31.68%
    Iter 2300, Cost= 0.866, Acc = 42.20%
    Iter 2400, Cost= 0.725, Acc = 46.32%
    Iter 2500, Cost= 0.793, Acc = 35.71%
    
    Sample Text
    
    i had not tried this tea before but i was hoping it was similar to one i tried while in england . i was not disappointed . the pack of 6 makes it a very good value as well .
    
    Sample Predicted Summary
    
    tea tea 
    
    Sample Actual Summary
    
    love this tea ! 
    
    
    Iter 2600, Cost= 0.864, Acc = 34.82%
    Iter 2700, Cost= 0.853, Acc = 38.10%
    Iter 2800, Cost= 0.694, Acc = 40.40%
    Iter 2900, Cost= 1.020, Acc = 34.26%
    Iter 3000, Cost= 0.782, Acc = 43.00%
    
    Sample Text
    
    extremely disappointing . frankly , i think plain old lipton tea is smoother and less bitter . when brewed , i could hardly recognize it as green tea . it tasted more like a very poor earl gray .
    
    Sample Predicted Summary
    
    sad 
    
    Sample Actual Summary
    
    not good at all 
    
    
    Iter 3100, Cost= 0.756, Acc = 35.64%
    
    
    STARTING VALIDATION
    
    
    Validating data # 0
    Validating data # 100
    Validating data # 200
    Validating data # 300
    
    
    Epoch: 3
    
    
    Average Training Loss: 0.820
    Average Training Accuracy: 38.18
    Average Validation Loss: 0.801
    Average Validation Accuracy: 39.24
    
    Model saved
    
    
    
    STARTING TRAINING
    
    
    Iter 0, Cost= 0.821, Acc = 39.00%
    
    Sample Text
    
    love this tea . i do not like the plain sleepytime but adding the vanilla is a great move ! highly recommend it . looking forward to trying the honey sleepy time !
    
    Sample Predicted Summary
    
    love ! 
    
    Sample Actual Summary
    
    love it 
    
    
    Iter 100, Cost= 0.725, Acc = 37.00%
    Iter 200, Cost= 0.805, Acc = 39.29%
    Iter 300, Cost= 0.838, Acc = 41.23%
    Iter 400, Cost= 0.713, Acc = 49.07%
    Iter 500, Cost= 0.722, Acc = 37.86%
    
    Sample Text
    
    the product arrived quickly . all bags and chips were in place ... and safe ; <UNK> these chips are delicious and only four ww points !
    
    Sample Predicted Summary
    
    delicious 
    
    Sample Actual Summary
    
    yum ! 
    
    
    Iter 600, Cost= 0.941, Acc = 34.82%
    Iter 700, Cost= 0.678, Acc = 42.00%
    Iter 800, Cost= 0.607, Acc = 47.47%
    Iter 900, Cost= 0.679, Acc = 41.94%
    Iter 1000, Cost= 0.763, Acc = 48.60%
    
    Sample Text
    
    this is a light to medium roast , wish it was slightly stronger , but the flavor is good and i am having it every morning using 2 6 oz . <UNK> pumps to make it as strong as possible .
    
    Sample Predicted Summary
    
    very good 
    
    Sample Actual Summary
    
    i like it ! 
    
    
    Iter 1100, Cost= 0.671, Acc = 44.44%
    Iter 1200, Cost= 0.810, Acc = 39.81%
    Iter 1300, Cost= 0.899, Acc = 31.78%
    Iter 1400, Cost= 0.865, Acc = 39.42%
    Iter 1500, Cost= 0.809, Acc = 36.54%
    
    Sample Text
    
    i expected a little more flavor as i usually like green mountain <UNK> < br / > next time i 'll look for a french roast !
    
    Sample Predicted Summary
    
    good 
    
    Sample Actual Summary
    
    too weak 
    
    
    Iter 1600, Cost= 0.873, Acc = 39.45%
    Iter 1700, Cost= 0.882, Acc = 38.14%
    Iter 1800, Cost= 0.953, Acc = 34.86%
    Iter 1900, Cost= 0.961, Acc = 33.66%
    Iter 2000, Cost= 0.774, Acc = 35.92%
    
    Sample Text
    
    i use this sauce on pork ribs , after baking them at 300 degrees for 3 hours . the sweet taste of honey along with the tomato is heavenly .
    
    Sample Predicted Summary
    
    great sauce 
    
    Sample Actual Summary
    
    the best 
    
    
    Iter 2100, Cost= 0.744, Acc = 39.13%
    Iter 2200, Cost= 0.697, Acc = 41.58%
    Iter 2300, Cost= 0.869, Acc = 34.26%
    Iter 2400, Cost= 0.867, Acc = 31.48%
    Iter 2500, Cost= 0.784, Acc = 38.14%
    
    Sample Text
    
    excellent < a <UNK> '' http : <UNK> '' > kellogg 's cereal in a cup , favorite assortment pack , 1.5 - <UNK> <UNK> cups ( pack of 60 ) < <UNK> >
    
    Sample Predicted Summary
    
    good 
    
    Sample Actual Summary
    
    kelloggs 
    
    
    Iter 2600, Cost= 0.653, Acc = 45.45%
    Iter 2700, Cost= 0.713, Acc = 46.73%
    Iter 2800, Cost= 0.777, Acc = 39.05%
    Iter 2900, Cost= 0.795, Acc = 38.10%
    Iter 3000, Cost= 0.802, Acc = 41.12%
    
    Sample Text
    
    this is a good product . the honey tastes great , and it 's very convenient and <UNK> . my local <UNK> store was trying to sell this to me for twice the price as amazon , so i 'm pretty sure this is a good buy .
    
    Sample Predicted Summary
    
    great 
    
    Sample Actual Summary
    
    honey ! 
    
    
    Iter 3100, Cost= 0.773, Acc = 45.54%
    
    
    STARTING VALIDATION
    
    
    Validating data # 0
    Validating data # 100
    Validating data # 200
    Validating data # 300
    
    
    Epoch: 4
    
    
    Average Training Loss: 0.804
    Average Training Accuracy: 39.03
    Average Validation Loss: 0.786
    Average Validation Accuracy: 40.62
    
    Model saved
    


### Future Works

* Beam Search
* Pointer Mechanisms
* BLEU\ROUGE evaluation
* Implement Testing
* Complete Training and Optimize Hyperparameters
