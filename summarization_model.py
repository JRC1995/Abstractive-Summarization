
# coding: utf-8

# In[ ]:


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


# In[ ]:


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


# In[ ]:


import pickle

with open ('vec_summaries', 'rb') as fp:
    vec_summaries = pickle.load(fp)

with open ('vec_texts', 'rb') as fp:
    vec_texts = pickle.load(fp)
    


# In[ ]:


with open ('vocab_limit', 'rb') as fp:
    vocab_limit = pickle.load(fp)

with open ('embd_limit', 'rb') as fp:
    embd_limit = pickle.load(fp)
    


# In[ ]:


vocab_limit.append('<SOS>')
embd_limit.append(np.zeros((word_vec_dim),dtype=np.float32))

SOS = embd_limit[vocab_limit.index('<SOS>')]

np_embd_limit = np.asarray(embd_limit,dtype=np.float32)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


train_len = int((.7)*len(vec_summaries_reduced))

train_texts = vec_texts_reduced[0:train_len]
train_summaries = vec_summaries_reduced[0:train_len]

val_len = int((.15)*len(vec_summaries_reduced))

val_texts = vec_texts_reduced[train_len:train_len+val_len]
val_summaries = vec_summaries_reduced[train_len:train_len+val_len]

test_texts = vec_texts_reduced[train_len+val_len:len(vec_summaries_reduced)]
test_summaries = vec_summaries_reduced[train_len+val_len:len(vec_summaries_reduced)]


# In[ ]:


print train_len


# In[ ]:


def transform_out(output_text):
    output_len = len(output_text)
    transformed_output = np.zeros([output_len],dtype=np.int32)
    for i in xrange(0,output_len):
        transformed_output[i] = vocab_limit.index(vec2word(output_text[i]))
    return transformed_output   


# In[ ]:


#Some MORE hyperparameters and other stuffs

hidden_size = 500
learning_rate = 0.003
K = 5
vocab_len = len(vocab_limit)
training_iters = 5 


# In[ ]:


import tensorflow as tf

#placeholders
tf_text = tf.placeholder(tf.float32, [None,word_vec_dim])
tf_seq_len = tf.placeholder(tf.int32)
tf_summary = tf.placeholder(tf.int32,[None])
tf_output_len = tf.placeholder(tf.int32)


# In[ ]:


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
        


# In[ ]:


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
        


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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
    
