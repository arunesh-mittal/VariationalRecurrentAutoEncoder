import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
#from jupyterthemes import jtplot
#from IPython.core.debugger import Tracer
#jtplot.style()


################################
# Params
################################
# X \in {0,1}^{batch_size, dim_x, time_steps}
time_steps = 10 #TEST
x_in_dim = 88
z_dim = 20 #TEST
num_hidden_units = 500
batch_size = 2#100 #TEST
#learning_rate = 5*1e-6
beta_1 = 0.05
beta_2 = 0.001
num_epochs = 400
starter_learning_rate = 1e-4
decay_rate = .6

load_root = './MIDI_Data_PianoRolls/Nottingham/train/'
samples = create_samples(load_root, time_steps=time_steps, verbose=False)
num_train_samples = 10 #TEST #samples.shape[0] #TEST only pick 20 samples and overfit


################################
# Helper functions
################################
def clip_roll(piano_roll, time_steps=50):
    samples = []
    num_samples = int(piano_roll.shape[1] / time_steps)
    for i in range(num_samples):
        start_idx = time_steps*i
        end_idx = (time_steps*(i+1))
        samples.append(piano_roll[:,start_idx:end_idx])
    return samples   

def create_samples(load_root, time_steps=50, verbose=False):
    samples = []
    for (dirpath, dirnames, filenames) in os.walk(load_root):
        for file in filenames:
            if file.endswith('.npy'):
                load_filepath = os.path.join(dirpath,file)
                if verbose:
                    print(load_filepath)
                piano_roll = np.load(load_filepath).T
                samples = samples + clip_roll(piano_roll,time_steps=time_steps)

    return np.stack(samples)

def feed_dict(batch_size):
    indeces = np.random.randint(num_train_samples, size=batch_size)
    return {X: np.take(samples, indeces, axis=0)}


##################################
# Setup tf Graph
##################################

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(batch_size, x_in_dim, time_steps))

# time_slices containts input x at time t across batches.
x_in = time_steps * [None]
x_out = time_steps * [None]
h_enc = time_steps * [None]
h_dec = (time_steps + 1) * [None]

for t in range(time_steps):
    x_in[t] = tf.squeeze(tf.slice(X,begin=[0,0,t],size=[-1,-1,1]),axis=2)

###### Encoder network ###########
with tf.variable_scope('encoder_rnn'):
    cell_enc = tf.nn.rnn_cell.BasicRNNCell(num_hidden_units,activation=tf.nn.tanh)
    h_enc[0] = tf.zeros([batch_size,num_hidden_units], dtype=tf.float32) # Initial state is 0

    # h_t+1 = tanh(Wenc*h_t + Win*x_t+1 + b )
    #Most basic RNN: output = new_state = act(W * input + U * state + B).
    #https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn_cell_impl.py
    for t in range(time_steps-1):
        _ , h_enc[t+1] = cell_enc(inputs=x_in[t+1], state=h_enc[t])


mu_enc = tf.layers.dense(h_enc[-1], z_dim, activation=None, name='mu_enc')
log_sigma_enc = tf.layers.dense(h_enc[-1], z_dim, activation=None, name='log_sigma_enc')

###### Reparametrize ##############
eps = tf.random_normal(tf.shape(log_sigma_enc))
z = mu_enc + tf.exp(log_sigma_enc) * eps

##### Decoder network ############
with tf.variable_scope('decoder_rnn'):
    W_out = tf.get_variable('W_out',shape=[num_hidden_units, x_in_dim])
    b_out = tf.get_variable('b_out',shape=[x_in_dim])
    
    cell_dec = tf.nn.rnn_cell.BasicRNNCell(num_hidden_units,activation=tf.nn.tanh)
    h_dec[0] = tf.layers.dense(z, num_hidden_units, activation=tf.nn.tanh)
    
    for t in range(time_steps):
        x_out[t] = tf.nn.sigmoid(tf.matmul(h_dec[t], W_out) + b_out)
        if t < time_steps - 1:
            _, h_dec[t+1] = cell_dec(inputs=x_out[t], state=h_dec[t])

##### Loss #####################
with tf.variable_scope('loss'):
    # Latent loss: -KL[q(z|x)|p(z)]
    with tf.variable_scope('latent_loss'):
        sigma_sq_enc = tf.square(tf.exp(log_sigma_enc))
        latent_loss = -.5 * tf.reduce_mean(tf.reduce_sum((1 + tf.log(1e-10 + sigma_sq_enc)) - tf.square(mu_enc) - sigma_sq_enc, axis=1),axis=0)
        latent_loss_summ = tf.summary.scalar('latent_loss',latent_loss)
        
    # Reconstruction Loss: log(p(x|z))    
    with tf.variable_scope('recon_loss'):    
        for i in range(time_steps):
            if i == 0:
                recon_loss_ = x_in[i] * tf.log(1e-10 + x_out[i]) + (1 - x_in[i]) * tf.log(1e-10+1-x_out[i])
            else:
                recon_loss_ += x_in[i] * tf.log(1e-10 + x_out[i]) + (1 - x_in[i]) * tf.log(1e-10+1-x_out[i])
            
        #collapse the loss, mean across a sample across all x_dim and time points, mean over batches
        recon_loss = -tf.reduce_mean(tf.reduce_mean(recon_loss_/(time_steps),axis=1),axis=0)

            
    recon_loss_summ = tf.summary.scalar('recon_loss', recon_loss)
                
    with tf.variable_scope('total_loss'):
        total_loss = latent_loss + recon_loss
    
    total_loss_summ = tf.summary.scalar('total_loss', total_loss)

global_step = tf.Variable(0,name='global_step') 
epoch_num = tf.Variable(1, name='epoch_num', trainable=False, dtype=tf.int32)
increment_epoch_num_op = tf.assign(epoch_num, epoch_num+1)


learning_rate = tf.train.exponential_decay(starter_learning_rate, epoch_num, num_epochs, decay_rate, staircase=False)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_1, beta2=beta_2).minimize(total_loss,global_step=global_step)    
scalar_summaries = tf.summary.merge([latent_loss_summ, recon_loss_summ, total_loss_summ])
#image_summaries = tf.summary.merge() 

train_summary_writer = tf.summary.FileWriter('./logs', tf.get_default_graph())


##################################
# Training
##################################
num_batches = int(num_train_samples/batch_size)
global_step_op = tf.train.get_global_step()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        epoch_loss = 0.
        epoch_latent_loss = 0.
        for batch in range(num_batches):
            batch_num = sess.run(global_step_op)
            _ , loss, scalar_summaries_, x_out_, x_in_,learning_rate_,latent_loss_ = sess.run([train_step, total_loss, scalar_summaries, x_out, x_in,learning_rate, latent_loss],feed_dict=feed_dict(batch_size))
            train_summary_writer.add_summary(scalar_summaries_, global_step=batch_num)
            epoch_loss += loss
            epoch_latent_loss += latent_loss_
            
            sigma_sq_enc_ = sess.run(sigma_sq_enc, feed_dict=feed_dict(batch_size))
            #Tracer()()
            
            
            #print('Epoch Loss: {}'.format(loss))
        print('Average loss epoch {0}: {1}'.format(epoch, epoch_loss/num_batches)) 
        print('Average latent loss epoch {0}: {1}'.format(epoch, epoch_latent_loss/num_batches)) 
        print('Learning Rate {}'.format(learning_rate_))
        sess.run(increment_epoch_num_op)

