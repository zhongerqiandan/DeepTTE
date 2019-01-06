import tensorflow as tf
import numpy as np

T = 30
K = 3
driverId_embed_size = 10
weather_embed_size = 10
time_embed_size = 10
hidden_size = 128
driverId_size = 100
weather_size = 10
time_size = 10


def STLComponent(inputs,sequence_length=T,recipe_length=K,distance=None,attr=None,conv_output_channel=32,padding='VALID',lstm_hidden_size=hidden_size):
    inputs = tf.expand_dims(inputs,axis=-1)
    def Geo_Conv_layer(inputs=inputs,recipe_length=recipe_length,channel=conv_output_channel,distance=distance):
        d = int(inputs.get_shape()[-2])
        kernel_size=[recipe_length,d]
        outputs = tf.contrib.layers.conv2d(inputs=inputs,num_outputs=conv_output_channel,kernel_size=kernel_size,padding=padding)
        if distance == None:
            return tf.squeeze(outputs)
    
    
    def concat_layer(inputs=None,attr=attr):
        if attr == None:
            return inputs
        else:
            d_1 = int(inputs.get_shape()[-2])
            d_2 = int(attr.get_shape()[-2])
            assert d_1 == d_2
            return tf.concat([inputs,attr],axis=-1)
    
    def single_cell_fn(num_units):

        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units,forget_bias=1.0,layer_norm=True)

        return single_cell



    def cell_list(num_units,num_layers):
        
        cell_list = []
        for i in range(num_layers):

            single_cell = single_cell_fn(num_units=num_units)
            cell_list.append(single_cell)

        return cell_list

    def create_rnn_cell(num_units,num_layers):
        cell_list_ = cell_list(num_units=num_units,num_layers=num_layers)
        if len(cell_list_) == 1:  # Single layer.
            return cell_list_[0]
        else:  # Multi layers
            return tf.contrib.rnn.MultiRNNCell(cell_list_)
    def multi_lstm_layer(inputs,hidden_size=lstm_hidden_size,num_layers=2):
        cell = create_rnn_cell(hidden_size,num_layers)
        outputs,state = tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32)
        return outputs
    geo_conv_layer_outputs = Geo_Conv_layer(inputs,K,conv_output_channel)
    concat_layer_outputs = concat_layer(geo_conv_layer_outputs,attr)
    multi_lstm_layer_outputs = multi_lstm_layer(concat_layer_outputs,lstm_hidden_size)
    return multi_lstm_layer_outputs


def AttributeComponent(driverId,weather,time,dis=None,driverId_size=driverId_size,weather_size=weather_size,time_size=time_size,driverId_embed_size=driverId_embed_size,weather_embed_size=weather_embed_size,time_embed_size=time_embed_size):
    driverId_embeddings = tf.Variable(tf.truncated_normal([driverId_size,driverId_embed_size]))
    weather_embeddings = tf.Variable(tf.truncated_normal([weather_size,weather_embed_size]))
    time_embeddings = tf.Variable(tf.truncated_normal([time_size,time_embed_size]))
    d_embed = tf.nn.embedding_lookup(driverId_embeddings,driverId)
    w_embed = tf.nn.embedding_lookup(weather_embeddings,weather)
    t_embed = tf.nn.embedding_lookup(time_embeddings,time)
    if dis == None:
        return tf.concat([d_embed,w_embed,t_embed],axis=-1)
    else:
        return tf.concat([d_embed,w_embed,t_embed,dis],axis=-1)

def Attention(hidden,attr):
    d_hidden = int(hidden.get_shape()[-1])
    d_attr = int(attr.get_shape()[-1])
    theta = tf.Variable(tf.truncated_normal([d_attr,d_hidden]))
    theta_attr = tf.nn.sigmoid(tf.matmul(attr,theta))
    theta_attr = tf.expand_dims(theta_attr,-1)
    z = tf.matmul(hidden,theta_attr)
    d = int(z.get_shape()[-2])
    z = tf.reshape(z,[-1,1,d])
    alfa = tf.nn.softmax(z)
    h_att = tf.squeeze(tf.matmul(alfa,hidden))
    return h_att

        

