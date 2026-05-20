import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GCNLayer(layers.Layer):
    def __init__(self, units, activation=None, use_bias=True, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.l2_reg = l2_reg

    def build(self, input_shape):
        reg = tf.keras.regularizers.l2(self.l2_reg)
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform", regularizer=reg, trainable=True, name="W")
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,), initializer="zeros", trainable=True, name="b")

    def call(self, H, adj):
        support = H @ self.W
        out = adj @ support
        if self.use_bias:
            out = out + self.b
        return self.activation(out) if self.activation else out

class SAGELayer(layers.Layer):
    def __init__(self, units, aggregator="pooling", activation=None, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.aggregator = aggregator
        self.activation = keras.activations.get(activation)
        self.l2_reg = l2_reg

    def build(self, input_shape):
        reg = tf.keras.regularizers.l2(self.l2_reg)
        in_dim = input_shape[-1]
        if self.aggregator == "pooling":
            self.W_pool = self.add_weight(shape=(in_dim, in_dim), initializer="glorot_uniform", regularizer=reg, name="W_pool")
            self.b_pool = self.add_weight(shape=(in_dim,), initializer="zeros", name="b_pool")
        
        self.W = self.add_weight(shape=(in_dim * 2, self.units), initializer="glorot_uniform", regularizer=reg, name="W")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", name="b")

    def call(self, H, adj):
        if self.aggregator == "pooling":
            h_pool = tf.nn.relu(H @ self.W_pool + self.b_pool)
            neigh_agg = tf.einsum("bij,bjk->bik", adj, h_pool)
            neigh_agg = tf.reduce_max(neigh_agg, axis=2, keepdims=True)
            neigh_agg = tf.broadcast_to(neigh_agg, tf.shape(h_pool))
        else:
            neigh_agg = adj @ H
            
        combined = tf.concat([H, neigh_agg], axis=-1)
        out = combined @ self.W + self.b
        out = tf.math.l2_normalize(out, axis=-1)
        return self.activation(out) if self.activation else out

class GATLayer(layers.Layer):
    def __init__(self, units, num_heads=8, concat=True, feature_dropout=0.6, attn_dropout=0.6,
                 l2_reg=1e-4, leaky_alpha=0.2, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units           = units
        self.num_heads       = num_heads
        self.concat          = concat
        self.feature_dropout = feature_dropout
        self.attn_dropout    = attn_dropout
        self.l2_reg          = l2_reg
        self.leaky_alpha     = leaky_alpha
        self.activation      = keras.activations.get(activation)

    def build(self, input_shape):
        in_dim = input_shape[0][-1]
        reg = tf.keras.regularizers.l2(self.l2_reg)
        self.W = self.add_weight(shape=(in_dim, self.num_heads * self.units), initializer="glorot_uniform", regularizer=reg, name="W")
        self.a_src = self.add_weight(shape=(self.num_heads, self.units), initializer="glorot_uniform", regularizer=reg, name="a_src")
        self.a_dst = self.add_weight(shape=(self.num_heads, self.units), initializer="glorot_uniform", regularizer=reg, name="a_dst")
        out_dim = self.num_heads * self.units if self.concat else self.units
        self.bias = self.add_weight(shape=(out_dim,), initializer="zeros", name="bias")

    def call(self, inputs, training=False):
        H, adj = inputs
        N = tf.shape(H)[1]
        Wh = H @ self.W
        Wh = tf.reshape(Wh, (-1, N, self.num_heads, self.units))
        if training:
            Wh = tf.nn.dropout(Wh, rate=self.feature_dropout)
        
        e_src = tf.einsum("bnhu,hu->bnh", Wh, self.a_src)
        e_dst = tf.einsum("bnhu,hu->bnh", Wh, self.a_dst)
        attn_logits = e_src[:, :, tf.newaxis, :] + e_dst[:, tf.newaxis, :, :]
        attn_logits = tf.nn.leaky_relu(attn_logits, alpha=self.leaky_alpha)
        
        adj_mask = tf.cast(adj == 0, tf.float32) * -1e9
        attn_logits = attn_logits + adj_mask[:, :, :, tf.newaxis]
        attn_weights = tf.nn.softmax(attn_logits, axis=2)
        if training:
            attn_weights = tf.nn.dropout(attn_weights, rate=self.attn_dropout)
            
        out = tf.einsum("bijh,bjhu->bihu", attn_weights, Wh)
        if self.concat:
            out = tf.reshape(out, (-1, N, self.num_heads * self.units))
        else:
            out = tf.reduce_mean(out, axis=2)
            
        out = out + self.bias
        return self.activation(out) if self.activation else out

class GCN(keras.Model):
    def __init__(self, hidden_dim, num_classes, num_layers=2, dropout=0.5, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.convs = [GCNLayer(hidden_dim, activation="relu", l2_reg=l2_reg) for _ in range(num_layers)]
        self.lns = [layers.LayerNormalization() for _ in range(num_layers)]
        self.dropout = dropout
        self.dense = layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    def call(self, inputs, training=False):
        x, adj = inputs
        mask = tf.cast(tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True) > 0, tf.float32)
        h = x
        for conv, ln in zip(self.convs, self.lns):
            h = conv(h, adj)
            h = ln(h)
            if training:
                h = tf.nn.dropout(h, rate=self.dropout)
        h = h * mask
        x_pool = tf.reduce_mean(h, axis=1)
        return self.dense(x_pool)

class GraphSAGE(keras.Model):
    def __init__(self, hidden_dim, num_classes, num_layers=2, aggregator="pooling", dropout=0.5, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.convs = [SAGELayer(hidden_dim, aggregator=aggregator, activation="relu", l2_reg=l2_reg) for _ in range(num_layers)]
        self.lns = [layers.LayerNormalization() for _ in range(num_layers)]
        self.dropout = dropout
        self.dense = layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    def call(self, inputs, training=False):
        x, adj = inputs
        mask = tf.cast(tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True) > 0, tf.float32)
        h = x
        for conv, ln in zip(self.convs, self.lns):
            h = conv(h, adj)
            h = ln(h)
            if training:
                h = tf.nn.dropout(h, rate=self.dropout)
        h = h * mask
        x_pool = tf.reduce_mean(h, axis=1)
        return self.dense(x_pool)

class GAT(keras.Model):
    def __init__(self, num_classes, hidden_units=32, num_heads=8, num_layers=2, dropout=0.6, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        D = hidden_units * num_heads
        self.input_proj = layers.Dense(D, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.layers_att = []
        self.layers_ln = []
        for i in range(num_layers):
            is_final = (i == num_layers - 1)
            self.layers_att.append(GATLayer(hidden_units, num_heads, concat=not is_final, 
                                           feature_dropout=dropout, attn_dropout=dropout, 
                                           l2_reg=l2_reg, activation="elu"))
            self.layers_ln.append(layers.LayerNormalization())
        self.dropout_rate = dropout
        self.mlp = layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    def call(self, inputs, training=False):
        x, adj = inputs
        mask = tf.cast(tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True) > 0, tf.float32)
        h = self.input_proj(x)
        for att, ln in zip(self.layers_att, self.layers_ln):
            h_att = att([h, adj], training=training)
            if h_att.shape[-1] == h.shape[-1]:
                h = ln(h_att + h)
            else:
                h = ln(h_att)
        h = h * mask
        x_pool = tf.reduce_mean(h, axis=1)
        return self.mlp(x_pool)
