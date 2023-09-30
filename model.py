from layers import GraphAttentionConvolution, GraphConvolutionSparse,InnerProductDecoder,GraphAttentionLSTMConvolution
from utils import *
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras import layers

class HDGATModel():
    def __init__(self, placeholders, num_features, emb_dim, features_nonzero, adj_nonzero, num_r, name, act=tf.compat.v1.nn.elu):
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.emb_dim = emb_dim
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']
        self.act = act
        self.att = tf.compat.v1.Variable(tf.compat.v1.constant([0.9,0.45,0.4]))
        self.num_r = num_r
        self.layer_norm=LayerNormalization(axis=1)


        with tf.compat.v1.variable_scope(self.name):
            self.build()

    def build(self):
        self.adj = dropout_sparse(self.adj, 1-self.adjdp, self.adj_nonzero)

        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act)(self.inputs)


        self.hidden2 = GraphAttentionLSTMConvolution(
            name='gat_dense_layer',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            dropout=self.dropout,
            )(self.hidden1,self.adj)


        self.emb = GraphAttentionConvolution(
            name='gat_dense_layer1',
            former=self.hidden1,
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            dropout=self.dropout
        )(self.hidden2, self.adj)



        self.embeddings=self.hidden1*self.att[0]+self.hidden2*self.att[1]+self.emb*self.att[2]

        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim, num_r=self.num_r, act=tf.compat.v1.nn.sigmoid)(self.embeddings)
