import tensorflow as tf
import numpy as np
import numpy as np


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        #res = tf.matmul(x, y)
        res =tf.matmul(tf.cast(x,tf.float64),y)
    return res


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class GraphConvolution:
    """Graph convolution layer."""
    def __init__(self,input_dim, output_dim,batch=70,
                 act=tf.nn.relu,dropout =True,name="gnn"):
        if dropout:
            self.dropout = 0.6
        else:
            self.dropout = 0.0
        self.act = act
        self.inputs = inputs
        self.sim_martix = sim_martix
        self.shape = [input_dim,output_dim]
        self.name = name
        self.vars = {}
        self.batch = batch
        with tf.variable_scope(self.name + '_vars'):
            for i in range(self.batch):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],name='weights_' + str(i))
    
    def gnn(self,inputs,sim_martix):
        # convolve 卷积的实现。
        #主要是根据论文中公式Z = \tilde{D}^{-1/2}\tilde{A}^{-1/2}X\theta实现
        supports = list()
        for i in range(self.batch):
            A = sim_martix[i]
            I = np.matrix(np.eye(A.shape[0]))
            A_hat = A + I
            D = np.array(np.sum(A_hat, axis=0))[0]
            D_hat = np.matrix(np.diag(D))
            #D**-1 * A * X
            mid_1 = dot(inputs[i], self.vars['weights_' + str(i)])#[64,4096] [4096,500]=>[64,500]
            mid_2 = dot(D_hat**-1,mid_1)#[64,64] [64,500] ==>[64,500]
            supports.append(mid_2)
        output = tf.add_n(supports)
        return self.act(output)
        
"""
#inputs,sim_martix,input_dim, output_dim,
#inputs =  tf.Variable(tf.random_normal([70,64,4096]))
#sim_martix =  tf.Variable(tf.random_normal([70,64,64]))

inputs = np.ones([70,64,4096],dtype=float)
sim_martix = np.ones([70,64,64])
input_dim = 4096
output_dim = 500
model = GraphConvolution(input_dim,output_dim)
model.gnn(inputs,sim_martix)
"""  

def bulild_gnn(inputs,sim_martix):
    model1 = GraphConvolution(4096,500)
    model2 = GraphConvolution(500,200)
    model3 = GraphConvolution(200,64)
    mid_1 = model1.gnn(inputs,sim_martix)
    #print(mid_1.shape)
    mid_2 = model2.gnn(mid_1,sim_martix)
    mid_3 = model3.gnn(mid_2,sim_martix)
    return mid_3
inputs = np.ones([70,64,4096],dtype=float)
sim_martix = np.ones([70,64,64],dtype=float)
out = bulild_gnn(inputs,sim_martix)





