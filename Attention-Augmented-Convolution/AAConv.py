"""Augmented Attention Convolution Block.

TODO: need to change this so it is compatible with the protein structure prediction code
"""

import tensorflow as tf
from tensorflow.keras import layers


class AAConv(layers.Layer):
    """Augmented attention block.

    B: batch size
    C: channels
    H: height
    W: width
    Nh: number of heads

    Shapes:
        INPUT: (B, C_IN, H, W)
        OUPUT: (B, C_OUT, H, W)

    NOTE: the format must be: "NCHW"

    Attributes:
        channels_out (int): output channels of this block
        kernel_size (tuple): size of the kernel
        depth_k (int): total depth size for the query or key
        depth_v (int): total depth size for the value
        num_heads (int): number of heads for the attention
        relative_pos (bool): whether to include relative positional encoding
        dilation (int): dilation of the convolution operation
        regularizer (tf.keras regularizer): regularization for the weights and biases
        activation (tf.keras activation): activation function
        kernel_init (function): function for kernel initializer
    """

    def __init__(
        self, 
        channels_out,
        kernel_size,
        depth_k, 
        depth_v, 
        num_heads, 
        relative_pos=False, 
        dilation=1,
        regularizer=None, 
        activation=None,
        kernel_init=None,
        **kwargs):

        super(AAConv, self).__init__(**kwargs)

        if depth_k % num_heads != 0:
            raise ValueError(
                'depth_k {} must be divisible by num_heads {}'.format(
                depth_k, num_heads))

        if depth_v % num_heads != 0:
            raise ValueError(
                'depth_v {} must be divisible by num_heads {}'.format(
                depth_k, num_heads))

        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.depth_k = depth_k
        self.depth_v = depth_v
        self.num_heads = num_heads
        self.relative_pos = relative_pos
        self.dilation = dilation
        self.regularizer = regularizer
        self.activation = activation
        self.kernel_init = kernel_init

        self.dkh = depth_k // num_heads
        self.dvh = depth_v // num_heads


    def build(self, input_shapes):
        """Initialize the weights of the convolution layers.
        """
        
        self.conv = layers.Conv2D(
            self.channels_out - self.depth_v, 
            self.kernel_size,
            data_format='channels_first',
            activation=self.activation,
            kernel_initializer=self.kernel_init,
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
            padding='same',
            dilation_rate=self.dilation,
            name='AA_Conv')

        self.self_atten_conv = layers.Conv2D(
            2 * self.depth_k + self.depth_v, 
            1,
            data_format='channels_first',
            activation=self.activation,
            kernel_initializer=self.kernel_init,
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
            padding='same',
            dilation_rate=self.dilation,
            name='AA_Atten_Conv')


    def _split_heads_2d(self, inputs):
        """Split channels into multiple heads.

        Args:
            inputs: tensor of shape (B, C, H, W)

        Returns:
            tensor of shape (B, Nh, H, W, C // Nh)
        """

        in_shape = tf.shape(inputs)

        ret_shape = [
            in_shape[0], 
            self.num_heads, 
            in_shape[1] // self.num_heads, 
            in_shape[2],
            in_shape[3]]

        # (B, Nh, C // Nh, H, W)
        split = tf.reshape(inputs, ret_shape)

        # (B, Nh, H, W, C // Nh)
        result = tf.transpose(split, [0, 1, 3, 4, 2])

        return result

    
    def _combine_heads_2d(self, inputs):
        """Combine the heads together.

        Args:
            tensor of shape (B, Nh, H, W, C)  

        Returns:
            tensor of shape (B, H, W, Nh * C)  
        """

        # (B, H, W, NUM_HEADS, C_IN)  
        transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])

        trans_shape = tf.shape(transposed)
        # N_H, C = tf.shape(transposed)[-2:]

        ret_shape = tf.concat(
            [tf.shape(transposed)[:-2], 
            [trans_shape[-1] * trans_shape[-2]]], 
            axis=0)
        result = tf.reshape(transposed, ret_shape)

        return result


    def _self_attention_2d(self, inputs):
        """Apply self 2d self attention to input.

        NOTE: unlike the implementation in the paper, we do 
        not have an extra convolution layer for projection at
        the end of the self attention
        
        Args:
            inputs: tensor of shape (B, C, H, W)

        Returns:
            tensor of shape (B, depth_v, H, W)  
        """

        in_shape = tf.shape(inputs)
        H = in_shape[2]
        W = in_shape[3]

        kqv = self.self_atten_conv(inputs)

        # (B, dk or dv, H, W)
        k, q, v = tf.split(
            kqv, 
            [self.depth_k, self.depth_k, self.depth_v], 
            axis=1)

        q *= self.dkh ** -0.5 # scaled dot−product

        # (B, Nh, H, W, dk or dv // Nh)
        q = self._split_heads_2d(q)
        k = self._split_heads_2d(k)
        v = self._split_heads_2d(v)


        # returns shape: (B, NUM_HEADS, H * W, d)
        flatten_hw = lambda x, d: tf.reshape(x, [-1, self.num_heads, H * W, d])

        # (B, NUM_HEADS, H * W, H * W)
        logits = tf.linalg.matmul(
            flatten_hw(q, self.dkh),
            flatten_hw(k, self.dkh),
            transpose_b=True)

        # TODO: include relative positional encoding here

        weights = tf.math.softmax(logits)

        # (B, NUM_HEADS, H * W, dvh)
        attn_out = tf.linalg.matmul(
            weights, 
            flatten_hw(v, self.dvh))

        # (B, NUM_HEADS, H, W, dvh)    
        attn_out = tf.reshape(
            attn_out, 
            [-1, self.num_heads, H, W, self.dvh]) 

        # (B, H, W, NUM_HEADS * dvh)   
        attn_out = self._combine_heads_2d(attn_out)  

        # (B, NUM_HEADS * dvh = dv, H, W)   
        attn_out = tf.transpose(attn_out, [0, 3, 1, 2])

        return attn_out

    @tf.function
    def call(self, inputs):
        conv_out = self.conv(inputs)
        attn_out = self._self_attention_2d(inputs)
        result = tf.concat([conv_out, attn_out], axis=1)

        return result