import tensorflow as tf
from tensorflow.keras import layers

class AAConv(layers.Layer):
    """Augmented attention block.

    B: batch size
    C: channels
    H: height
    W: width

    Shapes:
        INPUT: (B, C_IN, H, W)
        OUPUT: (B, C_OUT, H_OUT, W_OUT)

    NOTE: the format must be: "NCHW"

    Attributes:
        channels_out (int): output channels of this block
        kernel_size (tuple): size of the kernel
        depth_k (int): total depth size for the query or key
        depth_v (int): total depth size for the value
        num_heads (int): number of heads for the attention
        relative_pos (bool): whether to include relative positional encoding
        dilation (int): dilation of the convolution operation
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
        regularizer=None, **kwargs):

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

        self.dkh = depth_k // num_heads
        self.dvh = depth_v // num_heads


    def build(self, input_shapes):
        
        self.conv = layers.Conv2D(
            self.channels_out - self.depth_v, 
            self.kernel_size,
            # activation=nameToActivation(self.activation),
            # kernel_initializer=kernel_init,
            data_format='channels_first',
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
            padding='same',
            dilation_rate=self.dilation,
            name='AA_Conv')

        self.self_atten_conv1 = layers.Conv2D(
            2 * self.depth_k + self.depth_v, 
            1,
            data_format='channels_first',
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
            padding='same',
            dilation_rate=self.dilation,
            name='AA_Atten_Conv1')

        self.self_atten_conv2 = layers.Conv2D(
            self.depth_v, 
            1,
            data_format='channels_first',
            kernel_regularizer=self.regularizer,
            bias_regularizer=self.regularizer,
            padding='same',
            dilation_rate=self.dilation,
            name='AA_Atten_Conv2')


    def _split_heads_2d(self, inputs):
        """Split channels into multiple heads.

        Args:
            inputs: tensor of shape (B, C_IN, H, W)

        Returns:
            tensor of shape (B, NUM_HEADS, H, W, C_IN // NUM_HEADS)
        """

        B, C, H, W = tf.shape(inputs)

        ret_shape = [B, self.num_heads, C // self.num_heads, H, W]
        split = tf.reshape(inputs, ret_shape)

        return tf.transpose(split, [0, 1, 3, 4, 2])

    
    def _combine_heads_2d(self, inputs):
        """Combine the heads together.

        Args:
            tensor of shape (B, NUM_HEADS, H, W, C_IN)  

        Returns:
            tensor of shape (B, H, W, NUM_HEADS * C_IN)  
        """

        # (B, H, W, NUM_HEADS, C_IN)  
        transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])

        N_H, C = tf.shape(inputs)[-2:]
        ret_shape = tf.shape(transposed)[:-2] + (N_H * C, )
        return tf.reshape(transposed, ret_shape)


    def _self_attention_2d(self, inputs):
        """Apply self 2d self attention to input.

        Args:
            inputs: tensor of shape (B, C_IN, H, W)

        Returns:
            tensor of shape
        """


        kqv = self.self_atten_conv1(inputs)

        # (B, dk or dv, H, W)
        k, q, v = tf.split(
            kqv, 
            [self.depth_k, self.depth_k, self.depth_v], 
            axis=1)

        q *= self.dkh ** -0.5 # scaled dot−product

        q = self._split_heads_2d(q)
        k = self._split_heads_2d(k)
        v = self._split_heads_2d(v)

        _, _, H, W = tf.shape(inputs)

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

        # (B, dv, H, W)  
        attn_out_conv = self.self_atten_conv2(attn_out)

        return attn_out_conv

    # @tf.function
    def call(self, inputs):
        conv_out = self.conv(inputs)
        attn_out = self._self_attention_2d(inputs)

        return tf.concat([conv_out, attn_out], axis=1)