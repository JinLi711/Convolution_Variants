"""State of the art convolution layers.

Includes:
    Drop Block
    ECA Net
    CBAM
    Augmented Attention Convolution layer
    Mixed Depthwise Convolution layer

NOTE: the format for all layers must be: "NCHW"
NOTE: only ECA and CBAM can use multiple groups.

B: batch size
C: channels
H: height
W: width
Nh: number of heads
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models 
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
import tensorflow.keras.activations as activations
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.initializers as initializers
import tensorflow.keras.constraints as constraints
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn

# -------------------------------------------------------------------- #
# Helper Functions
# -------------------------------------------------------------------- #

castFloat32 = lambda x: tf.cast(x, dtype=tf.dtypes.float32)


# -------------------------------------------------------------------- #
# DropBlock
# -------------------------------------------------------------------- #

class DropBlock(layers.Layer):
    """Drop Block layer.

    For each channel of the input, randomly dropout an entire block.

    NOTE: This should be applied after a convolution operation.
    NOTE: Height must equal width.

    Shapes:
        INPUT: (B, C, H, W)
        OUPUT: (B, C, H, W)

    Attributes:
        keep_prob (float): probability of keeping the block
        dropblock_size (int): size of the drop block
    """

    def __init__(self, keep_prob, dropblock_size, **kwargs):

        super(DropBlock, self).__init__()

        self.keep_prob = keep_prob
        self.dropblock_size = dropblock_size


    def build(self, input_shapes):
        if input_shapes[2] != input_shapes[3]:
            raise ValueError('DropBlock only works when height = width.')


    @tf.function
    def call(self, inputs, training):

        if training:
            inshape = tf.shape(inputs)
            width = inshape[3]

            # original dropblock_size is not used because tf.math.minimum produces a
            # scalar tensor.
            # dropblock_size = tf.math.minimum(self.dropblock_size, width)
            dropblock_size = self.dropblock_size

            # aka the gamma parameter
            seed_drop_rate = (1.0 - self.keep_prob) * castFloat32(width) ** 2 \
                / castFloat32(dropblock_size) ** 2 / (castFloat32(width - dropblock_size) + 1.0) ** 2

            # force the block to be inside the feature map
            tf_int = lambda x: tf.cast(x, tf.int32)
            w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
            valid_block_center = tf.math.logical_and(
                tf.math.logical_and(w_i >= tf_int(dropblock_size // 2),
                                w_i < width - (dropblock_size - 1) // 2),
                tf.math.logical_and(h_i >= tf_int(dropblock_size // 2),
                                h_i < width - (dropblock_size - 1) // 2))

            valid_block_center = tf.expand_dims(valid_block_center, 0)
            valid_block_center = tf.expand_dims(valid_block_center, 0)

            randnoise = tf.random.uniform(inshape, dtype=tf.float32)
            block_pattern = (1 - castFloat32(valid_block_center) \
                + castFloat32(1 - seed_drop_rate) + randnoise) >= 1
            block_pattern = castFloat32(block_pattern)
            
            if dropblock_size == width:
                block_pattern = tf.math.reduce_min(
                    block_pattern,
                    axis=[2, 3],
                    keepdims=True)

            else:
                ksize = [1, 1, dropblock_size, dropblock_size]
                block_pattern = -tf.nn.max_pool(
                    -block_pattern, 
                    ksize=ksize, 
                    strides=[1, 1, 1, 1], 
                    padding='SAME',
                    data_format='NCHW')

            percent_ones = castFloat32(tf.reduce_sum(block_pattern)) \
                / castFloat32(tf.size(block_pattern))

            inputs = inputs / castFloat32(percent_ones) * castFloat32(block_pattern)
        
        return inputs 


# -------------------------------------------------------------------- #
# ECA
# -------------------------------------------------------------------- #

class ECAConv(layers.Layer):
    """ECA Conv layer. 

    NOTE: This should be applied after a convolution operation.

    Shapes:
        INPUT: (B, C, H, W)
        OUPUT: (B, C_OUT, H, W)

    Attributes:
        filters (int): number of channels of input
        eca_k_size (int): kernel size for the 1D ECA layer
    """

    def __init__(
        self, 
        filters,
        eca_k_size,
        groups=1,
        **kwargs):

        super(ECAConv, self).__init__()

        self.filters = filters
        self.eca_k_size = eca_k_size
        self.kwargs = kwargs
        self.groups = groups

    def build(self, input_shapes):

        if self.groups == 1:
            self.conv = layers.Conv2D(
                filters=self.filters,
                data_format='channels_first',
                **self.kwargs)
        else:
            self.conv = GroupConv2D(
                filters=self.filters,
                data_format='channels_first',
                groups=self.groups,
                **self.kwargs)
            
        self.eca_conv = layers.Conv1D(
            filters=1, 
            kernel_size=self.eca_k_size,
            padding='same', 
            use_bias=False,
            data_format='channels_first')

    def call(self, inputs):

        x = self.conv(inputs)

        # (B, 1, C)
        attn = tf.math.reduce_mean(x, [2, 3])[:, tf.newaxis, :]

        # (B, 1, C)
        attn = self.eca_conv(attn)

        # (B, C, 1, 1)
        attn = tf.expand_dims(tf.transpose(attn, [0, 2, 1]), 3)

        # (B, C, 1, 1)
        attn = tf.math.sigmoid(attn)

        return x * attn



# -------------------------------------------------------------------- #
# CBAM
# -------------------------------------------------------------------- #

class ChannelGate(layers.Layer):
    """Apply Channelwise attention to input.

    Shapes:
        INPUT: (B, C, H, W)
        OUPUT: (B, C, H, W)

    Attributes:
        gate_channels (int): number of channels of input
        reduction_ratio (int): factor to reduce the channels in FF layer
        pool_types (list): list of pooling operations
    """

    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=['avg', 'max'],
        **kwargs):

        super(ChannelGate, self).__init__()

        all_pool_types = {'avg', 'max'}
        if not set(pool_types).issubset(all_pool_types):
            raise ValueError('The available pool types are: {}'.format(all_pool_types))

        self.gate_channels = gate_channels
        self.reduction_ratio = reduction_ratio
        self.pool_types = pool_types
        self.kwargs = kwargs

    def build(self, input_shape):
        hidden_units = self.gate_channels // self.reduction_ratio
        self.mlp = models.Sequential([
            layers.Dense(hidden_units, activation='relu'),
            layers.Dense(self.gate_channels, activation=None)
        ])


    def apply_pooling(self, inputs, pool_type):
        """Apply pooling then feed into ff.

        Args:
            inputs (tf.tensor): shape (B, C, H, W)

        Returns:
            (tf.tensor) shape (B, C)
        """

        if pool_type == 'avg':
            pool = tf.math.reduce_mean(inputs, [2, 3])
        elif pool_type == 'max':
            pool = tf.math.reduce_max(inputs, [2, 3])

        channel_att = self.mlp(pool)
        return channel_att

    def call(self, inputs):
        pools = [self.apply_pooling(inputs, pool_type) \
            for pool_type in self.pool_types]

        # (B, C, 1, 1)
        attn = tf.math.sigmoid(tf.math.add_n(pools))[:, :, tf.newaxis, tf.newaxis]

        return attn * inputs


class SpatialGate(layers.Layer):
    """Apply spatial attention to input.

    Shapes:
        INPUT: (B, C, H, W)
        OUPUT: (B, C, H, W)

    Attributes:
        None
    """

    def __init__(self, **kwargs):
        super(SpatialGate, self).__init__()

        self.kwargs = kwargs

    def build(self, input_shapes):
        self.conv = layers.Conv2D(
            filters=1, 
            kernel_size=7,
            strides=1,
            padding='same',
            data_format='channels_first')

        # TODO: we may not want to do batch normalization over the batch dimensions
        self.bn = layers.BatchNormalization(axis=1)

    def call(self, inputs):
        pooled_channels = tf.concat(
            [tf.math.reduce_max(inputs, axis=1, keepdims=True),
            tf.math.reduce_mean(inputs, axis=1, keepdims=True)],
            axis=1)

        scale = self.bn(self.conv(pooled_channels))
        scale = tf.math.sigmoid(scale)

        return inputs * scale


class CBAM(layers.Layer):
    """CBAM layer. 

    NOTE: This should be applied after a convolution operation.

    Shapes:
        INPUT: (B, C, H, W)
        OUPUT: (B, C_OUT, H, W)

    Attributes:
        filters (int): number of channels of input
        reduction_ratio (int): factor to reduce the channels in FF layer
        pool_types (list): list of pooling operations
        spatial (bool): whether to use spatial attention 
    """

    def __init__(
        self, 
        filters, 
        reduction_ratio=16, 
        pool_types=['avg', 'max'], 
        spatial=True, 
        groups=1,
        **kwargs):

        super(CBAM, self).__init__()

        self.filters = filters
        self.reduction_ratio = reduction_ratio
        self.pool_types = pool_types
        self.spatial = spatial
        self.kwargs = kwargs
        self.groups = groups

    def build(self, input_shapes):

        if self.groups == 1:
            self.conv = layers.Conv2D(
                filters=self.filters,
                data_format='channels_first',
                **self.kwargs)

        else:
            self.conv = GroupConv2D(
                filters=self.filters,
                data_format='channels_first',
                groups=self.groups,
                **self.kwargs)
            
        self.ChannelGate = ChannelGate(
            self.filters,
            self.reduction_ratio,
            self.pool_types)

        if self.spatial:
            self.SpatialGate = SpatialGate()

    def call(self, inputs):

        x = self.conv(inputs)

        x = self.ChannelGate(x)

        if self.spatial:
            x = self.SpatialGate(x)

        return x

# -------------------------------------------------------------------- #
# Mixed Depthwise
# -------------------------------------------------------------------- #


class MixConv(layers.Layer):
    """MixConv, a convolution layer with different convolution kernel sizes.

    Shapes:
        INPUT: (B, C_IN, H, W)
        OUPUT: (B, C_OUT, H, W)

    Attributes:
        channels_out (int): total number of output channels
        kernel_sizes (list): list of kernel sizes, one kernel size for each group
        depthwise (bool): whether the convolution should be depthwise or not
    """

    def __init__(
        self, 
        channels_out,
        kernel_sizes,
        depthwise=True,
        **kwargs):

        super(MixConv, self).__init__()

        self.channels_out = channels_out
        self.kernel_sizes = kernel_sizes
        self.depthwise = depthwise
        self.kwargs = kwargs

        self.num_groups = len(kernel_sizes)


    def _split_channels(self, total_channels, num_groups):
        """Split channels into multiple groups.

        Each group will have approximately equal number of channels. 
        The left over channels with be placed in the first group.

        Args:
            total_channels (int): number of filters for input or output
            num_groups (int): number of groups to split the filters

        Returns:
            (list) list of filter size for each group
        """

        split = [total_channels // num_groups for _ in range(num_groups)]
        split[0] += total_channels - sum(split)
        return split


    def build(self, input_shape):

        in_channel = input_shape[1]

        self.in_channels = self._split_channels(
            in_channel, 
            self.num_groups)

        self.out_channels = self._split_channels(
            self.channels_out, 
            self.num_groups)
        
        if self.depthwise:
            self.depthwise_layers = []
            self.pointwise_layers = []

            for i, k_size in enumerate(self.kernel_sizes):

                layer = tf.keras.layers.DepthwiseConv2D(
                    kernel_size=k_size,
                    padding='same',
                    data_format='channels_first',
                    depth_multiplier=1,
                    **self.kwargs)
                self.depthwise_layers.append(layer)

                layer = tf.keras.layers.Conv2D(
                    filters=self.out_channels[i],
                    kernel_size=1,
                    padding='same',
                    data_format='channels_first',
                    **self.kwargs)
                self.pointwise_layers.append(layer)

        else:
            self.layers = []

            for i, k_size in enumerate(self.kernel_sizes):

                layer = tf.keras.layers.Conv2D(
                    filters=self.out_channels[i],
                    kernel_size=k_size,
                    padding='same',
                    data_format='channels_first',
                    **self.kwargs)
                self.layers.append(layer)

        


    @tf.function
    def call(self, inputs):
        
        x_splits = tf.split(
            inputs, 
            self.in_channels, 
            1)

        if self.depthwise:
            x_outputs = [p(d(x)) for x, d, p in zip(
                x_splits, 
                self.depthwise_layers,
                self.pointwise_layers)]
        else:
            x_outputs = [d(x) for x, d in zip(
                x_splits, 
                self.layers)]

        result = tf.concat(x_outputs, 1)
        return result


# -------------------------------------------------------------------- #
# Augmented Attention
# -------------------------------------------------------------------- #

class AAConv(layers.Layer):
    """Augmented attention block.

    Shapes:
        INPUT: (B, C_IN, H, W)
        OUPUT: (B, C_OUT, H, W)

    NOTE: relative positional encoding has not yet been implemented

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

        if self.relative_pos:
            raise NotImplementedError
            self.rel_embed_w = self.add_weight(
                # shape=(2 * , n_in, self.n_out),
                initializer=tf.random_normal_initializer(self.dkh ** -0.5),
                trainable=True,
                regularizer=self.regularizer,
                name='AA_rel_embed_w')


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


    def _relative_logits(self, inputs, height, width):
        """Compute relative logits
        """

        # relative logits in width dimension
        raise NotImplementedError(
            "Relative positional encoding has not yet been implemented.")

            
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

        if self.relative_pos:
            rel_logits_h, rel_logits_w = self._relative_logits(q, H, W)
            logits += rel_logits_h
            logits += rel_logits_w

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


# -------------------------------------------------------------------- #
# Grouped Convolution
# -------------------------------------------------------------------- #


class GroupConvBase(tf.keras.layers.Layer):

    def __init__(
        self, 
        rank, 
        filters, 
        kernel_size, 
        groups=1, 
        strides=1, 
        padding='VALID', 
        data_format=None,
        dilation_rate=1,
        activation=None, 
        use_bias=True, 
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros', 
        kernel_regularizer=None, 
        bias_regularizer=None, 
        activity_regularizer=None,
        kernel_constraint=None, 
        bias_constraint=None, **kwargs):

        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        if filters % groups != 0:
            raise ValueError("Groups must divide filters evenly, but got {}/{}".format(filters, groups))

        self.filters = filters
        self.groups = groups
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.data_format = data_format
        self.padding = padding
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.rank = rank

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if conv_utils.normalize_data_format(self.data_format) == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim // self.groups, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        outputs = tf.nn.conv2d(
            inputs, 
            self.kernel, 
            strides=self.strides,
            data_format="NCHW", 
            dilations=self.dilation_rate,
            name=self.name,
            padding=self.padding)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            "groups": self.groups,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        return {list(super(GroupConvBase, self).get_config().items()) + list(config.items())}


class GroupConv2D(GroupConvBase):
    """Grouped convolution. 
    
    Essentially the same as regular Conv2D, except inputs are seperated int
    groups.

    Code is taken from here: https://github.com/tensorflow/tensorflow/issues/34024
    Also, check out this pull request: https://github.com/tensorflow/tensorflow/pull/25818
    """

    def __init__(
        self, 
        filters, 
        kernel_size,
        groups, 
        strides=(1, 1), 
        padding='valid', 
        data_format=None, 
        dilation_rate=(1, 1), 
        activation=None,
        use_bias=True, 
        kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', 
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None, 
        kernel_constraint=None, 
        bias_constraint=None,
        **kwargs):

        super(GroupConv2D, self).__init__(
            rank=2,
            filters=filters, 
            kernel_size=kernel_size, 
            groups=groups, 
            strides=strides,
            padding=padding.upper(),
            data_format=data_format, dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias, kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)


