# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import
from ..expr import const
from .op import register_gradient
import tvm.relay as relay
import tvm.relay.op as op
from .transform import collapse_sum_like, broadcast_to_like, reshape_like, cast_like
from .transform import where, transpose
from .tensor import exp, negative, power, less, equal, divide
from .tensor import zeros_like, ones_like
from .reduce import sum as reduce_sum
from . import nn as _nn

@register_gradient("log")
def log_grad(orig, grad):
    """Returns [grad * (1 / x)]"""
    x = orig.args[0]
    return [grad * ones_like(x) / x]


@register_gradient("exp")
def exp_grad(orig, grad):
    """Returns [grad * exp(x)]"""
    return [grad * exp(orig.args[0])]


@register_gradient("sqrt")
def sqrt_grad(orig, grad):
    """Returns [grad * 0.5 * (x ^ -0.5)]"""
    a = const(0.5)  # (TODO) type?
    return [grad * a * power(orig.args[0], negative(a))]


@register_gradient("sigmoid")
def sigmoid_grad(orig, grad):
    """Returns [grad * sigmoid(x) * (1 - sigmoid(x))]."""
    return [grad * orig * (ones_like(orig) - orig)]


@register_gradient("tanh")
def tanh_grad(orig, grad):
    """Returns grad * (1 - tanh(x) * tanh(x))."""
    return [grad * ones_like(orig) - orig * orig]


@register_gradient("nn.relu")
def relu_grad(orig, grad):
    """Returns grad * (select(x < 0, 0, 1))."""
    x = orig.args[0]
    zeros = zeros_like(x)
    ones = ones_like(x)
    return [where(less(x, zeros), zeros, ones * grad)]


@register_gradient("add")
def add_grad(orig, grad):
    """Returns [grad, grad]"""
    return [collapse_sum_like(grad, orig.args[0]),
            collapse_sum_like(grad, orig.args[1])]


@register_gradient("subtract")
def subtract_grad(orig, grad):
    """Returns [grad, -grad]"""
    return [collapse_sum_like(grad, orig.args[0]),
            collapse_sum_like(negative(grad), orig.args[1])]


@register_gradient("multiply")
def multiply_grad(orig, grad):
    """Returns [grad * y, grad * x]"""
    x, y = orig.args
    return [collapse_sum_like(grad * y, x),
            collapse_sum_like(grad * x, y)]


@register_gradient("divide")
def divide_grad(orig, grad):
    """Returns [grad / y,  - grad * (x / y) / y]"""
    x, y = orig.args
    return [collapse_sum_like(grad / y, x),
            collapse_sum_like(- (grad * orig / y), y)]


@register_gradient("collapse_sum_like")
def collapse_sum_like_grad(orig, grad):
    """Returns [broadcast_to_like(grad, x), 0]"""
    x, y = orig.args
    return [broadcast_to_like(grad, x), zeros_like(y)]


@register_gradient("abs")
def abs_grad(orig, grad):
    """Returns grad * (select(x < 0, -1, 1))."""
    x = orig.args[0]
    zeros = zeros_like(x)
    ones = ones_like(x)
    return [where(less(x, zeros), -ones * grad, ones * grad)]


@register_gradient("clip")
def clip_grad(orig, grad):
    """Returns grad * (select(x < min || max < x , 0, 1))."""
    x = orig.args[0]
    a_min = orig.attrs.get_int("a_min")
    a_max = orig.attrs.get_int("a_max")
    a_mins = broadcast_to_like(const(a_min), x)
    a_maxs = broadcast_to_like(const(a_max), x)
    zeros = zeros_like(x)
    ones = ones_like(x)
    return [where(less(x, a_mins), zeros, where(less(a_maxs, x), zeros, ones * grad))]

@register_gradient("nn.max_pool2d")
def max_pool2d_grad(orig, grad):
    attrs = orig.attrs
    pool_grad = _nn.max_pool2d_grad(grad, orig.args[0], pool_size=attrs.pool_size,
                                    strides=attrs.strides, padding=attrs.padding,
                                    layout=attrs.layout, ceil_mode=attrs.ceil_mode)
    return [pool_grad]

@register_gradient("nn.avg_pool2d")
def avg_pool2d_grad(orig, grad):
    attrs = orig.attrs
    pool_grad = _nn.avg_pool2d_grad(grad, orig.args[0], pool_size=attrs.pool_size,
                                    strides=attrs.strides, padding=attrs.padding,
                                    layout=attrs.layout, ceil_mode=attrs.ceil_mode,
                                    count_include_pad=attrs.count_include_pad)
    return [pool_grad]

@register_gradient("sum")
def sum_grad(orig, grad):
    """Returns [broadcast_to_like(grad, x)]"""
    return [broadcast_to_like(grad, orig.args[0])]


@register_gradient("max")
def max_grad(orig, grad):
    """Returns the gradient of max"""
    # Only support axis=0, since broadcasting orig to x behaves incorrectly
    x, axis = orig.args[0], orig.attrs.axis
    assert(axis is not None and len(axis) == 1 and int(axis[0]) == 0)
    orig = broadcast_to_like(orig, x)
    grad = broadcast_to_like(grad, x)
    indicators = cast_like(equal(orig, x), grad)
    count = reduce_sum(indicators, axis, True)
    return [divide(indicators, count) * grad]


@register_gradient("nn.softmax")
def softmax_grad(orig, grad):
    """Returns [(grad - sum(grad * orig, orig.attrs.axis, True)) * orig]"""
    return [(grad - reduce_sum(grad * orig, orig.attrs.axis, True)) * orig]


@register_gradient("negative")
def negative_grad(orig, grad):
    return [negative(broadcast_to_like(grad, orig.args[0]))]


#x `dense` y = x . yT
@register_gradient("nn.dense")
def dense_grad(orig, grad):
    # todo: check attrs
    data, weight = orig.args
    return [collapse_sum_like(op.nn.dense(grad, transpose(weight)), data),
            collapse_sum_like(transpose(op.nn.dense(transpose(data), transpose(grad))), weight)]


# UNTESTED
@register_gradient("cast")
def cast_grad(orig, grad):
    grad = cast_like(grad, orig.args[0])
    return [broadcast_to_like(grad, orig.args[0])]


# UNTESTED
@register_gradient("reshape")
def reshape_grad(orig, grad):
    return [reshape_like(grad, orig.args[0])]


# UNTESTED
@register_gradient("take")
def take_grad(orig, grad):
    x, y = orig.args
    return [relay.zeros_like(x), relay.zeros_like(y)]
    return [relay.Call(op.get("take_grad"), [x, y, grad], orig.attrs), zeros_like(y)]


@register_gradient("shape_of")
def shape_of_grad(orig, grad):
    return [zeros_like(orig.args[0])]


@register_gradient("nn.bias_add")
def bias_add_grad(orig, grad):
    """repeat the bias, then it is just add."""
    # focuse on conv2d rn.
    return [collapse_sum_like(grad, orig.args[0]),
            collapse_sum_like(grad, orig.args[1])]


@register_gradient("nn.batch_flatten")
def batch_flatten_grad(orig, grad):
    """reshape"""
    # focuse on conv2d rn.
    return [orig.args[0]]


@register_gradient("nn.global_avg_pool2d")
def global_avg_pool2d_grad(orig, grad):
    """repeat the h w dimension"""
    # focuse on conv2d rn.
    return [orig.args[0]]


@register_gradient("nn.batch_norm")
def batch_norm_grad(orig, grad):
    """multiply some stuff"""
    # batchnorm has a wrong api so we will not waste time implementing it.
    a, b, c, d, e = orig.args
    return [a, b, c, d, e]


@register_gradient("nn.conv2d")
def conv2d_grad(orig, grad):
    a, b = orig.args
    x, y = relay.TupleWrapper(relay.Call(op.get("nn.conv2d_grad"), [a, b, grad], orig.attrs), 2)
    return [a, b]


@register_gradient("zeros")
def zeros_grad(orig, grad):
    """Returns []"""
    return []


@register_gradient("ones")
def ones_grad(orig, grad):
    """Returns []"""
    return []


@register_gradient("zeros_like")
def zeros_like_grad(orig, grad):
    """Returns [0]"""
    return [zeros_like(orig.args[0])]



@register_gradient("ones_like")
def ones_like_grad(orig, grad):
    """Returns [0]"""
    return [zeros_like(orig.args[0])]


@register_gradient("split")
def split_grad(orig, grad):
    # return zero
    return [orig.args[0]]


@register_gradient("nn.cross_entropy")
def cross_entropy_grad(orig, grad):
    x, y = orig.args
    sm = op.nn.softmax(x)
    shape = op.shape_of(x)
    batch_size = op.take(shape, relay.const(0, dtype='int32'), axis=0)
    grad = grad / batch_size.astype('float32')
    return [op.sum(y, axis=1) * grad * (sm - y), -grad * op.log(sm)]
