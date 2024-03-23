# --------------------------------------------------------
# DCNv4
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
import math
from torch.autograd import gradcheck
import pandas as pd
from easydict import EasyDict as edict

from torch.cuda import Event

from functions.dcnv3_func import DCNv3Function, dcnv3_core_pytorch
from functions.dcnv4_func import DCNv4Function
torch.set_printoptions(threshold=10000)

H_in, W_in = 56, 56
N, M, D = 64, 4, 32

# H_in, W_in = 28, 28
# N, M, D = 64, 8, 32

# H_in, W_in = 14, 14
# N, M, D = 64, 16, 32

# H_in, W_in = 7, 7
# N, M, D = 64, 32, 32

# H_in, W_in = 8, 8
# N, M, D = 128, 4, 16


Kh, Kw = 3, 3
remove_center = False
P = Kh * Kw - remove_center
offset_scale = 2.0
pad = 1
dilation = 1
stride = 1
H_out = (H_in + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
W_out = (W_in + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

torch.manual_seed(3)

#@torch.no_grad()
def speed_test(func, args, inputs, name='Unknown'):

    tic = Event(enable_timing=True)
    toc = Event(enable_timing=True)
    # warmup
    for i in range(args.warmup_num):
        func(*inputs)

    total_time = 0
    tic.record()
    for i in range(args.test_num):
        o = func(*inputs)
        torch.cuda.synchronize()
    toc.record()

    avg_time = tic.elapsed_time(toc) / args.test_num
    print(
        f'>>> {name: <10} finished {args.test_num} running, avg_time: {avg_time:.6f} ms')
    return avg_time

@torch.no_grad()
def check_forward_equal_with_pytorch_half():
    input = torch.rand(N, H_in, W_in, M*D).cuda()
    print(input.shape)
    offset = (torch.rand(N, H_out, W_out, M*P*2).cuda() * 2 - 1)*10
    # offset = (torch.rand(N, H_out, W_out, M*P*2).cuda() * 2 - 1)*0
    mask_origin = torch.rand(N, H_out, W_out, M, P).cuda() + 1e-5
    mask_origin = mask_origin.half()
    mask = mask_origin
    # mask = torch.nn.functional.softmax(mask_origin, dim=-1)
    offset_mask = torch.cat([offset.unflatten(-1, (M, P * 2)), mask_origin.detach()], dim=-1).flatten(-2)

    im2col_step = 128

    input = input.half()
    offset = offset.half()
    mask = mask.half()
    offset_mask = offset_mask.half()

    dcnv3_args = [
        input,
        offset,
        mask,
        Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale,
        im2col_step, remove_center,
    ]
    output_pytorch = DCNv3Function.apply(*dcnv3_args)

    input1 = input.detach()

    def pad(om):
        padded_zero = int(math.ceil(om.shape[3]/8)*8) - om.shape[3]
        padded = torch.zeros(om.shape[0], om.shape[1], om.shape[2], padded_zero).to(om)
        return torch.cat([om, padded], dim=-1)

    dcnv4_args = [
        input1, pad(offset_mask),
        Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale,
        im2col_step, remove_center, 8, 512, 2, 256, True, True,
    ]
    output_flash_cuda = DCNv4Function.apply(*dcnv4_args)

    fwdok = torch.allclose(output_flash_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_flash_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_flash_cuda - output_pytorch).abs() /
                   (output_pytorch.abs()+ 1e-3)).max()
    print('>>> forward half')
    print(f'* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')
    assert(fwdok)

    test_args = edict({'warmup_num': 1000, 'test_num': 1000})
    
    exp_time_dcnv4 = speed_test(DCNv4Function.apply, test_args, dcnv4_args, name='exp')
    exp_time_dcnv3 = speed_test(DCNv3Function.apply, test_args, dcnv3_args, name='exp')
    torch.cuda.synchronize()

    results = [{}]
    results[0]['dcnv3_time'] = exp_time_dcnv3
    results[0]['dcnv4_time'] = exp_time_dcnv4
    columns = list(results[0].keys())

    outputs = pd.DataFrame(results, columns=columns)
    with pd.option_context(
        'display.max_rows', None, 'display.max_columns', None,
        'display.max_colwidth', None, 'display.width', None,
        'display.precision', 4, ):
        print(outputs)


if __name__ == '__main__':
    check_forward_equal_with_pytorch_half()
