# ----------------------------------------------------------------------
#
# File: testGenerator.py
#
# Last edited: 5.03.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese (wiesep@iis.ee.ethz.ch), ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import numpy as np
import onnx

import PyITA as ITA


def create_initializer_tensor(name: str,
                              tensor_array: np.ndarray,
                              data_type: onnx.TensorProto = onnx.TensorProto.INT8) -> onnx.TensorProto:

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(name = name,
                                                 data_type = data_type,
                                                 dims = tensor_array.shape,
                                                 vals = tensor_array.flatten().tolist())

    return initializer_tensor


def generateMHA(**args):

    current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

    S = args['S']
    P = args['P']
    E = args['E']
    H = args['H']
    NO_BIAS = args['no_bias']
    NO_PARTIAL_SOFTMAX = args['no_partial_softmax']

    if NO_PARTIAL_SOFTMAX:
        path = f'{current_dir}/simvectors/data_S{S}_E{E}_P{P}_H{H}_B{int(not NO_BIAS)}_noPartialSoftmax/'
    else:
        path = f'{current_dir}/simvectors/data_S{S}_E{E}_P{P}_H{H}_B{int(not NO_BIAS)}/'
    os.makedirs(path, exist_ok = True)

    ITA.generateTestVectors(path, **args)

    ITA.exportONNX(path, **args)


class TestParser(argparse.ArgumentParser):

    def __init__(self, description = None):

        class ArgumentDefaultMetavarTypeFormatter(argparse.ArgumentDefaultsHelpFormatter,
                                                  argparse.MetavarTypeHelpFormatter):
            pass

        formatter = ArgumentDefaultMetavarTypeFormatter
        if description is None:
            super().__init__(description = "Test generator for ITA.", formatter_class = formatter)
        else:
            super().__init__(description = description, formatter_class = formatter)

        self.add_argument('--util', action = 'store_true', help = 'Run utility function instead of test generator')

        self.add_argument('--mem-path',
                          type = str,
                          dest = 'mem_path',
                          default = './',
                          help = 'Path to store C memory file')
        self.add_argument('--seed', default = 0, type = int, help = 'Random seed')
        self.add_argument('-v',
                          action = 'count',
                          dest = 'verbose',
                          default = 0,
                          help = 'Set whether to verbose or not n')
        self.add_argument('-p', '--plot-tensors', action = 'store_true', help = 'Plot tensor distributions')

        self.group1 = self.add_argument_group('MHA Settings')
        self.group1.add_argument('-B', default = 1, type = int, help = 'Number of batches')
        self.group1.add_argument('-S', default = 64, type = int, help = 'Sequence length')
        self.group1.add_argument('-E', default = 64, type = int, help = 'Embedding size')
        self.group1.add_argument('-P', default = 64, type = int, help = 'Projection size')
        self.group1.add_argument('-H', default = 1, type = int, help = 'Number of heads')
        self.group1.add_argument('--no-partial-softmax',
                                 action = 'store_true',
                                 help = 'Disable partial softmax calculation')
        self.group1.add_argument('--no-bias', default = False, action = 'store_true', help = 'Disable bias')


if __name__ == "__main__":
    parser = TestParser()
    args = parser.parse_args()
    if args.seed != -1:
        np.random.seed(args.seed)

    if args.util:
        ITA.util_main(**vars(args))
        exit()

    generateMHA(**vars(args))
