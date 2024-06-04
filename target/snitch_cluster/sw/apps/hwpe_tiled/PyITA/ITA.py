# ----------------------------------------------------------------------
#
# File: ITA.py
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

from functools import partial
import os
import sys
from collections.abc import Sequence
from typing import SupportsIndex, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .softmax import fastSoftmax, realSoftmax, streamingPartialSoftmax

STD_DEVS = 4


def random_shuffled_tensor(shape, bitwidth: int, type: DTypeLike = np.int8):
    scale = 2**(bitwidth - 1)
    tensor = np.random.standard_cauchy(size = shape) * scale / STD_DEVS
    tensor = np.clip(tensor, -scale, scale - 1)
    return tensor.astype(type)


class Transformer:
    WO = 26
    WI = 8

    def __init__(self,
                 S: int,
                 P: int,
                 E: int,
                 H: int,
                 path: Union[str, os.PathLike],
                 bias: bool = True,
                 Q: ArrayLike = None,
                 K: ArrayLike = None,
                 V: ArrayLike = None,
                 Wq: ArrayLike = None,
                 Wk: ArrayLike = None,
                 Wv: ArrayLike = None,
                 Wo: ArrayLike = None,
                 Bq: ArrayLike = None,
                 Bk: ArrayLike = None,
                 Bv: ArrayLike = None,
                 Bo: ArrayLike = None):

        self.ITA_N = 16
        self.ITA_M = 64

        # WIESEP: Set numpy print options
        np.set_printoptions(threshold = sys.maxsize)
        np.set_printoptions(linewidth = np.inf)

        self._init_paths(path)

        self.S_ITA = max(64, S)
        self.P_ITA = max(64, P)
        self.E_ITA = max(64, E)
        self.H_ITA = 4
        self.split = self.ITA_M // self.ITA_N

        self.S = S
        self.P = P
        self.E = E
        self.H = H
        self.bias = bias

        self._validate_matrix_constraints(K, V)
        self._initialize_quantization_parameters()
        self._initialize_tensors(Q, V, Wq, Wk, Wv, Wo, Bq, Bk, Bv, Bo)

    def _validate_matrix_constraints(self, K: ArrayLike, V: ArrayLike):
        # WIESEP: Ensure that K is the same as V because we do cross-attention
        assert (np.all(K == V))

        # WIESEP: Current restrictions for ITA
        assert (self.S % self.ITA_M == 0), "Sequence length must be divisible by ITA_M"
        assert (self.P % self.ITA_M == 0), "Projection space must be divisible by ITA_M"
        assert (self.E % self.ITA_M == 0), "Embedding size must be divisible by ITA_M"

        assert (
            self.E <= 512
        ), f"Embedding size must be less than {int(2**(self.WO-17))} because the internal bit width is {self.WO} bits"
        assert (
            self.P <= 512
        ), f"Projection space must be less than {int(2**(self.WO-17))} because the internal bit width is {self.WO} bits"
        assert (
            self.S <= 512
        ), f"Sequence length must be less than {int(2**(self.WO-17))} because the internal bit width is {self.WO} bits"

        # assert (self.H % self.H_ITA == 0 or self.H == 1), "Number of heads must be one or divisible by H_ITA"

    def _initialize_tensors(self, Q, V, Wq, Wk, Wv, Wo, Bq, Bk, Bv, Bo):

        self.exp_sum = np.zeros(self.S, dtype = np.int32)

        self.Q_in = random_shuffled_tensor((self.S, self.E), self.WI - 1) if Q is None else Q
        self.Q = np.pad(self.Q_in, ((0, self.S_ITA - self.S), (0, self.E_ITA - self.E)))

        self.V_in = random_shuffled_tensor((self.S, self.E), self.WI - 1) if V is None else V
        self.V = np.pad(self.V_in, ((0, self.S_ITA - self.S), (0, self.E_ITA - self.E)))

        # WIESEP: K is the same as V because we do cross-attention
        self.K_in = self.V_in
        self.K = self.V

        #### Weight matrices ####
        self.Wq_in = random_shuffled_tensor((self.H, self.E, self.P), self.WI - 1) if Wq is None else Wq
        self.Wq = np.pad(self.Wq_in, ((0, 0), (0, self.E_ITA - self.E), (0, self.P_ITA - self.P)))

        self.Wk_in = random_shuffled_tensor((self.H, self.E, self.P), self.WI - 1) if Wk is None else Wk
        self.Wk = np.pad(self.Wk_in, ((0, 0), (0, self.E_ITA - self.E), (0, self.P_ITA - self.P)))

        self.Wv_in = random_shuffled_tensor((self.H, self.E, self.P), self.WI - 1) if Wv is None else Wv
        self.Wv = np.pad(self.Wv_in, ((0, 0), (0, self.E_ITA - self.E), (0, self.P_ITA - self.P)))

        self.Wo_in = random_shuffled_tensor((self.H, self.P, self.E), self.WI - 1) if Wo is None else Wo
        self.Wo = np.pad(self.Wo_in, ((0, 0), (0, self.P_ITA - self.P), (0, self.E_ITA - self.E)))

        #### Bias matrices ####
        if self.bias:
            self.Bq_in = random_shuffled_tensor(
                (self.H, self.P), int(np.log2(self.P)) + 8, type = np.int32) if Bq is None else Bq
        else:
            self.Bq_in = np.zeros((self.H, self.P), dtype = np.int8)
        self.Bq = np.pad(self.Bq_in, ((0, 0), (0, self.P_ITA - self.P)))
        self.Bq_broadcast = np.reshape(np.repeat(self.Bq, self.S, axis = 0), (self.H, self.S, self.P))

        if self.bias:
            self.Bk_in = random_shuffled_tensor(
                (self.H, self.P), int(np.log2(self.P)) + 8, type = np.int32) if Bk is None else Bk
        else:
            self.Bk_in = np.zeros((self.H, self.P), dtype = np.int8)
        self.Bk = np.pad(self.Bk_in, ((0, 0), (0, self.P_ITA - self.P)))
        self.Bk_broadcast = np.reshape(np.repeat(self.Bk, self.S, axis = 0), (self.H, self.S, self.P))

        if self.bias:
            self.Bv_in = random_shuffled_tensor(
                (self.H, self.P), int(np.log2(self.P)) + 8, type = np.int32) if Bv is None else Bv
        else:
            self.Bv_in = np.zeros((self.H, self.P), dtype = np.int8)
        self.Bv = np.pad(self.Bv_in, ((0, 0), (0, self.P_ITA - self.P)))
        self.Bv_broadcast = np.reshape(np.repeat(self.Bv, self.S, axis = 0), (self.H, self.S, self.P))

        if self.bias:
            self.Bo_in = random_shuffled_tensor(
                (self.H, self.E), int(np.log2(self.E)) + 8, type = np.int32) if Bo is None else Bo
        else:
            self.Bo_in = np.zeros((self.H, self.E), dtype = np.int8)
        self.Bo = np.pad(self.Bo_in, ((0, 0), (0, self.E_ITA - self.E)))
        self.Bo_broadcast = np.reshape(np.repeat(self.Bo, self.S, axis = 0), (self.H, self.S, self.E))

        #### Intermediate tensors ####

        self.Qp = None
        self.Qp_requant = None
        self.Kp = None
        self.Kp_requant = None
        self.Vp = None
        self.Vp_requant = None

        self.A = None
        self.A_requant = None
        self.A_real_softmax = np.zeros([self.H, self.S, self.S], dtype = np.int8)
        self.A_partial_softmax = np.zeros([self.H, self.S, self.S], dtype = np.int8)

        self.O_soft = None
        self.O_soft_requant = None

        self.Out_soft = None
        self.Out_soft_requant = None

        self.Out_soft_sum = None
        self.Out_soft_sum_requant = None

    def _initialize_quantization_parameters(self):
        # WIESEP: 6 steps for attention layer and one to requantize the accumulated output
        self.requant_eps_mult = np.zeros((7, self.H), dtype = np.uint8)
        self.requant_right_shift = np.zeros((7, self.H), dtype = np.uint8)

        # WIESEP: Add parameter in transformers will always be zero as there are no batch normalization layers
        self.requant_add = np.zeros((7, self.H), dtype = np.int8)

        for i in range(7):
            self.requant_eps_mult[i, :] = np.random.randint(64, 127, size = (1, self.H), dtype = np.uint8)

            if i < 3:  # Q, K, V
                max_bit_width = np.log2(self.requant_eps_mult[i, :].astype(np.uint32) * self.E * 2**9).astype(np.uint32)
            elif i == 3:  # QK
                max_bit_width = np.log2(self.requant_eps_mult[i, :].astype(np.uint32) * self.P * 2**8).astype(np.uint32)
            elif i == 4:  # AV
                max_bit_width = np.log2(self.requant_eps_mult[i, :].astype(np.uint32) * self.S * 2**8).astype(np.uint32)
            elif i == 5:  # OW
                max_bit_width = np.log2(self.requant_eps_mult[i, :].astype(np.uint32) * self.E * 2**9).astype(np.uint32)
            elif i == 6:  # Sum OW
                max_bit_width = np.log2(self.requant_eps_mult[i, :].astype(np.uint32) * self.H * 2**7).astype(np.uint32)

            # WIESEP: Last requatization after head summation shares the same parameters
            if i == 6:
                self.requant_right_shift[i, :] = np.tile(max_bit_width[0] - 8 + 2, self.H)
            else:
                self.requant_right_shift[i, :] = max_bit_width - 8 + 2

        self.write_matrix([self.requant_eps_mult.T], "RQS_MUL", self.paths["base"])
        self.write_matrix([self.requant_right_shift.T], "RQS_SHIFT", self.paths["base"])
        self.write_matrix([self.requant_add.T], "RQS_ADD", self.paths["base"])

    def _init_paths(self, base_path: Union[str, os.PathLike]):
        self.paths = {
            "base": base_path,
            "mempool": os.path.join(base_path, "mempool/"),
            "hwpe": os.path.join(base_path, "hwpe/"),
            "standalone": os.path.join(base_path, "standalone/"),
            "snitch-cluster": os.path.join(base_path, "snitch-cluster/")
        }
        for path in self.paths.values():
            os.makedirs(path, exist_ok = True)

    def print_properties(self, verbose: int, text_align = 30):
        if verbose > 0:
            print(f"{'ITA Sequence Length ' :<{text_align}}: {self.S_ITA}")
            print(f"{'ITA Projection Space' :<{text_align}}: {self.P_ITA}")
            print(f"{'ITA Embedding Size  ' :<{text_align}}: {self.E_ITA}")
            print(f"{'ITA Number of Heads ' :<{text_align}}: {self.H_ITA}")
            print(f"{'Matrix Sequence Length ' :<{text_align}}: {self.S}")
            print(f"{'Matrix Projection Space' :<{text_align}}: {self.P}")
            print(f"{'Matrix Embedding Size  ' :<{text_align}}: {self.E}")
            print(f"{'Matrix Number of Heads ' :<{text_align}}: {self.H}")
            print(f"{'Bias ' :<{text_align}}: {bool(self.bias)}")
            print(f"{'Requant Mult ' :<{text_align}}: {list(self.requant_eps_mult)}")
            print(f"{'Requant Shift ' :<{text_align}}: {list(self.requant_right_shift)}")
            print(f"{'Requant Add ' :<{text_align}}: {list(self.requant_add)}")

    @staticmethod
    def write_matrix(matrix: np.ndarray, name: str, path: Union[str, os.PathLike]):
        with open('%s%s.txt' % (path, name), "wb+") as f:
            for row in matrix:
                np.savetxt(f, row, fmt = '%d')
            # Truncate file to remove last newline
            f.seek(-1, os.SEEK_END)
            f.truncate()

    @staticmethod
    def to_hex(val: int, bit_size: int):
        # Function to convert signed integer to hexadecimal representation based on bit size
        if val < 0:
            val += 2**bit_size  # Adjust for negative values based on bit size
        hex_digits = bit_size // 4
        format_string = f'0{hex_digits}x'
        return format(val, format_string)

    @staticmethod
    def pack_hex_08x(row: np.ndarray):
        # Function to pack every four 02x hex values into one 08x hex value
        # Reverse each group of four, then join and pack
        return [''.join(row[i:i + 4][::-1]) for i in range(0, len(row), 4)]

    @staticmethod
    def pack_hex_24b(row: np.ndarray):
        # Function to pack every four 24-bit hex values into three 32-bit hex values
        # Join each group of four and pack
        tmp = [''.join(row[i:i + 4][::-1]) for i in range(0, len(row), 4)]
        # Divide each tmp element into three
        result = ['' for i in range(3 * len(tmp))]
        for i in range(len(tmp)):
            result[3 * i] = tmp[i][16:24]
            result[3 * i + 1] = tmp[i][8:16]
            result[3 * i + 2] = tmp[i][0:8]
        return result

    @staticmethod
    def write_vector_mem_hex(vector: np.ndarray, name: str, path: Union[str, os.PathLike]):
        with open('%s%s.txt' % (path, name), "a+") as f:
            np.savetxt(f, vector, fmt = '%s')

    @staticmethod
    def write_matrix_mem_hex(matrix: np.ndarray, name: str, path: Union[str, os.PathLike]):
        with open('%s%s.txt' % (path, name), "a+") as f:
            for row in matrix:
                np.savetxt(f, row, fmt = '%s')

    @staticmethod
    def generate_matrix_mem(matrix: np.ndarray) -> str:
        return np.array2string(matrix.flatten(), separator = ',', formatter = {'numpystr': lambda x: x})[1:-1]

    @staticmethod
    def write_matrix_mem(matrix: np.ndarray, name: str, path: Union[str, os.PathLike]):
        with open('%s%s.c' % (path, name), "a+") as f:
            for row in matrix:
                np.savetxt(f, row, fmt = '%d', delimiter = ',', newline = ',')

    @staticmethod
    def clip(matrix: np.ndarray) -> np.ndarray:
        result = np.empty(matrix.shape, dtype = np.int8)
        for r_ind, row in enumerate(matrix):
            for c_ind, element in enumerate(row):
                if element > 127:
                    result[r_ind, c_ind] = 127
                elif element < -128:
                    result[r_ind, c_ind] = -128
                else:
                    result[r_ind, c_ind] = element
        return result

    @staticmethod
    def requantize2(matrix: np.ndarray, eps_mult: int, right_shift: int, add: int) -> np.ndarray:
        result = np.empty(matrix.shape, dtype = np.int8)
        for r_ind, row in enumerate(matrix):
            for c_ind, element in enumerate(row):
                # shifted = ((eps_mult * element) >> right_shift) + add
                shifted = ((eps_mult * element) / 2**right_shift) + add
                shifted = np.floor(shifted + 0.5 + np.finfo(np.float32).eps)
                if shifted > 127:
                    result[r_ind, c_ind] = 127
                elif shifted < -128:
                    result[r_ind, c_ind] = -128
                else:
                    result[r_ind, c_ind] = shifted.astype(np.int8)
        return result

    @staticmethod
    def requantize3(matrix: np.ndarray, eps_mult: int, right_shift: int, add: int):
        result = np.empty(matrix.shape, dtype = np.int8)
        for h_ind, heads in enumerate(matrix):
            for r_ind, row in enumerate(heads):
                for c_ind, element in enumerate(row):
                    # shifted = ((eps_mult[h_ind] * element) >> right_shift[h_ind]) + add[h_ind]
                    shifted = ((eps_mult[h_ind] * element) / 2**right_shift[h_ind]) + add[h_ind]
                    shifted = np.floor(shifted + 0.5 + np.finfo(np.float32).eps)
                    if shifted > 127:
                        result[h_ind, r_ind, c_ind] = 127
                    elif shifted < -128:
                        result[h_ind, r_ind, c_ind] = -128
                    else:
                        result[h_ind, r_ind, c_ind] = shifted.astype(np.int8)
        return result

    @staticmethod
    def requantize_step6(matrix: np.ndarray, eps_mult: int, right_shift: int, add: int) -> np.ndarray:
        result = np.full(matrix.shape[1:3], add, dtype = np.int8)
        for _, heads in enumerate(matrix):
            for r_ind, row in enumerate(heads):
                for c_ind, element in enumerate(row):
                    # shifted = ((eps_mult * element) >> right_shift) + result[r_ind, c_ind]
                    shifted = ((eps_mult * element) / 2**right_shift) + result[r_ind, c_ind]
                    shifted = np.floor(shifted + 0.5 + np.finfo(np.float32).eps)
                    if shifted > 127:
                        result[r_ind, c_ind] = 127
                    elif shifted < -128:
                        result[r_ind, c_ind] = -128
                    else:
                        result[r_ind, c_ind] = shifted.astype(np.int8)
        return result

    @staticmethod
    def split_matrix(m: np.ndarray, block_shape: Tuple[SupportsIndex, SupportsIndex]) -> np.ndarray:
        """
        Splits a 2-dimensional numpy array into smaller blocks of a specified shape.

        This function takes a 2D numpy array `m` and divides it into smaller 2D blocks. Each block will have the shape specified by `block_shape`. The array is first reshaped and then transposed to get the desired block layout.

        Parameters:
        m (np.ndarray): A 2-dimensional numpy array to be split into blocks.
        block_shape (Tuple[SupportsIndex, SupportsIndex]): A tuple specifying the shape of the blocks. The first element is the number of rows in each block and the second is the number of columns.

        Returns:
        np.ndarray: A 4D numpy array where each block is accessed by the indices [i, j], where `i` is the block row index and `j` is the block column index.

        Raises:
        ValueError: If the input matrix `m` is not 2-dimensional.

        Example:
        Given a 2D array like this (4x4):

        [[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]]

        and `block_shape` of (2, 2), the function will return a 4D array where each 2x2 block can be accessed separately.
        Illustration:
        Original Matrix:
        +----+----+----+----+
        |  1 |  2 |  3 |  4 |
        +----+----+----+----+
        |  5 |  6 |  7 |  8 |
        +----+----+----+----+
        |  9 | 10 | 11 | 12 |
        +----+----+----+----+
        | 13 | 14 | 15 | 16 |
        +----+----+----+----+

        After splitting into 2x2 blocks:
        +------+------+------+------+
        | [1 2 | [3 4 |   ...   ...
        |  5 6]|  7 8]|
        +------+------+      ...
        | [9 10| [11 12|   ...
        | 13 14| 15 16]|
        +------+------+------+------+

        Each pair of brackets [] represents a separate block.
        """
        if m.ndim == 2:
            return m.reshape((-1, block_shape[0], m.shape[1] // block_shape[1], block_shape[1])).transpose((0, 2, 1, 3))
        else:
            raise ValueError("Matrix must be 2D")

    def tiler_QK(self, qk: np.ndarray, weight: np.ndarray, bias: np.ndarray, output: np.ndarray, input_file: str,
                 weight_file: str, bias_file: str, output_file: str):
        """
        Tile input, weight, bias and output for Q and K generation
        """

        # Weight Wqk is H x E x P
        # Transpose Wqk to H x P x E
        weight = np.transpose(weight, (0, 2, 1))

        tile_x = qk.shape[0] // self.ITA_M  # S // ITA_M
        tile_inner = qk.shape[1] // self.ITA_M  # E // ITA_M
        tile_y = weight.shape[1] // self.ITA_M  # P // ITA_M
        print(f"=> Tile: {input_file} x {weight_file} + {bias_file} = {output_file}")
        print(f"    X: {tile_x}, Y: {tile_y}, Inner: {tile_inner}")

        # Input QK is S x E
        Input = self.split_matrix(qk, (self.ITA_M, self.ITA_M))
        # Repeat each row of each tile split times
        Input = np.tile(Input, [1, 1, self.split, 1])
        # Repeat each tile number of output row tiles times
        Input = np.tile(Input, [1, tile_y, 1, 1]).reshape((-1, self.ITA_M))
        self.write_matrix(Input, input_file, self.paths["standalone"])

        # Transposed Weight Wqk is H x P x E
        for h in range(self.H):
            Weight = self.split_matrix(weight[h], (self.ITA_M, self.ITA_M)).reshape((-1, self.ITA_M))
            # Repeat each tile number of output column tiles times
            Weight = np.tile(Weight, [tile_x, 1])
            self.write_matrix(Weight, f"{weight_file}_{h}", self.paths["standalone"])

        # Bias Bqk is H x P
        # Broadcast Bias Bqk to H x S x P
        bias = np.tile(bias, [1, self.S, 1])
        for h in range(self.H):
            Bias = self.split_matrix(bias[h], (self.ITA_M, self.ITA_N)).reshape((-1, self.ITA_N))
            self.write_matrix(Bias, f"{bias_file}_{h}", self.paths["standalone"])

        # Output QKp is H x S x P
        for h in range(self.H):
            Output = self.split_matrix(output[h], (self.ITA_M, self.ITA_N)).reshape((-1, self.ITA_N))
            self.write_matrix(Output, f"{output_file}_{h}", self.paths["standalone"])

    def tiler_V(self, v, weight, bias, output, input_file, weight_file, bias_file, output_file):
        """
        Tile input, weight, bias and output for V generation
        *Compute Vp in transposed form*
        """

        # Weight Wv is H x E x P
        # Transpose Wv to H x P x E
        weight = np.transpose(weight, (0, 2, 1))

        tile_x = v.shape[0] // self.ITA_M  # S // ITA_M
        tile_inner = v.shape[1] // self.ITA_M  # E // ITA_M
        tile_y = weight.shape[1] // self.ITA_M  # P // ITA_M
        print(f"=> Tile: {input_file} x {weight_file} + {bias_file} = {output_file}")
        print(f"    X: {tile_x}, Y: {tile_y}, Inner: {tile_inner}")

        # Input V is S x E (will be used as second input)
        Input = self.split_matrix(v, (self.ITA_M, self.ITA_M)).reshape((-1, self.ITA_M))
        # Repeat each tile number of output row tiles times
        Input = np.tile(Input, [tile_y, 1])
        self.write_matrix(Input, input_file, self.paths["standalone"])

        # Transposed Weight Wv is H x P x E (will be used as first input)
        for h in range(self.H):
            Weight = self.split_matrix(weight[h], (self.ITA_M, self.ITA_M))
            # Repeat each row of each tile split times
            Weight = np.tile(Weight, [1, 1, self.split, 1])
            # Repeat each tile number of output column tiles times
            Weight = np.tile(Weight, [1, tile_x, 1, 1]).reshape((-1, self.ITA_M))
            self.write_matrix(Weight, f"{weight_file}_{h}", self.paths["standalone"])

        # Bias Bv is H x P
        # Broadcast Bias Bv to H x S x P
        bias = np.tile(bias, [1, self.S, 1])
        # Transpose Bias Bv to H x P x S
        bias = np.transpose(bias, (0, 2, 1))
        for h in range(self.H):
            Bias = self.split_matrix(bias[h], (self.ITA_M, self.ITA_N)).reshape((-1, self.ITA_N))
            self.write_matrix(Bias, f"{bias_file}_{h}", self.paths["standalone"])

        # Output Vp is H x S x P
        # Transpose Vp to H x P x S
        output = np.transpose(output, (0, 2, 1))
        for h in range(self.H):
            Output = self.split_matrix(output[h], (self.ITA_M, self.ITA_N)).reshape((-1, self.ITA_N))
            self.write_matrix(Output, f"{output_file}_{h}", self.paths["standalone"])

    def tiler_AV(self, Qp, Kp, output, input_file, weight_file, output_file):
        """
        Tile input, weight, and output for Q.K = A and A.V = O generation
        """

        tile_x = Qp.shape[1] // self.ITA_M
        tile_inner = Qp.shape[2] // self.ITA_M
        tile_y = Kp.shape[1] // self.ITA_M
        print(f"=> Tile: {input_file} x {weight_file} = {output_file}")
        print(f"    X: {tile_x}, Y: {tile_y}, Inner: {tile_inner}")

        # Input Qp is H x S x P or A is S x S
        for h in range(self.H):
            Input = self.split_matrix(Qp[h], (self.ITA_M, self.ITA_M))
            # Repeat each row of each tile split times
            Input = np.tile(Input, [1, 1, self.split, 1])
            # Repeat each tile number of output row tiles times
            Input = np.tile(Input, [1, tile_y, 1, 1]).reshape((-1, self.ITA_M))
            self.write_matrix(Input, f"{input_file}_{h}", self.paths["standalone"])

        # Weight Kp is H x S x P or V is H x P x S
        for h in range(self.H):
            Weight = self.split_matrix(Kp[h], (self.ITA_M, self.ITA_M)).reshape((-1, self.ITA_M))
            # Repeat each tile number of output column tiles times
            Weight = np.tile(Weight, [tile_x, 1])
            self.write_matrix(Weight, f"{weight_file}_{h}", self.paths["standalone"])

        # Output A is H x S x S or O is H x S x P
        for h in range(self.H):
            Output = self.split_matrix(output[h], (self.ITA_M, self.ITA_N)).reshape((-1, self.ITA_N))
            self.write_matrix(Output, f"{output_file}_{h}", self.paths["standalone"])

    def tiler_Out(self, O, weight, bias, output, input_file, weight_file, bias_file, output_file):
        """
        Tile input, weight, bias and output for Output generation
        Same as QK but takes multi-head input
        """

        # Weight Wo is H x P x E
        # Transpose Wo to H x E x P
        weight = np.transpose(weight, (0, 2, 1))

        tile_x = O.shape[1] // self.ITA_M  # S // ITA_M
        tile_inner = O.shape[2] // self.ITA_M  # P // ITA_M
        tile_y = weight.shape[1] // self.ITA_M  # E // ITA_M

        print(f"=> Tile: {input_file} x {weight_file} + {bias_file} = {output_file}")
        print(f"    X: {tile_x}, Y: {tile_y}, Inner: {tile_inner}")

        # Input O is H x S x P
        for h in range(self.H):
            Input = self.split_matrix(O[h], (self.ITA_M, self.ITA_M))
            # Repeat each row of each tile split times
            Input = np.tile(Input, [1, 1, self.split, 1])
            # Repeat each tile number of output row tiles times
            Input = np.tile(Input, [1, tile_y, 1, 1]).reshape((-1, self.ITA_M))
            self.write_matrix(Input, f"{input_file}_{h}", self.paths["standalone"])

        # Transposed Weight Wo is H x E x P
        for h in range(self.H):
            Weight = self.split_matrix(weight[h], (self.ITA_M, self.ITA_M)).reshape((-1, self.ITA_M))
            # Repeat each tile number of output column tiles times
            Weight = np.tile(Weight, [tile_x, 1])
            self.write_matrix(Weight, f"{weight_file}_{h}", self.paths["standalone"])

        # Bias Bo is H x E
        # Broadcast Bias Bo to H x S x E
        bias = np.tile(bias, [1, self.S, 1])
        for h in range(self.H):
            Bias = self.split_matrix(bias[h], (self.ITA_M, self.ITA_N)).reshape((-1, self.ITA_N))
            self.write_matrix(Bias, f"{bias_file}_{h}", self.paths["standalone"])

        # Output is H x S x E
        for h in range(self.H):
            Output = self.split_matrix(output[h], (self.ITA_M, self.ITA_N)).reshape((-1, self.ITA_N))
            self.write_matrix(Output, f"{output_file}_{h}", self.paths["standalone"])

    def step1_Qp(self):
        self.Qp = np.matmul(self.Q, self.Wq, dtype = np.int32) + self.Bq_broadcast
        self.Qp = np.clip(self.Qp, -2**(self.WO - 1), 2**(self.WO - 1) - 1)
        self.Qp_requant = self.requantize3(self.Qp, self.requant_eps_mult[0], self.requant_right_shift[0],
                                           self.requant_add[0])

        self.tiler_QK(self.Q, self.Wq, self.Bq, self.Qp_requant, "Q", "Wq", "Bq", "Qp")

    def step2_Kp(self):
        self.Kp = np.matmul(self.K, self.Wk, dtype = np.int32) + self.Bk_broadcast
        self.Kp = np.clip(self.Kp, -2**(self.WO - 1), 2**(self.WO - 1) - 1)
        self.Kp_requant = self.requantize3(self.Kp, self.requant_eps_mult[1], self.requant_right_shift[1],
                                           self.requant_add[1])

        self.tiler_QK(self.K, self.Wk, self.Bk, self.Kp_requant, "K", "Wk", "Bk", "Kp")

    def step3_Vp(self):
        self.Vp = np.matmul(self.V, self.Wv, dtype = np.int32) + self.Bv_broadcast
        self.Vp = np.clip(self.Vp, -2**(self.WO - 1), 2**(self.WO - 1) - 1)
        self.Vp_requant = self.requantize3(self.Vp, self.requant_eps_mult[2], self.requant_right_shift[2],
                                           self.requant_add[2])

        # Compute Vp in transposed form
        self.tiler_V(self.V, self.Wv, self.Bv, self.Vp_requant, "V", "Wv", "Bv", "Vp")

    def step4_QK(self, no_partial_softmax):
        self.A = np.array(
            [np.matmul(self.Qp_requant[i], np.transpose(self.Kp_requant[i]), dtype = np.int32) for i in range(self.H)])
        self.A = np.clip(self.A, -2**(self.WO - 1), 2**(self.WO - 1) - 1)
        self.A_requant = self.requantize3(self.A, self.requant_eps_mult[3], self.requant_right_shift[3],
                                          self.requant_add[3])
        self.soft(no_partial_softmax)

        self.tiler_AV(self.Qp_requant, self.Kp_requant, self.A_requant, "Qp_in", "Kp_in", "A")

    def soft(self, no_partial_softmax = False):
        self.A_real_softmax = realSoftmax(self.A_requant)
        if no_partial_softmax:
            self.A_partial_softmax = fastSoftmax(self.A_requant)
        else:
            self.A_partial_softmax = streamingPartialSoftmax(self.A_requant)

        if self.H == 1:
            A_save = [np.tile(self.A_partial_softmax[i], [self.split, 1]) for i in range(self.H)]
            self.write_matrix(A_save, "A_soft_in", self.paths["standalone"])
        for h in range(self.H):
            A_save = self.A_partial_softmax[h]
            self.write_matrix(A_save, f"A_soft_{h}", self.paths["standalone"])

    def step5_AV(self):
        self.O_soft = np.array(
            [np.matmul(self.A_partial_softmax[i], self.Vp_requant[i], dtype = np.int32) for i in range(self.H)])
        self.O_soft = np.clip(self.O_soft, -2**(self.WO - 1), 2**(self.WO - 1) - 1)
        self.O_soft_requant = self.requantize3(self.O_soft, self.requant_eps_mult[4], self.requant_right_shift[4],
                                               self.requant_add[4])

        self.tiler_AV(self.A_requant, np.transpose(self.Vp_requant, (0, 2, 1)), self.O_soft_requant, "A_stream_soft_in",
                      "Vp_in", "O_soft")

    def step6_O(self):
        self.Out_soft = np.matmul(self.O_soft_requant, self.Wo, dtype = np.int32) + self.Bo_broadcast
        self.Out_soft = np.clip(self.Out_soft, -2**(self.WO - 1), 2**(self.WO - 1) - 1)
        self.Out_soft_requant = self.requantize3(self.Out_soft, self.requant_eps_mult[5], self.requant_right_shift[5],
                                                 self.requant_add[5])

        self.tiler_Out(self.O_soft_requant, self.Wo, self.Bo, self.Out_soft_requant, "O_soft_in", "Wo", "Bo",
                       "Out_soft")

    def step7_Osum(self):
        self.Out_soft_sum = np.sum(self.Out_soft_requant, axis = 0, dtype = np.int32, keepdims = True)
        self.Out_soft_sum_requant = self.requantize3(self.Out_soft_sum, self.requant_eps_mult[6],
                                                     self.requant_right_shift[6], self.requant_add[6])

    def write_mem_hex(self):
        path = self.paths["hwpe"]

        def remove_if_exists(file_name):
            if os.path.exists(file_name):
                os.remove(file_name)

        # WIESEP: Delete the old file otherwise it will lead to mismatches during RTL simulations as the files are memory mapped
        files = ["mem.txt", "Output.txt", "Q.txt", "K.txt", "V.txt", "QK.txt", "A.txt", "AV.txt", "OW.txt"]
        for file in files:
            remove_if_exists(f"{path}/{file}")

        def split_2Dmatrix(m: np.ndarray):
            return m.reshape((-1, self.ITA_M, m.shape[1] // self.ITA_M, self.ITA_M)).transpose((0, 2, 1, 3)).reshape(
                (-1, self.ITA_M))

        # Write the new mem file
        for h in range(self.H):
            q = split_2Dmatrix(self.Q)
            q_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 8))(q)
            packed_q_hex = np.array([self.pack_hex_08x(row) for row in q_hex])
            self.write_matrix_mem_hex(packed_q_hex, "mem", path)

            k = split_2Dmatrix(self.K)
            k_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 8))(k)
            packed_k_hex = np.array([self.pack_hex_08x(row) for row in k_hex])
            self.write_matrix_mem_hex(packed_k_hex, "mem", path)

            w1 = split_2Dmatrix(np.transpose(self.Wq[h]))
            w1_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 8))(w1)
            packed_w1_hex = np.array([self.pack_hex_08x(row) for row in w1_hex])
            self.write_matrix_mem_hex(packed_w1_hex, "mem", path)

            w2 = split_2Dmatrix(np.transpose(self.Wk[h]))
            w2_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 8))(w2)
            packed_w2_hex = np.array([self.pack_hex_08x(row) for row in w2_hex])
            self.write_matrix_mem_hex(packed_w2_hex, "mem", path)

            w3 = split_2Dmatrix(np.transpose(self.Wv[h]))
            w3_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 8))(w3)
            packed_w3_hex = np.array([self.pack_hex_08x(row) for row in w3_hex])
            self.write_matrix_mem_hex(packed_w3_hex, "mem", path)

            w4 = split_2Dmatrix(np.transpose(self.Wo[h]))
            w4_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 8))(w4)
            packed_w4_hex = np.array([self.pack_hex_08x(row) for row in w4_hex])
            self.write_matrix_mem_hex(packed_w4_hex, "mem", path)

            b1_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 24))(self.Bq[h])
            # pack 24-bit values into 32-bit words
            packed_b1_hex = np.array(self.pack_hex_24b(b1_hex))
            self.write_vector_mem_hex(packed_b1_hex, "mem", path)

            b2_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 24))(self.Bk[h])
            # pack 24-bit values into 32-bit words
            packed_b2_hex = np.array(self.pack_hex_24b(b2_hex))
            self.write_vector_mem_hex(packed_b2_hex, "mem", path)

            b3_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 24))(self.Bv[h])
            # pack 24-bit values into 32-bit words
            packed_b3_hex = np.array(self.pack_hex_24b(b3_hex))
            self.write_vector_mem_hex(packed_b3_hex, "mem", path)

            b4_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 24))(self.Bo[h])
            # pack 24-bit values into 32-bit words
            packed_b4_hex = np.array(self.pack_hex_24b(b4_hex))
            self.write_vector_mem_hex(packed_b4_hex, "mem", path)

            # Write output
            qp = split_2Dmatrix(self.Qp_requant[h])
            Qp_requant_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 8))(qp)
            packed_Qp_requant_hex = np.array([self.pack_hex_08x(row) for row in Qp_requant_hex])
            self.write_matrix_mem_hex(packed_Qp_requant_hex, "Q", path)

            kp = split_2Dmatrix(self.Kp_requant[h])
            Kp_requant_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 8))(kp)
            packed_Kp_requant_hex = np.array([self.pack_hex_08x(row) for row in Kp_requant_hex])
            self.write_matrix_mem_hex(packed_Kp_requant_hex, "K", path)

            v = split_2Dmatrix(np.transpose(self.Vp_requant[h]))
            Vp_requant_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 8))(v)
            packed_Vp_requant_hex = np.array([self.pack_hex_08x(row) for row in Vp_requant_hex])
            self.write_matrix_mem_hex(packed_Vp_requant_hex, "V", path)

            qk = split_2Dmatrix(self.A_requant[h])
            QK_requant_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 8))(qk)
            packed_QK_requant_hex = np.array([self.pack_hex_08x(row) for row in QK_requant_hex])
            self.write_matrix_mem_hex(packed_QK_requant_hex, "QK", path)

            a = split_2Dmatrix(self.A_partial_softmax[h])
            A_requant_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 8))(a)
            packed_A_requant_hex = np.array([self.pack_hex_08x(row) for row in A_requant_hex])
            self.write_matrix_mem_hex(packed_A_requant_hex, "A", path)

            o = split_2Dmatrix(self.O_soft_requant[h])
            O_requant_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 8))(o)
            packed_O_requant_hex = np.array([self.pack_hex_08x(row) for row in O_requant_hex])
            self.write_matrix_mem_hex(packed_O_requant_hex, "AV", path)

            out = split_2Dmatrix(self.Out_soft_requant[h])
            Out_requant_hex = np.vectorize(lambda val: self.to_hex(val, bit_size = 8))(out)
            packed_Out_requant_hex = np.array([self.pack_hex_08x(row) for row in Out_requant_hex])
            self.write_matrix_mem_hex(packed_Out_requant_hex, "OW", path)

    def generate_snitch_cluster(self) -> str:
        ret = ""

        ret += f"""/* This file is automatically generated by '{" ".join(sys.argv)}'
* Do not edit manually, any manual change will be overwritten.
*/

// clang-format off
"""
        ita_split_matrix = partial(Transformer.split_matrix, block_shape=(self.ITA_M, self.ITA_M))

        def generate_array(array, name, _type):
            return f"const {_type} {name}[{array.size}] = {{\n{self.generate_matrix_mem(array)}\n}};\n"

        ret += generate_array(ita_split_matrix(self.Q), "input_q", "int8_t")
        ret += generate_array(ita_split_matrix(self.K), "input_k", "int8_t")

        def ita_split_multihead_matrix(multihead_array):
            return [ita_split_matrix(array) for array in multihead_array]

        def generate_multihead_array(multihead_array, name, _type):
            ret = ""
            ret += f"const {_type} {name}[{self.H}][{multihead_array[0].size}] = {{\n"
            ret += ",\n".join([f"{{\n{self.generate_matrix_mem(array)}\n}}" for array in multihead_array])
            ret += "\n};\n"
            return ret

        ret += generate_multihead_array(ita_split_multihead_matrix(self.Wq.transpose(0, 2, 1)), "input_Wq", "int8_t")
        ret += generate_multihead_array(ita_split_multihead_matrix(self.Wk.transpose(0, 2, 1)), "input_Wk", "int8_t")
        ret += generate_multihead_array(ita_split_multihead_matrix(self.Wv.transpose(0, 2, 1)), "input_Wv", "int8_t")
        ret += generate_multihead_array(ita_split_multihead_matrix(self.Wo.transpose(0, 2, 1)), "input_Wo", "int8_t")

        ret += generate_multihead_array(self.Bq, "input_Bq", "ita_int24_t")
        ret += generate_multihead_array(self.Bk, "input_Bk", "ita_int24_t")
        ret += generate_multihead_array(self.Bv, "input_Bv", "ita_int24_t")
        ret += generate_multihead_array(self.Bo, "input_Bo", "ita_int24_t")

        # Requantization
        def pack_8b_to_word(array):
            ret = []
            for i in range(0, len(array), 4):
                ret.append((array[i] & 0xff) | ((array[i+1] & 0xff) << 8) | ((array[i+2] & 0xff) << 16) | ((array[i+3] & 0xff) << 24))
            return np.array(ret)

        def requant_multihead_harmonization_and_pack_8b(requant_array):
            ret = []
            for i in range(self.H):
                ret.append(pack_8b_to_word(np.pad(requant_array[:6, i], (0, 2))))
            return np.array(ret)

        ret += generate_multihead_array(requant_multihead_harmonization_and_pack_8b(self.requant_eps_mult), "requant_eps_mult", "int32_t")
        ret += generate_multihead_array(requant_multihead_harmonization_and_pack_8b(self.requant_right_shift), "requant_right_shift", "int32_t")
        ret += generate_multihead_array(requant_multihead_harmonization_and_pack_8b(self.requant_add), "requant_add", "int32_t")

        ret += generate_multihead_array(ita_split_multihead_matrix(self.Qp_requant), "golden_interm_Pq", "int8_t")
        ret += generate_multihead_array(ita_split_multihead_matrix(self.Kp_requant), "golden_interm_Pk", "int8_t")
        ret += generate_multihead_array(ita_split_multihead_matrix(self.Vp_requant.transpose((0, 2, 1))), "golden_interm_Pv", "int8_t")
        ret += generate_multihead_array(ita_split_multihead_matrix(self.A_requant), "golden_interm_attention", "int8_t")
        ret += generate_multihead_array(ita_split_multihead_matrix(self.O_soft_requant), "golden_interm_head_output", "int8_t")
        ret += generate_multihead_array(ita_split_multihead_matrix(self.Out_soft_requant), "golden_output", "int8_t")

        ret += "\n"

        def generate_define(name, value):
            return f"#define {name.upper()} {value}\n"

        ret += generate_define("heads", self.H)
        ret += generate_define("sequence_length", self.S)
        ret += generate_define("embedding_space", self.E)
        ret += generate_define("projection_space", self.P)
        ret += generate_define("n_tile_sequence_length", self.S // 64)
        ret += generate_define("n_tile_embedding_space", self.E // 64)
        ret += generate_define("n_tile_projection_space", self.P // 64)
        ret += generate_define("tile_size_sequence_length", 64)
        ret += generate_define("tile_size_embedding_space", 64)
        ret += generate_define("tile_size_projection_space", 64)

        ret += '\n// clang-format on\n'

        return ret

    def write_snitch_cluster(self, path, filename = "mem_snitch_cluster.h"):
        if path == './':
            path = self.paths["snitch-cluster"]

        print(f"=> Exporting memory file to '{path}'")

        with open(os.path.join(path, filename), "w") as f:
            f.write(self.generate_snitch_cluster())

    def write_mempool_l2(self, path):
        if path == './':
            path = self.paths["mempool"]

        print(f"=> Exporting memory file to '{path}'")

        requant_eps_mult = np.pad(self.requant_eps_mult[:6, :].T, ((0, 0), (0, 2)), mode = "constant")
        requant_right_shift = np.pad(self.requant_right_shift[:6, :].T, ((0, 0), (0, 2)), mode = "constant")
        requant_add = np.pad(self.requant_add[:6, :].T, ((0, 0), (0, 2)), mode = "constant")

        with open('%s%s.c' % (path, "mem"), "w+") as f:
            f.write(f"""/* This file is automatically generated by '{" ".join(sys.argv)}'
* Do not edit manually, any manual change will be overwritten.
*/

// clang-format off
""")

        with open('%s%s.c' % (path, "mem"), "a+") as f:
            f.write('#include <stdint.h>\n')
            f.write(f'\nconst uint8_t Requant_Mult[{self.H}][{requant_eps_mult[0].size}] = ' + '{')
        self.write_matrix_mem([requant_eps_mult], "mem", path)

        with open('%s%s.c' % (path, "mem"), "a+") as f:
            f.write('};' + f'\nconst uint8_t Requant_Shift[{self.H}][{requant_right_shift[0].size}] = ' + '{')
        self.write_matrix_mem([requant_right_shift], "mem", path)

        with open('%s%s.c' % (path, "mem"), "a+") as f:
            f.write('};' + f'\nconst int8_t Requant_Add[{self.H}][{requant_add[0].size}] = ' + '{')
        self.write_matrix_mem([requant_add], "mem", path)

        with open('%s%s.c' % (path, "mem"), "a+") as f:
            f.write('};\n\n')

        for h in range(self.H):
            with open('%s%s.c' % (path, "mem"), "a+") as f:
                f.write(f'const int8_t inputs_{h}[] __attribute__((aligned(0x1000))) = ' + '{\n')

            w4 = np.concatenate([np.transpose(self.Wo[h])])
            self.write_matrix_mem(w4, "mem", path)

            w3 = np.concatenate([np.transpose(self.Wv[h])])
            self.write_matrix_mem(w3, "mem", path)

            w2 = np.concatenate([np.transpose(self.Wk[h])])
            self.write_matrix_mem(w2, "mem", path)

            q = np.concatenate(np.split(self.Q, self.split, axis = 1))
            self.write_matrix_mem(q, "mem", path)

            k = np.concatenate(np.split(self.K, self.split, axis = 1))
            self.write_matrix_mem(k, "mem", path)

            # w1 = np.concatenate([np.transpose(self.Wq[i]) for i in range(self.H)])
            w1 = np.concatenate(np.split(np.concatenate([np.transpose(self.Wq[h])]), self.split, axis = 1))
            self.write_matrix_mem(w1, "mem", path)

            b4 = np.reshape(np.split(self.Bo_broadcast[h], self.split, axis = 1), (self.S_ITA, self.E_ITA))
            self.write_matrix_mem(b4, "mem", path)

            b3 = np.reshape(
                np.split(np.reshape(np.transpose(self.Bv_broadcast[h]), (self.P_ITA, self.S_ITA)), self.split,
                         axis = 1), (self.P_ITA, self.S_ITA))
            self.write_matrix_mem(b3, "mem", path)

            b2 = np.reshape(np.split(self.Bk_broadcast[h], self.split, axis = 1), (self.S_ITA, self.P_ITA))
            self.write_matrix_mem(b2, "mem", path)

            b1 = np.reshape(np.split(self.Bq_broadcast[h], self.split, axis = 1), (self.S_ITA, self.P_ITA))
            self.write_matrix_mem(b1, "mem", path)

            with open('%s%s.c' % (path, "mem"), "ab+") as f:
                f.seek(-1, os.SEEK_END)
                f.truncate()
            with open('%s%s.c' % (path, "mem"), "a+") as f:
                f.write('\n};\n\n')

        with open('%s%s.c' % (path, "mem"), "a+") as f:
            f.write('\n// clang-format on\n')
            tot_bytes = np.size(self.Q) + np.size(self.K) + np.size(self.Wq) + np.size(self.Bq_broadcast) \
                        + np.size(self.Wk) + np.size(self.Bk_broadcast) + np.size(self.Wv) + np.size(self.Bv_broadcast) + \
                        np.size(self.Wo) + np.size(self.Bo_broadcast)

            tot_params = tot_bytes = np.size(self.Q) + np.size(self.K) + np.size(self.Wq) + np.size(self.Bq) \
                        + np.size(self.Wk) + np.size(self.Bk) + np.size(self.Wv) + np.size(self.Bv) + \
                        np.size(self.Wo) + np.size(self.Bo)

        print(f"{'Number of Bytes' :<{30}}: {tot_bytes} ({tot_bytes/1024} kB)")
        print(f"{'Number of Parameters' :<{30}}: {tot_params} ({tot_params/1000} k)")

    def write_numpy(self):
        assert np.all(np.equal(self.K, self.V)), "For ITA, keys and values have to be equal"
        q = self.Q
        k = self.K
        # WIESEP: Hacky temporary solution to make sure inputs are in the right format. Should be handled by DumpO!
        # q = np.reshape(np.concatenate(np.split(self.Q, self.split, axis = 1)), (self.S_ITA, self.E_ITA))
        # k = np.reshape(np.concatenate(np.split(self.K, self.split, axis = 1)), (self.S_ITA, self.E_ITA))
        w1 = self.Wq_in
        b1 = self.Bq_in
        w2 = self.Wk_in
        b2 = self.Bk_in
        w3 = self.Wv_in
        b3 = self.Bv_in
        w4 = self.Wo_in
        b4 = self.Bo_in
        o = self.Out_soft_requant[:, :self.S, :self.E]
        o_sum = self.Out_soft_sum_requant[:, :self.S, :self.E]
        np.savez('%s%s.npz' % (self.paths["base"], "mha"),
                 q = q,
                 k = k,
                 w1 = w1,
                 b1 = b1,
                 w2 = w2,
                 b2 = b2,
                 w3 = w3,
                 b3 = b3,
                 w4 = w4,
                 b4 = b4,
                 o = o,
                 o_sum = o_sum,
                 rqs_mult = self.requant_eps_mult,
                 rqs_shift = self.requant_right_shift,
                 rqs_add = self.requant_add)


def generateTestVectors(path, **kwargs):
    s = kwargs['S']
    p = kwargs['P']
    e = kwargs['E']
    h = kwargs['H']
    bias = int(not kwargs['no_bias'])

    acc1 = Transformer(s, p, e, h, bias = bias, path = path)

    if kwargs['verbose']:
        print("=> Generating test vectors...")
    acc1.print_properties(kwargs['verbose'])
    acc1.step1_Qp()
    acc1.step2_Kp()
    acc1.step3_Vp()
    acc1.step4_QK(kwargs['no_partial_softmax'])
    acc1.step5_AV()
    acc1.step6_O()
    acc1.step7_Osum()

    acc1.write_mempool_l2(kwargs['mem_path'])
    acc1.write_snitch_cluster(kwargs['mem_path'])
    acc1.write_mem_hex()
    acc1.write_numpy()

    def print_tensor_stats(tensor):
        print(f"    Min: {np.min(tensor)}")
        print(f"    Max: {np.max(tensor)}")

        # Calculate the simmilarty of elements witin one row and over all comumns
        similarity_row = np.mean(np.abs(np.diff(tensor, axis = -2)))
        similarity_column = np.mean(np.abs(np.diff(tensor, axis = -1)))

        print(f"    Mean-Squared Difference (row)   : {similarity_row:5.1f}")
        print(f"    Mean-Squared Difference (column): {similarity_column:5.1f}")

    if kwargs['verbose'] > 1:
        print("=> Qp")
        print_tensor_stats(acc1.Qp_requant)
        if kwargs['verbose'] > 4:
            print(acc1.Qp)
        if kwargs['verbose'] > 3:
            print(acc1.Qp_requant)

        print("=> Kp")
        print_tensor_stats(acc1.Kp_requant)
        if kwargs['verbose'] > 4:
            print(acc1.Kp)
        if kwargs['verbose'] > 3:
            print(acc1.Kp_requant)

        print("=> Vp")
        print_tensor_stats(acc1.Vp_requant)
        if kwargs['verbose'] > 4:
            print(acc1.Vp)
        if kwargs['verbose'] > 3:
            print(acc1.Vp_requant)

        print("=> A")
        print_tensor_stats(acc1.A_requant)
        if kwargs['verbose'] > 4:
            print(acc1.A)
        if kwargs['verbose'] > 3:
            print(acc1.A_requant)

        print("=> A (partial softmax)")
        print_tensor_stats(acc1.A_partial_softmax)
        if kwargs['verbose'] > 3:
            print(acc1.A_partial_softmax)

        print("=> O (soft)")
        print_tensor_stats(acc1.O_soft_requant)
        if kwargs['verbose'] > 4:
            print(acc1.O_soft)
        if kwargs['verbose'] > 3:
            print(acc1.O_soft_requant)

        print("=> Output (all heads)")
        print_tensor_stats(acc1.Out_soft_requant)
        if kwargs['verbose'] > 3:
            print(acc1.Out_soft_requant)

        print("=> Output (accumulated)")
        print_tensor_stats(acc1.Out_soft_sum_requant)
        if kwargs['verbose'] > 3:
            print(acc1.Out_soft_sum_requant)

    if kwargs['plot_tensors']:
        # Plot distribution of all input and output tensors
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.gridspec import GridSpec

        def plot_distribution(tensor, title, ax):
            sns.histplot(tensor.flatten(), bins = 50, kde = True, ax = ax)
            ax.set_title(title)

        # Plot color values of all tensors
        def plot_heatmap(tensor, title, ax):
            # If tensor is more than 2D, only plot the first 2D
            if len(tensor.shape) > 2:
                tensor = tensor[0]

            sns.heatmap(tensor, ax = ax, cbar = False)
            # Do not show ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)

        # Create sublots
        fig = plt.figure(figsize = (12, 12), layout = 'tight', dpi = 72)

        gs = GridSpec(8, 12, figure = fig)

        ax = fig.add_subplot(gs[0, 0:3])
        plot_distribution(acc1.Q, "Q", ax)
        ax = fig.add_subplot(gs[0, 3:6])
        plot_heatmap(acc1.Q, "Q", ax)
        ax = fig.add_subplot(gs[0, 6:9])
        plot_distribution(acc1.K, "K", ax)
        ax = fig.add_subplot(gs[0, 9:12])
        plot_heatmap(acc1.K, "K", ax)

        ax = fig.add_subplot(gs[1, 0:3])
        plot_distribution(acc1.Wq, "Wq", ax)
        ax = fig.add_subplot(gs[1, 3:6])
        plot_distribution(acc1.Wk, "Wk", ax)
        ax = fig.add_subplot(gs[1, 6:9])
        plot_distribution(acc1.Wv, "Wv", ax)
        ax = fig.add_subplot(gs[1, 9:12])
        plot_distribution(acc1.Wo, "Wo", ax)

        ax = fig.add_subplot(gs[2, 0:3])
        plot_heatmap(acc1.Wq, "Wq", ax)
        ax = fig.add_subplot(gs[2, 3:6])
        plot_heatmap(acc1.Wk, "Wk", ax)
        ax = fig.add_subplot(gs[2, 6:9])
        plot_heatmap(acc1.Wv, "Wv", ax)
        ax = fig.add_subplot(gs[2, 9:12])
        plot_heatmap(acc1.Wo, "Wo", ax)

        ax = fig.add_subplot(gs[3, 0:3])
        plot_distribution(acc1.Bq, "Bq", ax)
        ax = fig.add_subplot(gs[3, 3:6])
        plot_distribution(acc1.Bk, "Bk", ax)
        ax = fig.add_subplot(gs[3, 6:9])
        plot_distribution(acc1.Bv, "Bv", ax)
        ax = fig.add_subplot(gs[3, 9:12])
        plot_distribution(acc1.Bo, "Bo", ax)

        ax = fig.add_subplot(gs[4, 0:3])
        plot_distribution(acc1.Qp_requant, "Qp", ax)
        ax = fig.add_subplot(gs[4, 3:6])
        plot_distribution(acc1.Kp_requant, "Kp", ax)
        ax = fig.add_subplot(gs[4, 6:9])
        plot_distribution(acc1.Vp_requant, "Vp", ax)

        ax = fig.add_subplot(gs[5, 0:3])
        plot_heatmap(acc1.Qp_requant, "Qp", ax)
        ax = fig.add_subplot(gs[5, 3:6])
        plot_heatmap(acc1.Kp_requant, "Kp", ax)
        ax = fig.add_subplot(gs[5, 6:9])
        plot_heatmap(acc1.Vp_requant, "Vp", ax)

        ax = fig.add_subplot(gs[6, 0:3])
        plot_distribution(acc1.A_requant, "QK", ax)
        ax = fig.add_subplot(gs[6, 3:6])
        plot_distribution(acc1.A_partial_softmax, "A", ax)
        ax = fig.add_subplot(gs[6, 6:9])
        plot_distribution(acc1.O_soft_requant, "O", ax)
        ax = fig.add_subplot(gs[6, 9:12])
        plot_distribution(acc1.Out_soft_requant, "Out", ax)

        ax = fig.add_subplot(gs[7, 0:3])
        plot_heatmap(acc1.A_requant, "QK", ax)
        ax = fig.add_subplot(gs[7, 3:6])
        plot_heatmap(acc1.A_partial_softmax, "A", ax)
        ax = fig.add_subplot(gs[7, 6:9])
        plot_heatmap(acc1.O_soft_requant, "O", ax)
        ax = fig.add_subplot(gs[7, 9:12])
        plot_heatmap(acc1.Out_soft_requant, "Out", ax)

        plt.show()


def util_main(**kwargs):
    B = 8
    log2e = np.log2(np.exp(1))
    eps_max = B / (2**B)

    N = 1024
    A = np.random.randint(-128, 127, size = (1, N, N), dtype = np.int8)
    input_float = A * eps_max  # Assume eps is eps_max
    input_int = A

    fast_softmax = fastSoftmax(input_float, False)
    fast_integer_softmax = fastSoftmax(input_int, True) / 255

    fast_partial_softmax = streamingPartialSoftmax(input_float, False)
    fast_partial_integer_softmax = streamingPartialSoftmax(input_int, True) / 255

    softmax = realSoftmax(input_float, False)
    integer_softmax = realSoftmax(input_int, True) / 255

    print(f"=> L2 Softmax Differences:")
    print(
        f"  Softmax              - Fast Softmax                    : {np.linalg.norm((softmax-fast_softmax)[0], 2):.10}"
    )
    print(
        f"  Softmax              - Fast Partial Softmax            : {np.linalg.norm((softmax-fast_partial_softmax)[0], 2):.10}"
    )
    print(
        f"  Softmax              - Fast Integer Softmax            : {np.linalg.norm((softmax-fast_integer_softmax)[0], 2):.10}"
    )
    print(
        f"  Softmax              - Fast Partial Integer Softmax    : {np.linalg.norm((softmax-fast_partial_integer_softmax)[0], 2):.10}"
    )
    # print(f"  Integer Softmax      - Fast Integer Softmax            : {np.linalg.norm((integer_softmax-fast_integer_softmax)[0], 2):.3}")
    # print(f"  Integer Softmax      - Fast Partial Integer Softmax    : {np.linalg.norm((integer_softmax-fast_partial_integer_softmax)[0], 2):.3}")
    # print(f"  Softmax              - Integer Softmax                 : {np.linalg.norm((integer_softmax-softmax)[0], 2):.3}")
    # print(f"  Fast Softmax         - Fast Partial Softmax            : {np.linalg.norm((fast_softmax-fast_partial_softmax)[0], 2):.3}")
    # print(f"  Fast Integer Softmax - Fast Partial Integer Softmax    : {np.linalg.norm((fast_integer_softmax-fast_partial_integer_softmax)[0], 2):.3}")

    TEST_QUANTLIB = True
    if TEST_QUANTLIB:
        import torch

        from quantlib.algorithms.pact.pact_ops import (PACTIntegerITAMax, PACTIntegerITAPartialMax, PACTITAMax,
                                                       PACTITAPartialMax)
        input = torch.tensor(input_float).unsqueeze(0).float()

        ITAMax = PACTITAMax()
        ITAPartialMax = PACTITAPartialMax(ita_sequence_length = N)
        ITAmax_softmax = ITAMax.forward(input).detach().numpy().squeeze(axis = 0)
        ITApartialmax_softmax = ITAPartialMax.forward(input).detach().numpy().squeeze(axis = 0)

        ITAMax.started = torch.tensor(1)
        ITAPartialMax.started = torch.tensor(1)
        ITAMax.set_eps_in(torch.tensor((eps_max,)))
        ITAPartialMax.set_eps_in(torch.tensor((eps_max,)))
        ITAMax_integer_softmax = ITAMax.forward(input).detach().numpy().squeeze(axis = 0)
        ITAPartialMax_integer_softmax = ITAPartialMax.forward(input).detach().numpy().squeeze(axis = 0)

        input = torch.tensor(input_int).unsqueeze(0).float()
        ITAIntegerMax_softmax = PACTIntegerITAMax.MySoftmax.forward(
            None, input, torch.tensor(256)).detach().numpy().squeeze(axis = 0)
        ITAPartialIntegerMax_softmax = PACTIntegerITAMax.MySoftmax.forward(
            None, input, torch.tensor(256)).detach().numpy().squeeze(axis = 0)

        print()
        print(f"=> L2 PyTorch Softmax Differences:")
        print(
            f"  Fast Softmax                 - ITAmax                       : {np.linalg.norm((fast_softmax-ITAmax_softmax)[0], 2):.3}"
        )
        print(
            f"  Fast Partial Softmax         - ITAPartialMax                : {np.linalg.norm((fast_partial_softmax-ITApartialmax_softmax)[0], 2):.3}"
        )
        print(
            f"  Fast Integer Softmax         - Fake-Quantized ITAmax        : {np.linalg.norm((fast_integer_softmax-ITAMax_integer_softmax)[0], 2):.3}"
        )
        print(
            f"  Fast Integer Partial Softmax - Fake-Quantized ITAPartialMax : {np.linalg.norm((fast_partial_integer_softmax-ITAPartialMax_integer_softmax)[0], 2):.3}"
        )
        print(
            f"  Fast Integer Softmax         - True-Quantized ITAmax        : {np.linalg.norm((fast_integer_softmax-ITAIntegerMax_softmax/255)[0], 2):.3}"
        )
        print(
            f"  Fast Integer Partial Softmax - True-Quantized ITAPartialMax : {np.linalg.norm((fast_partial_integer_softmax-ITAPartialIntegerMax_softmax/255)[0], 2):.3}"
        )
