# ----------------------------------------------------------------------
#
# File: ita.py
#
# Last edited: 4.06.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Authors:
# - Philip Wiese (wiesep@iis.ee.ethz.ch), ETH Zurich
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

import os
from vcd.reader import TokenKind, tokenize
import pandas as pd
from pprint import pprint

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_vcd(file_path):
    signal_names = {}
    signals = {}
    current_time = 0

    with open(file_path, 'rb') as f:
        tokens = tokenize(f)
        for token in tokens:
            if token.kind is TokenKind.VAR:
                id = token.data.id_code
                reference = token.data.reference
                signal_names[id] = reference
                signals[reference] = []
            if token.kind is TokenKind.CHANGE_TIME:
                current_time = token.data
            elif token.kind is TokenKind.CHANGE_SCALAR:
                id = token.data.id_code
                value = token.data.value
                reference = signal_names[id]
                signals[reference].append((current_time, value))
            elif token.kind is TokenKind.CHANGE_VECTOR:
                id = token.data.id_code
                value = token.data.value
                reference = signal_names[id]
                signals[reference].append((current_time, value))
    return signals


def extract_timestamps(signals, signal_name, target_values ):
    timestamps = []
    for timestamp, value in signals.get(signal_name, []):
        if value in target_values:
            timestamps.append(timestamp)
    return timestamps


def create_table(pc_q_timestamps, hwpe_busy_rises, hwpe_busy_falls):
    table_data = []
    for i in range(len(pc_q_timestamps)):
        row = {
            f'Trigger Tile': pc_q_timestamps[i],
            f'Start Tile': hwpe_busy_rises[i] if i < len(hwpe_busy_rises) else None,
            f'End Tile': hwpe_busy_falls[i + 1] if i < len(hwpe_busy_falls) else None,
        }
        table_data.append(row)
    return table_data


def extract_instruction(dump_file, string):
    occurrences = []
    with open(dump_file, 'r') as f:
        for line in f:
            if string in line:
                # Get next line (eg "800012b8: 37 05 04 10  	lui	a0, 65600") and extract the address
                line = next(f)
                address = int(line.split(":")[0], 16)
                occurrences.append(address)
    return occurrences


if __name__ == "__main__":
    # Sorry for the ugly code
    dump_file = os.path.join(LOCAL_DIR, '../sw/apps/hwpe_tiled/build/hwpe_tiled.dump')
    vcd_file = os.path.join(LOCAL_DIR, '../vcd/ita.vcd')

    print(f"Reading file {vcd_file}")
    # Hack the VCD file as the parser is not able to handle brakets in scopes
    # Replace "gen_core[8]" with "gen_core_8" and store back
    content = None
    with open(vcd_file, 'r') as f:
        content = f.read()
        content = content.replace("gen_core[8]", "gen_core_8")
    with open(vcd_file, 'w') as f:
        if content is not None:
            f.write(content)

    # Parse the VCD file
    signals = parse_vcd(vcd_file)
    print(f"Found signals: {signals.keys()}")

    # Extract the timestamps of the ita_acquire() function (also work if inlined)
    call_main = extract_instruction(
        dump_file, ";   while(*(volatile uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x04) < 1) ;")
    print(f"Found {len(call_main)} occurrences of the instruction")
    # print(f"Instruction addresses:")
    # for i in call_main:
    #     print(" ", hex(i))
    # print()

    # Extract the timestamps of the ita_trigger() function (also work if inlined)
    instruction = extract_instruction(dump_file, ";   *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x00) = 0;")
    print(f"Found {len(instruction)} occurrences of the instruction")
    # print(f"Instruction addresses:")
    # for i in instruction:
    #     print(" ", hex(i))

    # Extract the timestamps for trigger, tile start and tile end
    pc_main = extract_timestamps(signals, 'pc_q', call_main)
    pc_q_timestamps = extract_timestamps(signals, 'pc_q', instruction)
    hwpe_busy_rises = extract_timestamps(signals, 'hwpe_busy_o', ['1'])
    hwpe_busy_falls = extract_timestamps(signals, 'hwpe_busy_o', ['0'])

    print(f"Found ita_acquire() timestamps: {pc_main}")

    # Create table
    table_data = create_table(pc_q_timestamps, hwpe_busy_rises, hwpe_busy_falls)
    df = pd.DataFrame(table_data)

    # Calculate Runtime and Latency
    df["Tile Runtime"] = (df["End Tile"] - df["Start Tile"]) // 2000
    df["Wait Runtime"] = (df["Start Tile"].shift(-1) - df["End Tile"]) // 2000
    df["Trigger Latency"] = (df["Trigger Tile"].shift(-1) - df["End Tile"]) // 2000

    print(df)
    df.to_csv(os.path.join(LOCAL_DIR, 'output.csv'), index = False)
