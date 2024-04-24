// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "snrt.h"

#define HWPE_ADDR_BASE 0x0010040000

int main() {

  if (snrt_global_core_idx() == 0) {

    int status;

    // Soft clear
    *(int *)(HWPE_ADDR_BASE + 0x14) = 0;

    // acquire job
    status = -1;
    while(status < 0)
      status = *(int *)(HWPE_ADDR_BASE + 0x04);

    // Program ITA
    *(int *)(HWPE_ADDR_BASE + 0x20) = 0x00000000; // input ptr
    *(int *)(HWPE_ADDR_BASE + 0x24) = 0x00002000; // weight ptr 0
    *(int *)(HWPE_ADDR_BASE + 0x28) = 0x00003000; // weight ptr 1
    *(int *)(HWPE_ADDR_BASE + 0x2C) = 0x00006000; // bias ptr
    *(int *)(HWPE_ADDR_BASE + 0x30) = 0x00007300; // output ptr
    *(int *)(HWPE_ADDR_BASE + 0x38) = 0x00000111; // ita tiles
    *(int *)(HWPE_ADDR_BASE + 0x3C) = 0x534E6C77; // eps_mult 1
    *(int *)(HWPE_ADDR_BASE + 0x40) = 0x00005A46; // eps_mult 2
    *(int *)(HWPE_ADDR_BASE + 0x44) = 0x0F0F0E0E; // right shift 1
    *(int *)(HWPE_ADDR_BASE + 0x48) = 0x00000E0F; // right shift 2
    *(int *)(HWPE_ADDR_BASE + 0x4C) = 0xAA4EDF63; // add 1
    *(int *)(HWPE_ADDR_BASE + 0x50) = 0x00008BC1; // add 2
    *(int *)(HWPE_ADDR_BASE + 0x54) = 0x00000003; // ctrl stream

    // Trigger ITA
    *(int *)(HWPE_ADDR_BASE + 0x00) = 0;

    // Wait for completion
    while (snrt_hwpe_busy() == 0);
    while (snrt_hwpe_busy() == 1);

  }

  return 0;
}
