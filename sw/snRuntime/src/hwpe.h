// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

inline uint32_t snrt_hwpe_evt(core_idx) {
    uint32_t evt = * (volatile uint32_t*)snrt_cluster_hwpe_evt_addr();
    return (0x3 << (core_idx)) & evt;
}

inline uint32_t snrt_hwpe_busy() {
    return * (volatile uint32_t*)snrt_cluster_hwpe_busy_addr();
}
