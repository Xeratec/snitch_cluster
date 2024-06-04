#ifndef __ITA_HAL_H__
#define __ITA_HAL_H__

#define SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR 0x0010040000

typedef struct __attribute__((__packed__)) {
  int value: 24;
} ita_int24_t;

#define ITA_TILES(s, e, p) (s | (e << 4) | (p << 8))
#define ITA_FLAGS(weight_preload, weight_nextload, bias_disable, bias_direction, output_disable) \
  ((weight_preload) | ((weight_nextload) << 1) | ((bias_disable) << 2) | ((bias_direction) << 3) | ((output_disable) << 4))

static inline void __attribute((always_inline))  ita_write_regs(uint32_t input_addr,
                    uint32_t weight_addr,
                    uint32_t weight_next_addr,
                    uint32_t bias_addr,
                    uint32_t output_addr,
                    uint32_t tiles,
                    uint32_t eps1,
                    uint32_t eps2,
                    uint32_t right_shift1,
                    uint32_t right_shift2,
                    uint32_t add1,
                    uint32_t add2,
                    uint32_t flags) {
  // Program ITA
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x20) = input_addr - snrt_l1_start_addr();
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x24) = weight_addr - snrt_l1_start_addr();
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x28) = weight_next_addr - snrt_l1_start_addr();
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x2C) = bias_addr - snrt_l1_start_addr();
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x30) = output_addr - snrt_l1_start_addr();
  // unused sequence length
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x38) = tiles;
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x3C) = eps1;
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x40) = eps2;
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x44) = right_shift1;
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x48) = right_shift2;
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x4C) = add1;
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x50) = add2;
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x54) = flags; // ctrl stream
}

static inline void __attribute((always_inline))  ita_soft_clear() {
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x14) = 0;
  for (volatile int i = 0; i < 10; i++) ;
}

static inline void __attribute((always_inline))  ita_soft_clear_keep_regs() {
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x14) = 1;
  for (volatile uint32_t i = 0; i < 10; i++) ;
}

static inline void __attribute((always_inline))  ita_acquire_job() {
  while(*(volatile uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x04) < 1) ;
}

static inline void __attribute((always_inline))  ita_wait_job() {
  while (snrt_hwpe_busy() != 0);
}

static inline void __attribute((always_inline)) ita_trigger() {
  *(uint32_t *)(SNITCH_CLUSTER_ITA_HWPE_BASE_ADDR + 0x00) = 0;
}

#endif  // __ITA_HAL_H__
