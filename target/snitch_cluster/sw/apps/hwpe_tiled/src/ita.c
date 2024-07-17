// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snrt.h"
#include "ItaHal.h"
#include "mem_snitch_cluster.h"

void check(int step, uint8_t *data, uint8_t *golden, int size) {
  int errors = 0;
  int first_error = -1;
  int last_error = -1;
  DUMP(step);
  for (int i = 0; i < size; i++) {
    if (data[i] != golden[i]) {
      if (first_error == -1) {
        first_error = i;
      }
      last_error = i;
      //DUMP(i);
      //DUMP(golden[i]);
      //DUMP(data[i]);
      errors++;
    }
  }
  DUMP(errors);
  if (first_error != -1) {
    DUMP(first_error);
    DUMP(last_error);
  }
}

void dump_data(int32_t *ptr, int size) {
  for (int i = 0; i < size; i++) {
    DUMP(ptr[i]);
  }
}

//#define MEASURE_PERF

static inline void __attribute((always_inline)) ita_compute_projection_step(
  int8_t *input_a_l3,
  int8_t *input_b_l3,
  ita_int24_t *bias_l3,
  int8_t *output_l3,
  int8_t *input_a_l1[2],
  int8_t *input_b_l1[2],
  ita_int24_t *bias_l1,
  int8_t *output_l1[2],
  int8_t *prev_output_tile_l3,
  int8_t *prev_output_tile_l1,
  int prev_output_tile_size,
  int8_t *next_step_input_b_tile_l3,
  int8_t *next_step_input_b_tile_l1,
  int next_step_input_b_tile_size,
  uint32_t requant_eps_mult[2],
  uint32_t requant_right_shift[2],
  uint32_t requant_add[2],
  int is_first_job
) {
  int index_buff = 0;

  if (is_first_job) {
    // Load first tile of input_b
    const int tile_size_input_b = TILE_SIZE_PROJECTION_SPACE * TILE_SIZE_EMBEDDING_SPACE;
    snrt_dma_start_1d((void *)input_b_l1[index_buff], (void *)input_b_l3, tile_size_input_b);
  }

#ifdef MEASURE_PERF
  uint32_t wait_ita_cycles = 0;
  uint32_t wait_dma_cycles = 0;
#endif // MEASURE_PERF
  for (int i = 0; i < N_TILE_SEQUENCE_LENGTH; i++) {
    for (int j = 0; j < N_TILE_PROJECTION_SPACE; j++) {
      const int offset_bias = j * TILE_SIZE_PROJECTION_SPACE;
      snrt_dma_start_1d((void *)bias_l1, (void *)(bias_l3 + offset_bias), TILE_SIZE_PROJECTION_SPACE * sizeof(ita_int24_t));

      for (int k = 0; k < N_TILE_EMBEDDING_SPACE; k++) {
        const int is_first_tile = (i == 0) && (j == 0) && (k == 0);
        const int is_last_tile = (i == N_TILE_SEQUENCE_LENGTH - 1) && (j == N_TILE_PROJECTION_SPACE - 1) && (k == N_TILE_EMBEDDING_SPACE - 1);

        // Fetch inputs
        const int tile_size_input_a = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_EMBEDDING_SPACE;
        const int offset_input_a = (i * N_TILE_EMBEDDING_SPACE + k) * tile_size_input_a;
        snrt_dma_start_1d((void *)input_a_l1[index_buff], (void *)(input_a_l3 + offset_input_a), tile_size_input_a);

        const int index_buff_output = (i * N_TILE_PROJECTION_SPACE + j) % 2;

        ita_write_regs((uint32_t)input_a_l1[index_buff],
                       (uint32_t)input_b_l1[index_buff],
                       (uint32_t)(is_last_tile ? next_step_input_b_tile_l1 : input_b_l1[!index_buff]),
                       (uint32_t)bias_l1,
                       (uint32_t)output_l1[index_buff_output],
                       (uint32_t)ITA_TILES(N_TILE_SEQUENCE_LENGTH, N_TILE_EMBEDDING_SPACE, N_TILE_PROJECTION_SPACE),
                       (uint32_t)requant_eps_mult[0],
                       (uint32_t)requant_eps_mult[1],
                       (uint32_t)requant_right_shift[0],
                       (uint32_t)requant_right_shift[1],
                       (uint32_t)requant_add[0],
                       (uint32_t)requant_add[1],
                       (uint32_t)ITA_FLAGS(is_first_job && is_first_tile, 1, 0, 0, k == N_TILE_EMBEDDING_SPACE -1 ? 0 : 1));

#ifdef MEASURE_PERF
        uint32_t pre_cycles = snrt_get_perf_counter(SNRT_PERF_CNT0);
#endif // MEASURE_PERF
        snrt_dma_wait_all();
#ifdef MEASURE_PERF
        uint32_t post_cycles = snrt_get_perf_counter(SNRT_PERF_CNT0);
        wait_dma_cycles += post_cycles - pre_cycles;
#endif // MEASURE_PERF

        if (is_last_tile) {
          snrt_dma_start_1d((void *)next_step_input_b_tile_l1, (void *)next_step_input_b_tile_l3, next_step_input_b_tile_size);
        } else {
          // Load next input_b tile
          const int next_k = k == N_TILE_EMBEDDING_SPACE - 1 ? 0 : k + 1;
          const int inc_j = j == N_TILE_PROJECTION_SPACE - 1 ? 0 : j + 1;
          const int next_j = k == N_TILE_EMBEDDING_SPACE - 1 ? inc_j : j;
          const int tile_size_input_b = TILE_SIZE_PROJECTION_SPACE * TILE_SIZE_EMBEDDING_SPACE;
          const int offset_input_b = (next_j * N_TILE_EMBEDDING_SPACE + next_k) * tile_size_input_b;
          snrt_dma_start_1d((void *)input_b_l1[!index_buff], (void *)(input_b_l3 + offset_input_b), tile_size_input_b);
        }

        if (!(is_first_job && is_first_tile)) {
#ifdef MEASURE_PERF
          uint32_t pre_cycles = snrt_get_perf_counter(SNRT_PERF_CNT0);
#endif // MEASURE_PERF
          ita_wait_job();
#ifdef MEASURE_PERF
          uint32_t post_cycles = snrt_get_perf_counter(SNRT_PERF_CNT0);
          wait_ita_cycles += post_cycles - pre_cycles;
#endif // MEASURE_PERF
        }

        ita_trigger();

        // Previous output tile writeback
        if (k == 0) {
          if (i == 0 && j == 0) {
            // Previous step's last output tile writeback
            if (prev_output_tile_l3 != 0 && prev_output_tile_l1 != 0 && prev_output_tile_size != 0) {
              snrt_dma_start_1d((void *)prev_output_tile_l3, (void *)prev_output_tile_l1, prev_output_tile_size);
            }
          } else {
            const int prev_i = j == 0 ? i - 1 : i;
            const int prev_j = j == 0 ? N_TILE_PROJECTION_SPACE - 1 : j - 1;
            const int tile_size_output = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
            const int offset_output = (prev_i * N_TILE_PROJECTION_SPACE + prev_j) * tile_size_output;
            const int index_buff_output = (prev_i * N_TILE_PROJECTION_SPACE + prev_j) % 2;
            snrt_dma_start_1d((void *)(output_l3 + offset_output), (void *)output_l1[index_buff_output], tile_size_output);
          }
        }

        index_buff = !index_buff;
      }
    }
  }
#ifdef MEASURE_PERF
  DUMP(wait_ita_cycles);
  DUMP(wait_dma_cycles);
#endif // MEASURE_PERF
}

static inline void __attribute((always_inline))  ita_compute_value_projection_step(
  int8_t *input_a_l3,
  int8_t *input_b_l3,
  ita_int24_t *bias_l3,
  int8_t *output_l3,
  int8_t *input_a_l1[2],
  int8_t *input_b_l1[2],
  ita_int24_t *bias_l1,
  int8_t *output_l1[2],
  int8_t *prev_output_tile_l3,
  int8_t *prev_output_tile_l1,
  int prev_output_tile_size,
  int8_t *next_step_input_b_tile_l3,
  int8_t *next_step_input_b_tile_l1,
  int next_step_input_b_tile_size,
  uint32_t requant_eps_mult[2],
  uint32_t requant_right_shift[2],
  uint32_t requant_add[2]
) {
  int index_buff = 0;

  for (int i = 0; i < N_TILE_PROJECTION_SPACE; i++) {
    const int offset_bias = i * TILE_SIZE_PROJECTION_SPACE;
    snrt_dma_start_1d((void *)bias_l1, (void *)(bias_l3 + offset_bias), TILE_SIZE_PROJECTION_SPACE * sizeof(ita_int24_t));

    for (int j = 0; j < N_TILE_SEQUENCE_LENGTH; j++) {
      for (int k = 0; k < N_TILE_EMBEDDING_SPACE; k++) {
        const int is_first_tile = (i == 0) && (j == 0) && (k == 0);
        const int is_last_tile = (i == N_TILE_PROJECTION_SPACE - 1) && (j == N_TILE_SEQUENCE_LENGTH - 1) && (k == N_TILE_EMBEDDING_SPACE - 1);

        // Fetch inputs
        const int tile_size_input_b = TILE_SIZE_PROJECTION_SPACE * TILE_SIZE_EMBEDDING_SPACE;
        const int offset_input_b = (i * N_TILE_EMBEDDING_SPACE + k) * tile_size_input_b;
        snrt_dma_start_1d((void *)input_b_l1[index_buff], (void *)(input_b_l3 + offset_input_b), tile_size_input_b);

        const int index_buff_output = (i * N_TILE_SEQUENCE_LENGTH + j) % 2;

        ita_write_regs((uint32_t)input_b_l1[index_buff],
                       (uint32_t)input_a_l1[index_buff],
                       (uint32_t)is_last_tile ? next_step_input_b_tile_l1 : input_a_l1[!index_buff],
                       (uint32_t)bias_l1,
                       (uint32_t)output_l1[index_buff_output],
                       (uint32_t)ITA_TILES(N_TILE_SEQUENCE_LENGTH, N_TILE_EMBEDDING_SPACE, N_TILE_PROJECTION_SPACE),
                       (uint32_t)requant_eps_mult[0],
                       (uint32_t)requant_eps_mult[1],
                       (uint32_t)requant_right_shift[0],
                       (uint32_t)requant_right_shift[1],
                       (uint32_t)requant_add[0],
                       (uint32_t)requant_add[1],
                       (uint32_t)ITA_FLAGS(0, 1, 0, 1, k == N_TILE_EMBEDDING_SPACE -1 ? 0 : 1));


        snrt_dma_wait_all();

        if (is_last_tile) {
          if (N_TILE_SEQUENCE_LENGTH == 1 && N_TILE_PROJECTION_SPACE == 1) {
            // Only one Pk tile case
            memcpy(next_step_input_b_tile_l1, prev_output_tile_l1, next_step_input_b_tile_size);
          } else {
            snrt_dma_start_1d((void *)next_step_input_b_tile_l1, (void *)next_step_input_b_tile_l3, next_step_input_b_tile_size);
          }
        } else {
          const int next_k = k == N_TILE_EMBEDDING_SPACE - 1 ? 0 : k + 1;
          const int inc_j = j == N_TILE_SEQUENCE_LENGTH - 1 ? 0 : j + 1;
          const int next_j = k == N_TILE_EMBEDDING_SPACE - 1 ? inc_j : j;
          const int tile_size_input_a = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_EMBEDDING_SPACE;
          const int offset_input_a = (next_j * N_TILE_EMBEDDING_SPACE + next_k) * tile_size_input_a;
          snrt_dma_start_1d((void *)input_a_l1[!index_buff], (void *)(input_a_l3 + offset_input_a), tile_size_input_a);
        }

        ita_wait_job();
        // if (!(is_first_tile)) {
          // ita_wait_job();
        // }

        ita_trigger();

        // Previous output tile writeback
        if (k == 0) {
          if (i == 0 && j == 0) {
            // Previous step's last output tile writeback
            if (prev_output_tile_l3 != 0 && prev_output_tile_l1 != 0 && prev_output_tile_size != 0) {
              snrt_dma_start_1d((void *)prev_output_tile_l3, (void *)prev_output_tile_l1, prev_output_tile_size);
            }
          } else {
            const int prev_i = j == 0 ? i - 1 : i;
            const int prev_j = j == 0 ? N_TILE_SEQUENCE_LENGTH - 1 : j - 1;
            const int tile_size_output = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
            const int offset_output = (prev_i * N_TILE_SEQUENCE_LENGTH + prev_j) * tile_size_output;
            const int index_buff_output = (prev_i * N_TILE_SEQUENCE_LENGTH + prev_j) % 2;
            snrt_dma_start_1d((void *)(output_l3 + offset_output), (void *)output_l1[index_buff_output], tile_size_output);
          }
        }

        index_buff = !index_buff;
      }
    }
  }
}

static inline void __attribute((always_inline))  ita_compute_qk_row(
  int8_t *input_Pq_row_l3,
  int8_t *input_Pk_row_l3,
  int8_t *output_row_l3,
  int8_t *input_Pq_l1[2],
  int8_t *input_Pk_l1[2],
  int8_t *output_l1[2],
  int8_t *prev_output_tile_l3,
  int8_t *prev_output_tile_l1,
  int prev_output_tile_size,
  int8_t *next_step_input_b_tile_l3,
  int8_t *next_step_input_b_tile_l1,
  int next_step_input_b_tile_size,
  uint32_t requant_eps_mult[2],
  uint32_t requant_right_shift[2],
  uint32_t requant_add[2]
) {
  int index_buff = 0;

  for (int j = 0; j < N_TILE_SEQUENCE_LENGTH; j++) {
    for (int k = 0; k < N_TILE_PROJECTION_SPACE; k++) {
      const int is_first_tile = (j == 0) && (k == 0);
      const int is_last_tile = (j == N_TILE_SEQUENCE_LENGTH - 1) && (k == N_TILE_PROJECTION_SPACE - 1);

      // Fetch inputs
      const int tile_size_input_Pq = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
      const int offset_input_Pq = k * tile_size_input_Pq;
      snrt_dma_start_1d((void *)input_Pq_l1[index_buff], (void *)(input_Pq_row_l3 + offset_input_Pq), tile_size_input_Pq);

      const int index_buff_output = j % 2;

      ita_write_regs((uint32_t)input_Pq_l1[index_buff],
                     (uint32_t)input_Pk_l1[index_buff],
                     (uint32_t)is_last_tile ? next_step_input_b_tile_l1 : input_Pk_l1[!index_buff],
                     (uint32_t)0,
                     (uint32_t)output_l1[index_buff_output],
                     (uint32_t)ITA_TILES(N_TILE_SEQUENCE_LENGTH, N_TILE_EMBEDDING_SPACE, N_TILE_PROJECTION_SPACE),
                     (uint32_t)requant_eps_mult[0],
                     (uint32_t)requant_eps_mult[1],
                     (uint32_t)requant_right_shift[0],
                     (uint32_t)requant_right_shift[1],
                     (uint32_t)requant_add[0],
                     (uint32_t)requant_add[1],
                     (uint32_t)ITA_FLAGS(0, 1, 1, 0, k == N_TILE_PROJECTION_SPACE -1 ? 0 : 1));

      snrt_dma_wait_all();

      ita_wait_job();

      if (is_last_tile) {
        if ((N_TILE_SEQUENCE_LENGTH == 1) && (N_TILE_PROJECTION_SPACE == 1)) {
          // Only one Pv tile case
          memcpy(next_step_input_b_tile_l1, prev_output_tile_l1, next_step_input_b_tile_size);
        } else {
          snrt_dma_start_1d((void *)next_step_input_b_tile_l1, (void *)next_step_input_b_tile_l3, next_step_input_b_tile_size);
        }
      } else {
        const int next_k = k == N_TILE_PROJECTION_SPACE - 1 ? 0 : k + 1;
        const int inc_j = j == N_TILE_SEQUENCE_LENGTH - 1 ? 0 : j + 1;
        const int next_j = k == N_TILE_PROJECTION_SPACE - 1 ? inc_j : j;
        const int tile_size_input_Pk = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
        const int offset_input_Pk = (next_j * N_TILE_PROJECTION_SPACE + next_k) * tile_size_input_Pk;
        snrt_dma_start_1d((void *)input_Pk_l1[!index_buff], (void *)(input_Pk_row_l3 + offset_input_Pk), tile_size_input_Pk);
      }

      ita_trigger();

      // Previous output tile writeback
      if (k == 0) {
        if (j == 0) {
          snrt_dma_start_1d((void *)prev_output_tile_l3, (void *)prev_output_tile_l1, prev_output_tile_size);
        } else {
          const int tile_size_output = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_SEQUENCE_LENGTH;
          const int offset_output = (j - 1) * tile_size_output;
          const int index_buff_output = (j - 1) % 2;
          snrt_dma_start_1d((void *)(output_row_l3 + offset_output), (void *)output_l1[index_buff_output], tile_size_output);
        }
      }

      index_buff = !index_buff;
    }
  }
}

static inline void __attribute((always_inline))  ita_compute_attention_row(
  int8_t *input_qk_row_l3,
  int8_t *input_Pv_row_l3,
  int8_t *output_row_l3,
  int8_t *input_qk_l1[2],
  int8_t *input_Pv_l1[2],
  int8_t *output_l1[2],
  int8_t *prev_output_tile_l3,
  int8_t *prev_output_tile_l1,
  int prev_output_tile_size,
  int8_t *next_step_input_b_tile_l3,
  int8_t *next_step_input_b_tile_l1,
  int next_step_input_b_tile_size,
  uint32_t requant_eps_mult[2],
  uint32_t requant_right_shift[2],
  uint32_t requant_add[2]
) {
  int index_buff = 0;

  // Fetch inputs
  if (N_TILE_SEQUENCE_LENGTH == 1) {
    const int tile_size_input_qk = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_SEQUENCE_LENGTH;
    memcpy(input_qk_l1[0], prev_output_tile_l1, tile_size_input_qk);
  }

  for (int j = 0; j < N_TILE_PROJECTION_SPACE; j++) {
    for (int k = 0; k < N_TILE_SEQUENCE_LENGTH; k++) {
      const int is_first_tile = (j == 0) && (k == 0);
      const int is_last_tile = (j == N_TILE_PROJECTION_SPACE - 1) && (k == N_TILE_SEQUENCE_LENGTH - 1);

      // Fetch inputs
      if (N_TILE_SEQUENCE_LENGTH > 1) {
        const int tile_size_input_qk = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_SEQUENCE_LENGTH;
        const int offset_input_qk = k * tile_size_input_qk;
        snrt_dma_start_1d((void *)input_qk_l1[index_buff], (void *)(input_qk_row_l3 + offset_input_qk), tile_size_input_qk);
      }

      const int index_buff_output = j % 2;

      ita_write_regs((uint32_t)input_qk_l1[index_buff],
                     (uint32_t)input_Pv_l1[index_buff],
                     (uint32_t)is_last_tile ? next_step_input_b_tile_l1 : input_Pv_l1[!index_buff],
                     (uint32_t)0,
                     (uint32_t)output_l1[index_buff_output],
                     (uint32_t)ITA_TILES(N_TILE_SEQUENCE_LENGTH, N_TILE_EMBEDDING_SPACE, N_TILE_PROJECTION_SPACE),
                     (uint32_t)requant_eps_mult[0],
                     (uint32_t)requant_eps_mult[1],
                     (uint32_t)requant_right_shift[0],
                     (uint32_t)requant_right_shift[1],
                     (uint32_t)requant_add[0],
                     (uint32_t)requant_add[1],
                     (uint32_t)ITA_FLAGS(0, 1, 1, 0, k == N_TILE_SEQUENCE_LENGTH -1 ? 0 : 1));

      snrt_dma_wait_all();

      if (is_last_tile) {
        snrt_dma_start_1d((void *)next_step_input_b_tile_l1, (void *)next_step_input_b_tile_l3, next_step_input_b_tile_size);
      } else {
        const int next_k = k == N_TILE_SEQUENCE_LENGTH - 1 ? 0 : k + 1;
        const int inc_j = j == N_TILE_PROJECTION_SPACE - 1 ? 0 : j + 1;
        const int next_j = k == N_TILE_SEQUENCE_LENGTH - 1 ? inc_j : j;
        const int tile_size_input_Pv = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
        const int offset_input_Pv = (next_j * N_TILE_SEQUENCE_LENGTH + next_k) * tile_size_input_Pv;
        snrt_dma_start_1d((void *)input_Pv_l1[!index_buff], (void *)(input_Pv_row_l3 + offset_input_Pv), tile_size_input_Pv);
      }

      ita_wait_job();

      ita_trigger();

      // Previous output tile writeback
      if (k == 0) {
        if (j == 0) {
          snrt_dma_start_1d((void *)prev_output_tile_l3, (void *)prev_output_tile_l1, prev_output_tile_size);
          snrt_dma_wait_all();
        } else {
          const int tile_size_output = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
          const int offset_output = (j - 1) * tile_size_output;
          const int index_buff_output = (j - 1) % 2;
          snrt_dma_start_1d((void *)(output_row_l3 + offset_output), (void *)output_l1[index_buff_output], tile_size_output);
        }
      }

      index_buff = !index_buff;
    }
  }
}

static inline void __attribute((always_inline))  ita_compute_output_embedding_step(
  int8_t *input_av_l3,
  int8_t *input_Wo_l3,
  ita_int24_t *bias_l3,
  int8_t *output_l3,
  int8_t *input_av_l1[2],
  int8_t *input_Wo_l1[2],
  ita_int24_t *bias_l1,
  int8_t *output_l1[2],
  int8_t *prev_output_tile_l3,
  int8_t *prev_output_tile_l1,
  int prev_output_tile_size,
  uint32_t requant_eps_mult[2],
  uint32_t requant_right_shift[2],
  uint32_t requant_add[2]
) {
  int index_buff = 0;

  if ((N_TILE_SEQUENCE_LENGTH == 1) && (N_TILE_PROJECTION_SPACE == 1)) {
    // Fetch inputs
    const int tile_size_input_av = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
    memcpy(input_av_l1[0], prev_output_tile_l1, tile_size_input_av);
  }

  for (int i = 0; i < N_TILE_SEQUENCE_LENGTH; i++) {
    for (int j = 0; j < N_TILE_EMBEDDING_SPACE; j++) {
      const int offset_bias = j * TILE_SIZE_EMBEDDING_SPACE;
      snrt_dma_start_1d((void *)bias_l1, (void *)(bias_l3 + offset_bias), TILE_SIZE_EMBEDDING_SPACE * sizeof(ita_int24_t));

      for (int k = 0; k < N_TILE_PROJECTION_SPACE; k++) {
        const int is_first_tile = (i == 0) && (j == 0) && (k == 0);
        const int is_last_tile = (i == N_TILE_SEQUENCE_LENGTH - 1) && (j == N_TILE_EMBEDDING_SPACE - 1) && (k == N_TILE_PROJECTION_SPACE - 1);

        if ((N_TILE_SEQUENCE_LENGTH > 1) && (N_TILE_EMBEDDING_SPACE > 1)) {
          // Fetch inputs
          const int tile_size_input_av = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
          const int offset_input_av = (i * N_TILE_PROJECTION_SPACE + k) * tile_size_input_av;
          snrt_dma_start_1d((void *)input_av_l1[index_buff], (void *)(input_av_l3 + offset_input_av), tile_size_input_av);
        }

        const int index_buff_output = (i * N_TILE_EMBEDDING_SPACE + j) % 2;

        ita_write_regs((uint32_t)input_av_l1[index_buff],
                       (uint32_t)input_Wo_l1[index_buff],
                       (uint32_t)input_Wo_l1[!index_buff],
                       (uint32_t)bias_l1,
                       (uint32_t)output_l1[index_buff_output],
                       (uint32_t)ITA_TILES(N_TILE_SEQUENCE_LENGTH, N_TILE_EMBEDDING_SPACE, N_TILE_PROJECTION_SPACE),
                       (uint32_t)requant_eps_mult[0],
                       (uint32_t)requant_eps_mult[1],
                       (uint32_t)requant_right_shift[0],
                       (uint32_t)requant_right_shift[1],
                       (uint32_t)requant_add[0],
                       (uint32_t)requant_add[1],
                       (uint32_t)ITA_FLAGS(0, !is_last_tile, 0, 0, k == N_TILE_PROJECTION_SPACE -1 ? 0 : 1));

        snrt_dma_wait_all();

        if (!is_last_tile) {
          const int next_k = k == N_TILE_PROJECTION_SPACE - 1 ? 0 : k + 1;
          const int inc_j = j == N_TILE_EMBEDDING_SPACE - 1 ? 0 : j + 1;
          const int next_j = k == N_TILE_PROJECTION_SPACE - 1 ? inc_j : j;
          const int tile_size_input_Wo = TILE_SIZE_EMBEDDING_SPACE * TILE_SIZE_PROJECTION_SPACE;
          const int offset_input_Wo = (next_j * N_TILE_PROJECTION_SPACE + next_k) * tile_size_input_Wo;
          snrt_dma_start_1d((void *)input_Wo_l1[!index_buff], (void *)(input_Wo_l3 + offset_input_Wo), tile_size_input_Wo);
        }

        ita_wait_job();

        ita_trigger();

        // Previous output tile writeback
        if (k == 0) {
          if (i == 0 && j == 0) {
            // Previous step's last output tile writeback
            if (prev_output_tile_l3 != 0 && prev_output_tile_l1 != 0 && prev_output_tile_size != 0) {
              snrt_dma_start_1d((void *)prev_output_tile_l3, (void *)prev_output_tile_l1, prev_output_tile_size);
            }
          } else {
            const int prev_i = j == 0 ? i - 1 : i;
            const int prev_j = j == 0 ? N_TILE_EMBEDDING_SPACE - 1 : j - 1;
            const int tile_size_output = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_EMBEDDING_SPACE;
            const int offset_output = (prev_i * N_TILE_EMBEDDING_SPACE + prev_j) * tile_size_output;
            const int index_buff_output = (prev_i * N_TILE_EMBEDDING_SPACE + prev_j) % 2;
            snrt_dma_start_1d((void *)(output_l3 + offset_output), (void *)output_l1[index_buff_output], tile_size_output);
          }
        }

        index_buff = !index_buff;
      }
    }
  }
}

int main() {
  if (snrt_is_dm_core()) {
    snrt_alloc_init();
    int8_t *interm_Pq = (int8_t *)snrt_l3alloc(SEQUENCE_LENGTH * PROJECTION_SPACE);
    int8_t *interm_Pk = (int8_t *)snrt_l3alloc(SEQUENCE_LENGTH * PROJECTION_SPACE);
    int8_t *interm_Pv = (int8_t *)snrt_l3alloc(SEQUENCE_LENGTH * PROJECTION_SPACE);
    int8_t *interm_qk = (int8_t *)snrt_l3alloc(SEQUENCE_LENGTH * SEQUENCE_LENGTH);
    int8_t *interm_attention = (int8_t *)snrt_l3alloc(SEQUENCE_LENGTH * PROJECTION_SPACE);
    int8_t *interm_output = (int8_t *)snrt_l3alloc(SEQUENCE_LENGTH * EMBEDDING_SPACE);

    // L1 buffers
    const uint8_t *l1_start_addr = (uint8_t *)snrt_l1_next();

    const uint8_t *input_a0_buff[2] = {l1_start_addr, l1_start_addr + 4096};
    const uint8_t *input_a1_buff[2] = {l1_start_addr + 2*4096, l1_start_addr + 3*4096};
    const uint8_t *input_b0_buff[2] = {l1_start_addr + 4*4096, l1_start_addr + 5*4096};
    const uint8_t *input_b1_buff[2] = {l1_start_addr + 6*4096, l1_start_addr + 7*4096};
    const uint8_t *input_b2_buff[2] = {l1_start_addr + 8*4096, l1_start_addr + 9*4096};
    const uint8_t *output_0_buff[2] = {l1_start_addr + 10*4096, l1_start_addr + 11*4096};
    const uint8_t *output_1_buff[2] = {l1_start_addr + 12*4096, l1_start_addr + 13*4096};
    const uint8_t *output_2_buff[2] = {l1_start_addr + 14*4096, l1_start_addr + 15*4096};

    uint8_t *local_input_Bq = l1_start_addr + 16*4096;
    uint8_t *local_input_Bk = local_input_Bq + 384;
    uint8_t *local_input_Bv = local_input_Bk + 384;
    uint8_t *local_input_Bo = local_input_Bv + 384;

    // requantization
    uint8_t *local_requant_eps_mult_ptr = local_input_Bo + 384;
    uint8_t *local_requant_right_shift_ptr = local_requant_eps_mult_ptr + 8;
    uint8_t *local_requant_add_ptr = local_requant_right_shift_ptr + 8;

    int32_t *local_requant_eps_mult = (int32_t *)local_requant_eps_mult_ptr;
    int32_t *local_requant_right_shift = (int32_t *)local_requant_right_shift_ptr;
    int32_t *local_requant_add = (int32_t *)local_requant_add_ptr;


    // Get that mem to L1
    snrt_dma_start_1d((void *)local_requant_eps_mult, (void *)requant_eps_mult, 8);
    snrt_dma_start_1d((void *)local_requant_right_shift, (void *)requant_right_shift, 8);
    snrt_dma_start_1d((void *)local_requant_add, (void *)requant_add, 8);

    snrt_dma_wait_all();

    ita_soft_clear();
    ita_acquire_job();

#ifdef MEASURE_PERF
    snrt_reset_perf_counter(SNRT_PERF_CNT0);
    snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_CYCLES, 8);
#endif // MEASURE_PERF

    const int tile_size_Wk = TILE_SIZE_PROJECTION_SPACE * TILE_SIZE_EMBEDDING_SPACE;

    ita_compute_projection_step(input_q, input_Wq[0], input_Bq[0], interm_Pq,
                                input_a0_buff, input_b0_buff, local_input_Bq, output_0_buff,
                                0, 0, 0,
                                input_Wk, input_b1_buff[0], tile_size_Wk,
                                local_requant_eps_mult, local_requant_right_shift, local_requant_add,
                                1);

#ifdef MEASURE_PERF
    uint32_t q_projection_cycles = snrt_get_perf_counter(SNRT_PERF_CNT0);
    DUMP(q_projection_cycles);
#endif // MEASURE_PERF

    const int prev_i_Pq = N_TILE_SEQUENCE_LENGTH - 1;
    const int prev_j_Pq = N_TILE_PROJECTION_SPACE - 1;
    const int tile_size_Pq = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
    const int offset_Pq = (prev_i_Pq * N_TILE_PROJECTION_SPACE + prev_j_Pq) * tile_size_Pq;
    int8_t *interm_Pq_tile_ptr = interm_Pq + offset_Pq;
    const int index_buff_Pq = (N_TILE_SEQUENCE_LENGTH * N_TILE_PROJECTION_SPACE - 1) % 2;

    const int tile_size_k = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_EMBEDDING_SPACE;

    ita_compute_projection_step(input_k, input_Wk[0], input_Bk[0], interm_Pk,
                                input_a1_buff, input_b1_buff, local_input_Bk, output_1_buff,
                                interm_Pq_tile_ptr, output_0_buff[index_buff_Pq], tile_size_Pq,
                                input_k, input_b2_buff[0], tile_size_k,
                                local_requant_eps_mult, local_requant_right_shift, local_requant_add,
                                0);

#ifdef MEASURE_PERF
    uint32_t k_projection_cycles = snrt_get_perf_counter(SNRT_PERF_CNT0) - q_projection_cycles;
    DUMP(k_projection_cycles);

    snrt_stop_perf_counter(SNRT_PERF_CNT0);
#endif // MEASURE_PERF

    const int prev_i_Pk = N_TILE_SEQUENCE_LENGTH - 1;
    const int prev_j_Pk = N_TILE_PROJECTION_SPACE - 1;
    const int tile_size_Pk = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
    const int offset_Pk = (prev_i_Pk * N_TILE_PROJECTION_SPACE + prev_j_Pk) * tile_size_Pk;
    int8_t *interm_Pk_tile_ptr = interm_Pk + offset_Pk;
    const int index_buff_Pk = (N_TILE_SEQUENCE_LENGTH * N_TILE_PROJECTION_SPACE - 1) % 2;

    ita_compute_value_projection_step(input_k, input_Wv[0], input_Bv[0], interm_Pv,
                                      input_b2_buff, input_a0_buff, local_input_Bv, output_2_buff,
                                      interm_Pk_tile_ptr, output_1_buff[index_buff_Pk], tile_size_Pk,
                                      interm_Pk, input_b0_buff[0], tile_size_Pk,
                                      local_requant_eps_mult, local_requant_right_shift, local_requant_add);

    const int prev_i_Pv = N_TILE_PROJECTION_SPACE - 1;
    const int prev_j_Pv = N_TILE_SEQUENCE_LENGTH - 1;
    const int tile_size_Pv = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
    const int offset_Pv = (prev_i_Pv * N_TILE_SEQUENCE_LENGTH + prev_j_Pv) * tile_size_Pv;
    int8_t *interm_Pv_tile_ptr = interm_Pv + offset_Pv;
    const int index_buff_Pv = (N_TILE_SEQUENCE_LENGTH * N_TILE_PROJECTION_SPACE - 1) % 2;

    const int tile_size_Wo = TILE_SIZE_EMBEDDING_SPACE * TILE_SIZE_PROJECTION_SPACE;

    // Steps 4 & 5
    for (int i = 0; i < N_TILE_SEQUENCE_LENGTH; i++) {
      int8_t *prev_tile_ptr;
      int8_t *local_prev_tile_ptr;
      int prev_tile_size;

      if (i == 0) {
        prev_tile_ptr = interm_Pv_tile_ptr;
        local_prev_tile_ptr = output_2_buff[index_buff_Pv];
        prev_tile_size = tile_size_Pv;
      } else {
        const int tile_size_attention = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
        const int prev_i_attention = i - 1;
        const int prev_j_attention = N_TILE_PROJECTION_SPACE - 1;
        const int offset_attention = (prev_i_attention * N_TILE_PROJECTION_SPACE + prev_j_attention) * tile_size_attention;
        int8_t *interm_attention_tile_ptr = interm_attention + offset_attention;
        const int index_buff_attention = (N_TILE_PROJECTION_SPACE - 1) % 2;

        prev_tile_ptr = interm_attention_tile_ptr;
        local_prev_tile_ptr = output_1_buff[index_buff_attention];
        prev_tile_size = tile_size_attention;
      }

      int8_t *interm_Pq_row = interm_Pq + i * N_TILE_PROJECTION_SPACE * TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
      int8_t *interm_qk_row = interm_qk + i * N_TILE_SEQUENCE_LENGTH * TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_SEQUENCE_LENGTH;

      ita_compute_qk_row(interm_Pq_row, interm_Pk, interm_qk_row,
                         input_a1_buff, input_b0_buff, output_0_buff,
                         prev_tile_ptr, local_prev_tile_ptr, prev_tile_size,
                         interm_Pv, input_b1_buff[0], tile_size_Pv,
                         local_requant_eps_mult, local_requant_right_shift, local_requant_add);

      const int tile_size_qk = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_SEQUENCE_LENGTH;
      const int offset_qk = (N_TILE_SEQUENCE_LENGTH - 1) * tile_size_qk;
      int8_t *interm_qk_tile_ptr = interm_qk_row + offset_qk;
      const int index_buff_qk = (N_TILE_SEQUENCE_LENGTH - 1) % 2;

      int8_t *interm_attention_row = interm_attention + i * N_TILE_PROJECTION_SPACE * TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
      const is_last_row = i == N_TILE_SEQUENCE_LENGTH - 1;
      ita_compute_attention_row(interm_qk_row, interm_Pv, interm_attention_row,
                                input_a0_buff, input_b1_buff, output_1_buff,
                                interm_qk_tile_ptr, output_0_buff[index_buff_qk], tile_size_qk,
                                is_last_row ? input_Wo : interm_Pk, is_last_row ? input_b2_buff[0] : input_b0_buff[0], is_last_row ? tile_size_Wo : tile_size_Pk,
                                local_requant_eps_mult, local_requant_right_shift, local_requant_add);
    }

    // Send back last tile
    const int prev_i_attention = N_TILE_SEQUENCE_LENGTH - 1;
    const int prev_j_attention = N_TILE_PROJECTION_SPACE - 1;
    const int tile_size_attention = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_PROJECTION_SPACE;
    const int offset_attention = (prev_i_attention * N_TILE_PROJECTION_SPACE + prev_j_attention) * tile_size_attention;
    int8_t *interm_attention_tile_ptr = interm_attention + offset_attention;
    const int index_buff_attention = (N_TILE_PROJECTION_SPACE - 1) % 2;

    // Use q_buff_l1 just as an intermediate buffer to not overwrite attention_buff
    ita_compute_output_embedding_step(interm_attention, input_Wo[0], input_Bo[0], interm_output,
                                      input_a1_buff, input_b2_buff, local_input_Bo, output_2_buff,
                                      interm_attention_tile_ptr, output_1_buff[index_buff_attention], tile_size_attention,
                                      local_requant_eps_mult, local_requant_right_shift, local_requant_add);

    ita_wait_job();

    // Send back last tile
    const int prev_i_output = N_TILE_SEQUENCE_LENGTH - 1;
    const int prev_j_output = N_TILE_EMBEDDING_SPACE - 1;
    const int tile_size_output = TILE_SIZE_SEQUENCE_LENGTH * TILE_SIZE_EMBEDDING_SPACE;
    const int offset_output = (prev_i_output * N_TILE_EMBEDDING_SPACE + prev_j_output) * tile_size_output;
    int8_t *interm_output_tile_ptr = interm_output + offset_output;
    const int index_buff_output = (N_TILE_SEQUENCE_LENGTH * N_TILE_EMBEDDING_SPACE - 1) % 2;

    // Send back last tile
    snrt_dma_start_1d((void *)interm_output_tile_ptr, (void *)output_2_buff[index_buff_output], tile_size_output);
    snrt_dma_wait_all();

    check(1, (uint8_t *)interm_Pq, (uint8_t *)golden_interm_Pq, SEQUENCE_LENGTH * PROJECTION_SPACE);
    check(2, (uint8_t *)interm_Pk, (uint8_t *)golden_interm_Pk, SEQUENCE_LENGTH * PROJECTION_SPACE);
    check(3, (uint8_t *)interm_Pv, (uint8_t *)golden_interm_Pv, SEQUENCE_LENGTH * PROJECTION_SPACE);
    check(4, (uint8_t *)interm_qk, (uint8_t *)golden_interm_attention, SEQUENCE_LENGTH * SEQUENCE_LENGTH);
    check(5, (uint8_t *)interm_attention, (uint8_t *)golden_interm_head_output, SEQUENCE_LENGTH * PROJECTION_SPACE);
    check(6, (uint8_t *)interm_output, (uint8_t *)golden_output, SEQUENCE_LENGTH * EMBEDDING_SPACE);
  }

  return 0;
}
