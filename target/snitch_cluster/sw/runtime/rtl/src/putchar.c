// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

extern uintptr_t volatile tohost, fromhost;

// Rudimentary string buffer for putc calls.
#define PUTC_BUFFER_LEN (1024 - sizeof(size_t))

typedef struct  {
    size_t size;
    uint64_t syscall_mem[8];
} putc_buffer_header_t;

typedef struct putc_buffer {
    putc_buffer_header_t hdr;
    char data[PUTC_BUFFER_LEN];
} putc_buffer_t;

static volatile putc_buffer_t putc_buffer[SNRT_CLUSTER_NUM*SNRT_CLUSTER_CORE_NUM] __attribute__((section(".dram")));

// Provide an implementation for putchar.
void _putchar(char character) {
    volatile struct putc_buffer *buf = &putc_buffer[snrt_hartid()];
    buf->data[buf->hdr.size++] = character;
    if (buf->hdr.size == PUTC_BUFFER_LEN || character == '\n') {
        buf->hdr.syscall_mem[0] = 64;  // sys_write
        buf->hdr.syscall_mem[1] = 1;   // file descriptor (1 = stdout)
        buf->hdr.syscall_mem[2] = (uintptr_t)&buf->data;  // buffer
        buf->hdr.syscall_mem[3] = buf->hdr.size;          // length

        tohost = (uintptr_t)buf->hdr.syscall_mem;
        while (fromhost == 0)
            ;
        fromhost = 0;

        buf->hdr.size = 0;
    }
}
