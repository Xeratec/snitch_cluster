// Copyright 2022 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Angelo Garofalo <agarofalo@iis.ee.ethz.ch>
//         Gamze Islamoglu <gislamoglu@iis.ee.ethz.ch>

/**************************************************/
/* This module wraps the hardware processing engine
   and expose its data and ctrl interfaces 
   towards the snitch cluster */
/**************************************************/

// includes
`include "axi/assign.svh"
`include "axi/typedef.svh"
`include "common_cells/assertions.svh"
`include "common_cells/registers.svh"

`include "mem_interface/typedef.svh"
`include "register_interface/typedef.svh"
`include "reqrsp_interface/typedef.svh"
`include "tcdm_interface/typedef.svh"
`include "snitch_vm/typedef.svh"

module snitch_hwpe_subsystem 
  import hci_package::*;
  import hwpe_stream_package::*;
#(
  // struct params
  parameter type         tcdm_req_t     = logic,
  parameter type         tcdm_rsp_t     = logic,
  parameter type         hwpectrl_req_t = logic,
  parameter type         hwpectrl_rsp_t = logic,

  // hwpe params
  parameter int unsigned AccDataWidth   = 1024,
  parameter int unsigned IdWidth        = 8,
  parameter int unsigned NrCores        = 8,

  // system params
  parameter int unsigned TCDMDataWidth  = 64,
  parameter int unsigned NrTCDMPorts    = (AccDataWidth / TCDMDataWidth)
) (
  input logic clk_i, 
  input logic rst_ni, 
  input logic test_mode_i,
  
  // data interface towards the memory (HWPE mst of the interco)
  output tcdm_req_t [NrTCDMPorts-1:0] hwpe_tcdm_req_o,
  input  tcdm_rsp_t [NrTCDMPorts-1:0] hwpe_tcdm_rsp_i, // check datawidth of this typedef "tcdm_rsp_t"
  
  // ctrl interface towards the hwpe (hwpe is AXI slave)
  input  hwpectrl_req_t hwpe_ctrl_req_i,
  output hwpectrl_rsp_t hwpe_ctrl_rsp_o
  
);

  /**********************************/
  /* Internal Signals declaration   */
  /**********************************/

  // internal signals definition: event signals
  logic [NrCores-1:0][1:0] evt;

  // internal signals definition: busy signal
  logic busy;

  // TODO: clock gating cells

  // HWPE
  hci_core_intf #( 
    .DW ( AccDataWidth )
  ) tcdm (
    .clk (clk_i)
  );

  hwpe_ctrl_intf_periph #( 
    .ID_WIDTH (IdWidth) 
  ) periph ( 
    .clk (clk_i) 
  );

  /********************************************************************************/
  // binding tcdm struct with hwpe internals (from tcdm to pure req/gnt)-- data port
  /********************************************************************************/

  // Generate the TCDM signals
  logic [NrTCDMPorts-1:0] local_tcdm_gnt;
  logic [NrTCDMPorts-1:0] local_tcdm_r_valid;
  logic [NrTCDMPorts-1:0][TCDMDataWidth-1:0] local_tcdm_r_data;

  //bindings
  genvar i;
  generate
    for (i = 0; i < NrTCDMPorts; i++) begin
      // request channel
      assign hwpe_tcdm_req_o[i].q_valid        = tcdm.req;
      assign hwpe_tcdm_req_o[i].q.addr         = tcdm.add + i*(TCDMDataWidth/8);
      assign hwpe_tcdm_req_o[i].q.write        = ~tcdm.wen;
      assign hwpe_tcdm_req_o[i].q.strb         = tcdm.be[(i+1)*(TCDMDataWidth/8)-1:i*(TCDMDataWidth/8)];
      assign hwpe_tcdm_req_o[i].q.data         = tcdm.data[(i+1)*TCDMDataWidth-1:i*TCDMDataWidth];
      assign hwpe_tcdm_req_o[i].q.amo          = reqrsp_pkg::AMONone;
      assign hwpe_tcdm_req_o[i].q.user.core_id = '0;
      assign hwpe_tcdm_req_o[i].q.user.is_core = 1'b0;
      assign local_tcdm_gnt[i] = hwpe_tcdm_rsp_i[i].q_ready;
      assign local_tcdm_r_valid[i] = hwpe_tcdm_rsp_i[i].p_valid;
      assign local_tcdm_r_data[i] = hwpe_tcdm_rsp_i[i].p.data;
      // response channel
      if (i == NrTCDMPorts-1) begin
        assign tcdm.gnt = &(local_tcdm_gnt);
        assign tcdm.r_valid = &(local_tcdm_r_valid);
        assign tcdm.r_data = { >> {local_tcdm_r_data} };
        assign tcdm.r_user = '0;
        assign tcdm.r_id = '0;
        assign tcdm.r_ecc = '0;
        assign tcdm.egnt = '0;
        assign tcdm.r_evalid = '0;
      end
    end
  endgenerate

  /**********************************************************************************/
  // binding tcdm struct with hwpe internals (from tcdm to pure req/gnt) -- ctrl port
  /**********************************************************************************/
  always_comb begin
    periph.req              = hwpe_ctrl_req_i.q_valid;
    periph.add              = hwpe_ctrl_req_i.q.addr[31:0]; // HWPE Ctrl Addr Width is overwritten with PhysicalAddrWidth (48)
    periph.wen              = ~hwpe_ctrl_req_i.q.write;
    periph.be               = hwpe_ctrl_req_i.q.strb;
    periph.data             = hwpe_ctrl_req_i.q.data[31:0]; // (TODO) Parametrize
    periph.id               = hwpe_ctrl_req_i.q.user;
    hwpe_ctrl_rsp_o.q_ready = periph.gnt;
    hwpe_ctrl_rsp_o.p.data  = periph.r_data;
    hwpe_ctrl_rsp_o.p_valid = periph.r_valid;
  end

  /*******************/
  /* HWPE Instance   */
  /*******************/

  ita_top dut (
    .clk_i,
    .rst_ni,
    .test_mode_i (1'b0       ),
    .evt_o       (evt        ),
    .tcdm        (tcdm       ),
    .periph      (periph     )
  );

  /***************/
  /* Assertions  */
  /***************/

endmodule : snitch_hwpe_subsystem
