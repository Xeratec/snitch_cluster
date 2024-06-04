restart -nowave
add wave /tb_bin/i_dut/i_snitch_cluster/i_cluster/gen_core[8]/i_snitch_cc/i_snitch/pc_q
add wave /tb_bin/i_dut/i_snitch_cluster/i_cluster/i_snitch_hwpe_subsystem/hwpe_busy_o
mkdir -p ./vcd
vcd file ./vcd/ita.vcd
vcd add /tb_bin/i_dut/i_snitch_cluster/i_cluster/i_snitch_hwpe_subsystem/hwpe_busy_o
vcd add -internal /tb_bin/i_dut/i_snitch_cluster/i_cluster/gen_core\[8\]/i_snitch_cc/i_snitch/pc_q
run 25000ns
vcd off
vcd flush