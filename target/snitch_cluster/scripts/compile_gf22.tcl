
set DIR_LIB "/usr/pack/gf-22-kgf/invecas/std/2020.01"
set MEM_LIB "/usr/pack/gf-22-kgf/dz/mem"
set NETLIST_DIR "/usr/scratch2/bismantova/gislamoglu/ita_snitch/ita_snitch_gf22/gf22/synopsys/out/ita_snitch_cluster_2000ps"

vlog -incr +nospecify -work work-vsim/ \
    "${DIR_LIB}/GF22FDX_SC8T_104CPP_BASE_CSC20L_FDK_RELV06R40/model/verilog/prim.v" \
    "${DIR_LIB}/GF22FDX_SC8T_104CPP_BASE_CSC24L_FDK_RELV06R40/model/verilog/prim.v" \
    "${DIR_LIB}/GF22FDX_SC8T_104CPP_BASE_CSC28L_FDK_RELV06R40/model/verilog/prim.v" \
    "${DIR_LIB}/GF22FDX_SC8T_104CPP_BASE_CSC20SL_FDK_RELV06R40/model/verilog/prim.v" \
    "${DIR_LIB}/GF22FDX_SC8T_104CPP_BASE_CSC24SL_FDK_RELV06R40/model/verilog/prim.v" \
    "${DIR_LIB}/GF22FDX_SC8T_104CPP_BASE_CSC28SL_FDK_RELV06R40/model/verilog/prim.v"

vlog -incr +nospecify -work work-vsim/ \
    "${DIR_LIB}/GF22FDX_SC8T_104CPP_BASE_CSC20L_FDK_RELV06R40/model/verilog/GF22FDX_SC8T_104CPP_BASE_CSC20L.v" \
    "${DIR_LIB}/GF22FDX_SC8T_104CPP_BASE_CSC24L_FDK_RELV06R40/model/verilog/GF22FDX_SC8T_104CPP_BASE_CSC24L.v" \
    "${DIR_LIB}/GF22FDX_SC8T_104CPP_BASE_CSC28L_FDK_RELV06R40/model/verilog/GF22FDX_SC8T_104CPP_BASE_CSC28L.v" \
    "${DIR_LIB}/GF22FDX_SC8T_104CPP_BASE_CSC20SL_FDK_RELV06R40/model/verilog/GF22FDX_SC8T_104CPP_BASE_CSC20SL.v" \
    "${DIR_LIB}/GF22FDX_SC8T_104CPP_BASE_CSC24SL_FDK_RELV06R40/model/verilog/GF22FDX_SC8T_104CPP_BASE_CSC24SL.v" \
    "${DIR_LIB}/GF22FDX_SC8T_104CPP_BASE_CSC28SL_FDK_RELV06R40/model/verilog/GF22FDX_SC8T_104CPP_BASE_CSC28SL.v"

vlog -incr +nospecify -sv -timescale "1 ns / 1 ps" -work work-vsim/ \
    -suppress vlog-2583 -suppress vlog-13314 -suppress vlog-13233 \
    +define+IVCS_INIT_MEM \
    "$MEM_LIB/R1PH/V04R20SZ/model/verilog/IN22FDX_R1PH_NFHN_W00128B128M02C256.v" \
    "$MEM_LIB/R1PH/V04R20SZ/model/verilog/IN22FDX_R1PH_NFHN_W00512B032M02C256.v" \
    "$MEM_LIB/R2PV/V04R20SZ/model/verilog/IN22FDX_R2PV_WFVG_W00512B032M04C128.v" \

vlog -incr +nospecify -work work-vsim/ \
    $NETLIST_DIR/snitch_cluster.v
