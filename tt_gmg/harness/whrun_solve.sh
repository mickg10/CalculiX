#!/bin/bash
# TT-GMG validation harness (preserved from the tt-quietbox /tmp on 2026-07-10; /tmp is ephemeral and a BMC
# cold-cycle wipes it, so this repo copy is the durable source). Rebuilds libtt_spmv from /tmp/tt_spmv.cpp,
# JIT-copies /tmp/kernels/*.cpp, runs solve_driver.py with the eig-deflation env + SPMV_TIMING, 850s timeout.
# HAZARD (learned 2026-07-10): NEVER kill a multi-chip run mid-workload (kill -9). A hung workload left this way
# -- or even a timeout-killed L1-overflow hang -- leaves the Tensix cores in a bad run-state so that EVERY later
# run (incl. known-good kernels) hangs at the first apply with TT_FATAL "run_mailbox 0x40". Recovery is a BMC
# cold-cycle only (10.0.0.48; tt-smi -r is FORBIDDEN). Let the 850s timeout (exit=124) end a hang instead.
FAB="${1:-FABRIC_1D}"; NCH="${2:-32}"
MT=/home/ttuser/src/tt-metal; BR=$MT/build_Release/build_Release
D=$MT/tt_metal/programming_examples/spmv_mac; mkdir -p $D/kernels 2>/dev/null; cp /tmp/kernels/*.cpp $D/kernels/ 2>/dev/null
mkdir -p $MT/generated/fabric 2>/dev/null
export HOME=/tmp/whhome; mkdir -p $HOME/.cache/tt-metal-cache
ZS=$(command -v zstd || echo zstd)
$ZS -d -f -q /tmp/row236_real_op.bin.zst -o /tmp/row236_real_op.bin 2>/dev/null
$ZS -d -f -q /tmp/row236_nbr.bin.zst -o /tmp/row236_nbr.bin 2>/dev/null
cp /tmp/tt_spmv.cpp /tmp/ttb.cpp
python3 -c "
s=open('/tmp/ttb.cpp').read()
s=s.replace('#include <tt-metalium/distributed.hpp>','#include <tt-metalium/distributed.hpp>\n#include <tt-metalium/experimental/fabric/fabric.hpp>')
s=s.replace('K->dev = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape(1, K->NCHIP)));','if(K->NCHIP>1) tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::$FAB);\n    K->dev = distributed::MeshDevice::create(distributed::MeshDeviceConfig(K->NCHIP==32 ? distributed::MeshShape(8,4) : distributed::MeshShape(1, K->NCHIP)));')
s=s.replace('CoreRangeSet all_set(CoreRange({0,0},{grid.x-1,grid.y-1}));','CoreRangeSet all_set = K->dev->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::SubDeviceId{0});')
s=s.replace('split_work_to_cores(grid, n_local, true)','split_work_to_cores(all_set, n_local, true)')
open('/tmp/ttb.cpp','w').write(s)"
find $MT -type d -name tt-metalium 2>/dev/null|xargs -n1 dirname 2>/dev/null|sort -u|sed "s/^/-I/">/tmp/whi.txt
find $MT/tt_metal/third_party $MT/.cpmcache $BR/include -maxdepth 4 -type d -name include 2>/dev/null|sed "s/^/-I/">>/tmp/whi.txt
ls -d $MT/build_Release/build_Release/include $MT/build_Release/build_Release/include/metalium-thirdparty $MT/.cpmcache/reflect/*/ $MT/.cpmcache/*/*/ 2>/dev/null | sed "s|^|-I|" >> /tmp/whi.txt
NLD=$(find $MT $BR -name json_fwd.hpp 2>/dev/null|head -1|xargs dirname 2>/dev/null|xargs dirname 2>/dev/null)
API="-std=c++20 -fPIC -I$MT/tt_metal -I$MT -I$MT/ttnn -I$MT/tt_stl -I$MT/tt_metal/hw/inc -I$MT/tt_metal/include -I$MT/tt_metal/third_party/umd/device/api -I$MT/tt_metal/hostdevcommon/api"
rm -f /tmp/libtt_spmv.so   # FAIL-CLOSED: delete first so "OK" means g++ actually rebuilt it (not a stale .so)
g++ $API -I$NLD $(tr "\n" " " </tmp/whi.txt) -DFMT_HEADER_ONLY -shared /tmp/ttb.cpp -o /tmp/libtt_spmv.so -L$BR/lib -Wl,-rpath,$BR/lib -ltt_metal -ldevice -ltt_stl 2>/tmp/whb.err
gpp_rc=$?
echo "WH rebuild=$([ $gpp_rc -eq 0 ] && [ -f /tmp/libtt_spmv.so ]&&echo OK||echo FAIL) arch=WORMHOLE nchip=$NCH fab=$FAB"
{ [ $gpp_rc -eq 0 ] && [ -f /tmp/libtt_spmv.so ]; } || { grep -aiE "error:|FabricConfig" /tmp/whb.err|head -4|cut -c1-120; exit 1; }
PY=""; for p in /home/ttuser/src/tt-metal/python_env/bin/python /usr/bin/python3 python3; do $p -c "import numpy" 2>/dev/null && { PY=$p; break; }; done
[ -z "$PY" ] && { python3 -m pip install --quiet --user numpy 2>/dev/null; python3 -c "import numpy" 2>/dev/null && PY=python3; }
echo "PY=$PY numpy=$($PY -c 'import numpy;print(numpy.__version__)' 2>&1|tail -1)"
export TT_METAL_HOME=$MT TT_METAL_RUNTIME_ROOT=$MT LD_LIBRARY_PATH=$BR/lib:$BR/ttnn:$BR/tt_metal:$LD_LIBRARY_PATH TT_METAL_LOGGER_LEVEL=ERROR
export OMP_NUM_THREADS=16 SPMV_NCHIP=$NCH
ulimit -n 1048576 2>/dev/null
echo "WH-RUN $(date +%T)"
export SPMV_TIMING=1 SPMV_REUSE_WORKLOAD=1 GMG_DEFL_CORR=1 GMG_DEFL_EIG=1 GMG_DEFL_K=24 GMG_HYBRID_TOL=1e-2; timeout 850 $PY /tmp/solve_driver.py > /tmp/wh.out 2>/tmp/wh.err
echo "exit=$?"
grep -aE "GRID|EXEC|DIAG3 DONE|init rc|COMPARE" /tmp/wh.out 2>/dev/null | cut -c1-140
grep -aviE "0x[0-9a-f]{6}" /tmp/wh.err 2>/dev/null|grep -aiE "throw|fatal|fabric|ethernet|what\("|head -2|cut -c1-110
echo "WH-RUN DONE $(date +%T)"
