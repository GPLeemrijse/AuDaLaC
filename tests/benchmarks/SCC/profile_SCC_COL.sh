echo "Profiling 1000_1289..."
echo "Profiling 1000_1289..." > throughput.txt
sudo /usr/local/NVIDIA-Nsight-Compute-2023.2/target/linux-desktop-glibc_2_11_3-x64/ncu --print-summary per-gpu --config-file off /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/SCC_COL.out /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/testcases/random_col_1000_1289.init |& grep -E "DRAM Throughput|    kernel_" >> throughput.txt
echo "Profiling 3162_4157..."
echo "Profiling 3162_4157..." >> throughput.txt
sudo /usr/local/NVIDIA-Nsight-Compute-2023.2/target/linux-desktop-glibc_2_11_3-x64/ncu --print-summary per-gpu --config-file off /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/SCC_COL.out /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/testcases/random_col_3162_4157.init |& grep -E "DRAM Throughput|    kernel_" >> throughput.txt
echo "Profiling 10000_13021..."
echo "Profiling 10000_13021..." >> throughput.txt
sudo /usr/local/NVIDIA-Nsight-Compute-2023.2/target/linux-desktop-glibc_2_11_3-x64/ncu --print-summary per-gpu --config-file off /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/SCC_COL.out /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/testcases/random_col_10000_13021.init |& grep -E "DRAM Throughput|    kernel_"  >> throughput.txt
echo "Profiling 31623_40761..." >> throughput.txt
echo "Profiling 31623_40761..."
sudo /usr/local/NVIDIA-Nsight-Compute-2023.2/target/linux-desktop-glibc_2_11_3-x64/ncu --print-summary per-gpu --config-file off /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/SCC_COL.out /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/testcases/random_col_31623_40761.init |& grep -E "DRAM Throughput|    kernel_" >> throughput.txt
echo "Profiling 100000_129651..." >> throughput.txt
echo "Profiling 100000_129651..."
sudo /usr/local/NVIDIA-Nsight-Compute-2023.2/target/linux-desktop-glibc_2_11_3-x64/ncu --print-summary per-gpu --config-file off /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/SCC_COL.out /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/testcases/random_col_100000_129651.init |& grep -E "DRAM Throughput|    kernel_" >> throughput.txt
echo "Profiling 316228_411169..." >> throughput.txt
echo "Profiling 316228_411169..."
sudo /usr/local/NVIDIA-Nsight-Compute-2023.2/target/linux-desktop-glibc_2_11_3-x64/ncu --print-summary per-gpu --config-file off /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/SCC_COL.out /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/testcases/random_col_316228_411169.init |& grep -E "DRAM Throughput|    kernel_"  >> throughput.txt
echo "Profiling 1000000_1301143..." >> throughput.txt
echo "Profiling 1000000_1301143..."
sudo /usr/local/NVIDIA-Nsight-Compute-2023.2/target/linux-desktop-glibc_2_11_3-x64/ncu --print-summary per-gpu --config-file off /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/SCC_COL.out /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/testcases/random_col_1000000_1301143.init |& grep -E "DRAM Throughput|    kernel_"  >> throughput.txt
echo "Profiling 3162278_4110178..." >> throughput.txt
echo "Profiling 3162278_4110178..."
sudo /usr/local/NVIDIA-Nsight-Compute-2023.2/target/linux-desktop-glibc_2_11_3-x64/ncu --print-summary per-gpu --config-file off /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/SCC_COL.out /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/testcases/random_col_3162278_4110178.init |& grep -E "DRAM Throughput|    kernel_" >> throughput.txt
echo "Profiling 10000000_12996160..." >> throughput.txt
echo "Profiling 10000000_12996160..."
sudo /usr/local/NVIDIA-Nsight-Compute-2023.2/target/linux-desktop-glibc_2_11_3-x64/ncu --print-summary per-gpu --config-file off /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/SCC_COL.out /home/gpleemrijse/repos/ADL/tests/benchmarks/SCC/testcases/random_col_10000000_12996160.init |& grep -E "DRAM Throughput|    kernel_"
