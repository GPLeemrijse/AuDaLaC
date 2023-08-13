echo "Profiling 1000_1289..." | tee profile.txt
sudo -E env "PATH=$PATH" ncu $1 --graph-profiling graph --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SCC_COL.out testcases/random_col_1000_1289.init | tee -a profile.txt
echo "Profiling 3162_4157..." | tee -a profile.txt
sudo -E env "PATH=$PATH" ncu $1 --graph-profiling graph --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SCC_COL.out testcases/random_col_3162_4157.init | tee -a profile.txt
echo "Profiling 10000_13021..." | tee -a profile.txt
sudo -E env "PATH=$PATH" ncu $1 --graph-profiling graph --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SCC_COL.out testcases/random_col_10000_13021.init | tee -a profile.txt
echo "Profiling 31623_40761..." | tee -a profile.txt
sudo -E env "PATH=$PATH" ncu $1 --graph-profiling graph --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SCC_COL.out testcases/random_col_31623_40761.init | tee -a profile.txt
echo "Profiling 100000_129651..." | tee -a profile.txt
sudo -E env "PATH=$PATH" ncu $1 --graph-profiling graph --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SCC_COL.out testcases/random_col_100000_129651.init | tee -a profile.txt
echo "Profiling 316228_411169..." | tee -a profile.txt
sudo -E env "PATH=$PATH" ncu $1 --graph-profiling graph --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SCC_COL.out testcases/random_col_316228_411169.init | tee -a profile.txt
echo "Profiling 1000000_1301143..." | tee -a profile.txt
sudo -E env "PATH=$PATH" ncu $1 --graph-profiling graph --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SCC_COL.out testcases/random_col_1000000_1301143.init | tee -a profile.txt
echo "Profiling 3162278_4110178..." | tee -a profile.txt
sudo -E env "PATH=$PATH" ncu $1 --graph-profiling graph --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SCC_COL.out testcases/random_col_3162278_4110178.init | tee -a profile.txt
echo "Profiling 10000000_12996160..." | tee -a profile.txt
sudo -E env "PATH=$PATH" ncu $1 --graph-profiling graph --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SCC_COL.out testcases/random_col_10000000_12996160.init | tee -a profile.txt