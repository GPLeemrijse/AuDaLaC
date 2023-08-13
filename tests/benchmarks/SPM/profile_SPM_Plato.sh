echo "Profiling 570_2018..." | tee profile.txt
sudo -E env "PATH=$PATH" ncu $1 --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SPM.out testcases/dining/dining_5.invariantly_plato_starves.init | tee -a profile.txt
echo "Profiling 1878_7854..." | tee -a profile.txt
sudo -E env "PATH=$PATH" ncu $1 --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SPM.out testcases/dining/dining_6.invariantly_plato_starves.init | tee -a profile.txt
echo "Profiling 6198_29888..." | tee -a profile.txt
sudo -E env "PATH=$PATH" ncu $1 --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SPM.out testcases/dining/dining_7.invariantly_plato_starves.init | tee -a profile.txt
echo "Profiling 20466_111774..." | tee -a profile.txt
sudo -E env "PATH=$PATH" ncu $1 --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SPM.out testcases/dining/dining_8.invariantly_plato_starves.init | tee -a profile.txt
echo "Profiling 67590_412322..." | tee -a profile.txt
sudo -E env "PATH=$PATH" ncu $1 --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SPM.out testcases/dining/dining_9.invariantly_plato_starves.init | tee -a profile.txt
echo "Profiling 223230_1504368..." | tee -a profile.txt
sudo -E env "PATH=$PATH" ncu $1 --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SPM.out testcases/dining/dining_10.invariantly_plato_starves.init | tee -a profile.txt
echo "Profiling 737274_5439458..." | tee -a profile.txt
sudo -E env "PATH=$PATH" ncu $1 --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --print-details all --config-file off ./SPM.out testcases/dining/dining_11.invariantly_plato_starves.init | tee -a profile.txt
