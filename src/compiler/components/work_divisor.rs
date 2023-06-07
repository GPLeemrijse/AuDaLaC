use crate::compiler::compilation_traits::*;
use indoc::formatdoc;
use std::collections::BTreeSet;


pub enum DivisionStrategy {
    Dynamic,
}

pub struct WorkDivisor {
    threads_per_block: usize,
    division_strategy: DivisionStrategy,
    print_unstable: bool,
}

impl<'a> WorkDivisor {
    pub fn new(
        threads_per_block: usize,
        division_strategy: DivisionStrategy,
        print_unstable: bool,
    ) -> WorkDivisor {
        WorkDivisor {
            threads_per_block,
            division_strategy,
            print_unstable,
        }
    }

    pub fn get_dims(&self, kernel_name: &str) -> String {
        formatdoc!(
            "
            get_launch_dims(max_nrof_executing_instances, (void*){kernel_name})"
        )
    }

    pub fn execute_step(&self, kernel_name: &String, nrof_instances: String) -> String {
        if self.print_unstable {
            formatdoc!("executeStep<{kernel_name}>({nrof_instances}, grid, block, &stable, \"{kernel_name}\");")
        } else {
            formatdoc!("executeStep<{kernel_name}>({nrof_instances}, grid, block, &stable);")
        }
    }

    fn loop_header(&self) -> String {
        use DivisionStrategy::*;

        match self.division_strategy {
            Dynamic => {
                formatdoc!(
                    "
                    \tfor(RefType self = grid.thread_rank(); self < nrof_instances; self += grid.size()){{"
                )
            }
        }
    }

    fn launch_dims_function(&self) -> String {
        formatdoc!{"
            __host__ std::tuple<dim3, dim3> get_launch_dims(inst_size max_nrof_executing_instances, const void* kernel){{
              int numBlocksPerSm = 0;
              int tpb = THREADS_PER_BLOCK;

              cudaDeviceProp deviceProp;
              cudaGetDeviceProperties(&deviceProp, 0);
              cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, tpb, 0);
              
              int max_blocks = deviceProp.multiProcessorCount*numBlocksPerSm;
              int wanted_blocks = (max_nrof_executing_instances + tpb - 1)/tpb;
              int used_blocks = max(max_blocks, wanted_blocks);

              fprintf(stderr, \"Launching %u/%u blocks of %u threads = %u threads. Resulting in max %u instances per thread.\\n\", used_blocks, max_blocks, tpb, used_blocks * tpb, max_nrof_executing_instances / (used_blocks * tpb));

              dim3 dimBlock(tpb, 1, 1);
              dim3 dimGrid(used_blocks, 1, 1);
              return std::make_tuple(dimGrid, dimBlock);
            }}

        "}
    }

    fn execute_step_function(&self) -> String {
        let loop_header = self.loop_header();
        let execution = if self.print_unstable {
            formatdoc!(
                "
                \t\tbool __stable = true;
                \t\tStep(self, &__stable);
                \t\tif (!__stable) {{
                \t\t\tprintf(\"Step %s was unstable (instance %u)\\n\", step_str, self);
                \t\t\t*stable = false;
                \t\t}}"
            )
        } else {
            formatdoc!(
                "
                \t\tStep(self, stable);"
            )
        };

        let step_str_par = if self.print_unstable {
            ", const char* step_str"
        } else {
            ""
        };

        formatdoc!("
            typedef void(*step_func)(RefType, bool*);
            template <step_func Step>
            __device__ void executeStep(inst_size nrof_instances, grid_group grid, thread_block block, bool* stable{step_str_par}){{
            {loop_header}

            {execution}
                }}
            }}

        ")
    }
}

impl CompileComponent for WorkDivisor {
    fn add_includes(&self, set: &mut BTreeSet<&str>) {
        set.insert("<cooperative_groups.h>");
        set.insert("<tuple>");
    }

    fn defines(&self) -> Option<String> {
        let tpb = self.threads_per_block;

        Some(formatdoc! {"
			#define THREADS_PER_BLOCK {tpb}
		"})
    }
    fn typedefs(&self) -> Option<String> {
        None
    }

    fn globals(&self) -> Option<String> {
        None
    }

    fn functions(&self) -> Option<String> {
        Some(
            self.execute_step_function()
            +
            &self.launch_dims_function()
        )
    }

    fn kernels(&self) -> Option<String> {
        None
    }

    fn pre_main(&self) -> Option<String> {
        None
    }

    fn main(&self) -> Option<String> {
        None
    }

    fn post_main(&self) -> Option<String> {
        None
    }
}
