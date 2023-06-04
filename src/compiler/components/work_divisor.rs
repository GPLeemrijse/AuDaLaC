use crate::compiler::compilation_traits::*;
use crate::parser::ast::Program;
use indoc::formatdoc;
use std::collections::BTreeSet;
use std::collections::HashMap;

pub enum DivisionStrategy {
    BlockSizeIncrease,
    GridSizeIncrease,
}

pub struct WorkDivisor<'a> {
    program: &'a Program,
    instances_per_thread: usize,
    threads_per_block: usize,
    allocated_per_instance: &'a HashMap<String, usize>,
    division_strategy: DivisionStrategy,
    print_unstable: bool,
}

impl<'a> WorkDivisor<'a> {
    pub fn new(
        program: &'a Program,
        instances_per_thread: usize,
        threads_per_block: usize,
        allocated_per_instance: &'a HashMap<String, usize>,
        division_strategy: DivisionStrategy,
        print_unstable: bool,
    ) -> WorkDivisor<'a> {
        WorkDivisor {
            program,
            instances_per_thread,
            threads_per_block,
            allocated_per_instance,
            division_strategy,
            print_unstable,
        }
    }

    pub fn get_dims(&self, kernel_name: &str) -> String {
        formatdoc!(
            "
			ADL::get_launch_dims((max_nrof_executing_instances + I_PER_THREAD - 1) / I_PER_THREAD, (void*){kernel_name})"
        )
    }

    pub fn execute_step(&self, kernel_name: &String) -> String {
        if self.print_unstable {
            formatdoc!("executeStep<{kernel_name}>(nrof_instances, grid, block, &stable, \"{kernel_name}\");")
        } else {
            formatdoc!("executeStep<{kernel_name}>(nrof_instances, grid, block, &stable);")
        }
    }

    fn loop_header(&self) -> String {
        use DivisionStrategy::*;

        match self.division_strategy {
            BlockSizeIncrease => {
                if self.instances_per_thread > 1 {
                    formatdoc!("
						\tfor(int i = 0; i < I_PER_THREAD; i++){{
						\t\tconst RefType self = block.size() * (i + grid.block_rank() * I_PER_THREAD) + block.thread_rank();
						\t\tif (self >= nrof_instances) break;"
					)
                } else {
                    formatdoc!(
                        "
						\tconst RefType self = block.size() * grid.block_rank() + block.thread_rank();
						\tif (self >= nrof_instances) return;"
                    )
                }
            }
            GridSizeIncrease => {
                if self.instances_per_thread > 1 {
                    formatdoc!(
                        "
						\tfor(int i = 0; i < I_PER_THREAD; i++){{
						\t\tconst RefType self = grid.thread_rank() + i * grid.size();
						\t\tif (self >= nrof_instances) break;"
                    )
                } else {
                    formatdoc!(
                        "
						\tconst RefType self = grid.thread_rank();
						\tif (self >= nrof_instances) return;"
                    )
                }
            }
        }
    }
}

impl CompileComponent for WorkDivisor<'_> {
    fn add_includes(&self, set: &mut BTreeSet<&str>) {
        set.insert("<cooperative_groups.h>");
    }

    fn defines(&self) -> Option<String> {
        let ipt = self.instances_per_thread;
        let tpb = self.threads_per_block;

        Some(formatdoc! {"
			#define I_PER_THREAD {ipt}
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

        Some(formatdoc!("
			typedef void(*step_func)(RefType, bool*);
			template <step_func Step>
			__device__ void executeStep(inst_size nrof_instances, grid_group grid, thread_block block, bool* stable{step_str_par}){{
			{loop_header}

			{execution}{}
			}}
		", if self.instances_per_thread > 1 { "\n	}" } else {""}))
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
