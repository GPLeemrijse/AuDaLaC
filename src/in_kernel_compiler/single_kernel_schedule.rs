use crate::ast::Program;
use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use indoc::formatdoc;

pub struct SingleKernelSchedule<'a> {
	program : &'a Program,
}

impl SingleKernelSchedule<'_> {
	const KERNEL_NAME : &str = "schedule_kernel";

	pub fn new<'a>(program : &'a Program) -> SingleKernelSchedule<'a> {
		SingleKernelSchedule {
			program
		}
	}

	fn kernel_arguments(&self) -> Vec<String> {
		self.program.structs.iter()
							.map(|s| format!("&gm_{}", s.name))
		 					.collect()
	}

	fn kernel_parameters(&self) -> Vec<String> {
		self.program.structs.iter()
							.map(|s| format!("{}* {}", s.name, s.name.to_lowercase()))
		 					.collect()
	}
}

impl CompileComponent for SingleKernelSchedule<'_> {
	fn add_includes(&self, _set: &mut BTreeSet<&str>) {
	}

	fn defines(&self) -> Option<String> {
		None
	}
	
	fn typedefs(&self) -> Option<String> {
		None
	}
	
	fn globals(&self) -> Option<String> {
		None
	}
	
	fn functions(&self) -> Option<String> {
		None
	}

	fn kernels(&self) -> Option<String> {
		let kernel_header = format!("__global__ void {}(", SingleKernelSchedule::KERNEL_NAME);
		let param_indent = format!(",\n{}{}",
								   "\t".repeat(kernel_header.len() / 4),
								   " ".repeat(kernel_header.len() % 4)
								  );
		let kernel_parameters = self.kernel_parameters().join(&param_indent);
		
		Some(formatdoc!{"
			{kernel_header}{kernel_parameters}){{

			}}
		"})
	}
	
	fn pre_main(&self) -> Option<String> {
		None
	}
	
	fn main(&self) -> Option<String> {
		let kernel_name = SingleKernelSchedule::KERNEL_NAME;
		let arg_array = self.kernel_arguments().join(&format!(",\n\t\t"));

		Some(formatdoc!{"
			\tinst_size nrof_instances = 10000; // TODO
			\tvoid* {kernel_name}_args[] = {{
			\t	{arg_array}
			\t}};
			\tauto dims = ADL::get_launch_dims(nrof_instances, (void*){kernel_name});

			\tCHECK(
			\t	cudaLaunchCooperativeKernel(
			\t		(void*){kernel_name},
			\t		std::get<0>(dims),
			\t		std::get<1>(dims),
			\t		{kernel_name}_args
			\t	)
			\t);
			\tCHECK(cudaDeviceSynchronize());
		"})
	}
	
	fn post_main(&self) -> Option<String> {
		None
	}
}