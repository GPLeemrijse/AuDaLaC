use crate::compiler::CompileComponent;
use indoc::formatdoc;
use std::collections::BTreeSet;

pub struct Timer {
	active: bool,
	stream: Option<String>
}

impl Timer {
	pub fn new(active: bool, stream: Option<String>) -> Timer {
		Timer {
			active,
			stream
		}
	}
}

impl CompileComponent for Timer {
	fn add_includes(&self, _set: &mut BTreeSet<&str>) {}

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
		None
	}

	fn pre_main(&self) -> Option<String> {
		if self.active {
			let stream_arg = self.stream.clone().map_or("".to_string(), |s| format!(", {s}"));
			Some(formatdoc! {"

				\tcudaEvent_t start, stop;
				\tcudaEventCreate(&start);
				\tcudaEventCreate(&stop);
				\tcudaEventRecord(start{stream_arg});
			"})
		} else {
			None
		}
	}

	fn main(&self) -> Option<String> {
		None
	}
	fn post_main(&self) -> Option<String> {
		if self.active {
			let stream_arg = self.stream.clone().map_or("".to_string(), |s| format!(", {s}"));
			Some(formatdoc! {"
				\tcudaEventRecord(stop{stream_arg});
				\tcudaEventSynchronize(stop);
				\tfloat ms = 0;
				\tcudaEventElapsedTime(&ms, start, stop);
				\tprintf(\"Total walltime GPU: %0.6f ms\\n\", ms);
			"})
		} else {
			Some("\tCHECK(cudaDeviceSynchronize());".to_string())
		}
	}
}
