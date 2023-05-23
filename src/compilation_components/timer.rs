use crate::transpilation_traits::*;
use indoc::formatdoc;
use std::collections::BTreeSet;

pub struct Timer {
    active: bool,
}

impl Timer {
    pub fn new(active: bool) -> Timer {
        Timer { active }
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
            Some(formatdoc! {"
				\tcudaEvent_t start, stop;
				\tcudaEventCreate(&start);
				\tcudaEventCreate(&stop);
				\tcudaEventRecord(start);
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
            Some(formatdoc! {"
				\tcudaEventRecord(stop);
				\tcudaEventSynchronize(stop);
				\tfloat ms = 0;
				\tcudaEventElapsedTime(&ms, start, stop);
				\tprintf(\"Total walltime: %0.2f ms\\n\", ms);
			"})
        } else {
            None
        }
    }
}
