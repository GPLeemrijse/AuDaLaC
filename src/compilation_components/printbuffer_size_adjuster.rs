use crate::transpilation_traits::*;
use indoc::formatdoc;
use std::collections::BTreeSet;

pub struct PrintbufferSizeAdjuster {
    size: usize,
}

impl PrintbufferSizeAdjuster {
    pub fn new(size: usize) -> PrintbufferSizeAdjuster {
        PrintbufferSizeAdjuster { size }
    }
}

impl CompileComponent for PrintbufferSizeAdjuster {
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
        Some(formatdoc! {"
			\tcudaDeviceSetLimit(cudaLimitPrintfFifoSize, {});
		", 1024 * self.size})
    }

    fn main(&self) -> Option<String> {
        None
    }
    fn post_main(&self) -> Option<String> {
        None
    }
}
