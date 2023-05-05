use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use indoc::formatdoc;

pub struct InitFileReader {
}

impl CompileComponent for InitFileReader {
	fn add_includes(&self, set: &mut BTreeSet<&str>) {
		set.insert("<stdio.h>");
		set.insert("<vector>");
	}

	fn defines(&self) -> Option<String> { None }
	fn typedefs(&self) -> Option<String> { None }
	fn globals(&self) -> Option<String> { None }
	fn functions(&self) -> Option<String> { None }
	fn kernels(&self) -> Option<String> { None }
	
	fn pre_main(&self) -> Option<String> {
		Some(formatdoc!{"
			if (argc != 2) {{
				printf(\"Supply a .init file.\\n\");
				exit(1);
			}}
			
			std::vector<InitFile::StructInfo> structs = InitFile::parse(argv[1]);
		"})
	}
	
	fn main(&self) -> Option<String> { None }
	fn post_main(&self) -> Option<String> { None }
}