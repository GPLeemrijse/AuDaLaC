use crate::compiler::compilation_traits::*;
use indoc::formatdoc;
use std::collections::BTreeSet;

pub fn compile(components: Vec<&dyn CompileComponent>) -> String {
    // Program sections
    let (
        mut includes,
        mut defines,
        mut typedefs,
        mut globals,
        mut functs,
        mut pre_main,
        mut post_main,
        mut main,
        mut kernels,
    ) = (
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
    );

    let mut includes_set: BTreeSet<&str> = BTreeSet::new();

    let mut sec2func_map: Vec<(
        &mut String,
        Box<dyn Fn(&dyn CompileComponent) -> Option<String>>,
    )> = vec![
        (&mut defines, Box::new(|c| c.defines())),
        (&mut typedefs, Box::new(|c| c.typedefs())),
        (&mut globals, Box::new(|c| c.globals())),
        (&mut functs, Box::new(|c| c.functions())),
        (&mut pre_main, Box::new(|c| c.pre_main())),
        (&mut post_main, Box::new(|c| c.post_main())),
        (&mut main, Box::new(|c| c.main())),
        (&mut kernels, Box::new(|c| c.kernels())),
    ];

    for c in components {
        for (sec, f) in &mut sec2func_map {
            if let Some(src) = f(c) {
                sec.push_str(&src);
            }
        }

        c.add_includes(&mut includes_set);
    }

    for i in includes_set {
        includes.push_str(&format!("#include {}\n", i));
    }

    formatdoc! {"
		{defines}

		{includes}

		{typedefs}

		{globals}

		{functs}

		{kernels}

		int main(int argc, char **argv) {{
		{pre_main}

		{main}

		{post_main}
		}}
	"}
}
