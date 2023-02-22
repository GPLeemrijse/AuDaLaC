use std::fs;
use indoc::formatdoc;

#[test]
fn test_generate_init_file_reachability() {
	let output_loc = "/tmp/reachability_init_file";
	generate_init_file("tests/ast_fixtures/reachability", output_loc);
	let result = fs::read_to_string(output_loc).expect("Could not open init file.");
	assert_eq!(result,
		formatdoc!(
			"ADL structures 2
			Node Bool
			Edge Node Node
			Node instances 5
			0
			1
			0
			0
			0
			Edge instances 5
			0 0
			1 3
			4 3
			2 4
			3 4
			")
	);
}

#[test]
fn test_generate_init_file_circular() {
	let output_loc = "/tmp/circular_init";
	generate_init_file("tests/init_file_fixtures/circular_dependency.adl", output_loc);
	let result = fs::read_to_string(output_loc).expect("Could not open init file.");
	assert_eq!(result,
		formatdoc!(
			"ADL structures 2
			Node Bool Node
			Edge Node Node
			Node instances 5
			0 0
			1 1
			0 3
			0 0
			0 2
			Edge instances 5
			0 0
			1 3
			4 3
			2 4
			3 4
			")
	);
}

fn generate_init_file(file_in: &str, file_out: &str) {
    let _ = std::process::Command::new("target/debug/adl")
        .arg("-o")
        .arg(file_out)
        .arg("--init_file")
        .arg(file_in)
        .output()
        .expect("ADL could not generate init file.");
}