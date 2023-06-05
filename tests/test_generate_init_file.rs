use indoc::formatdoc;
use std::fs;

#[test]
fn test_generate_init_file_shortest_path() {
    let output_loc = "tests/init_file_fixtures/shortest_path.init";
    generate_init_file("tests/init_file_fixtures/shortest_path.adl", output_loc);
    let result = fs::read_to_string(output_loc).expect("Could not open init file.");
    assert_eq!(
        result,
        formatdoc!(
            "ADL structures 2
			Edge Node Node Nat
			Node Int Edge
			Edge instances 5
			0 0 0
			1 3 1
			1 3 2
			1 2 3
			2 1 1
			Node instances 4
			0 0
			0 0
			-1 0
			-1 0
			"
        )
    );
}

#[test]
fn test_generate_init_file_reachability() {
    let output_loc = "tests/init_file_fixtures/reachability.init";
    generate_init_file("tests/ast_fixtures/reachability", output_loc);
    let result = fs::read_to_string(output_loc).expect("Could not open init file.");
    assert_eq!(
        result,
        formatdoc!(
            "ADL structures 2
			Edge Node Node
			Node Bool
			Edge instances 5
			0 0
			1 3
			4 3
			2 4
			3 4
			Node instances 5
			0
			1
			0
			0
			0
			"
        )
    );
}

#[test]
fn test_generate_init_file_circular() {
    let output_loc = "tests/init_file_fixtures/circular_dependency.init";
    generate_init_file(
        "tests/init_file_fixtures/circular_dependency.adl",
        output_loc,
    );
    let result = fs::read_to_string(output_loc).expect("Could not open init file.");
    assert_eq!(
        result,
        formatdoc!(
            "ADL structures 2
			Edge Node Node
			Node Bool Node
			Edge instances 5
			0 0
			1 3
			4 3
			2 4
			3 4
			Node instances 5
			0 0
			1 1
			0 3
			0 0
			0 2
			"
        )
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
