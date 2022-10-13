use assert_cmd::Command;
use std::fs;
use std::str;

#[test]
#[ignore]
fn test_ast_fixtures() {
    let paths = fs::read_dir("tests/ast_fixtures").unwrap();

    for path in paths {
        let path_string = path.as_ref().unwrap().path().display().to_string();
        if !path_string.ends_with("_expected") {
            println!("Testing AST of '{}'", path_string);
            check_file_ast(&path_string);
        }
    }
}

fn check_file_ast(file: &str) {
    let expected_output = fs::read_to_string(format!("{}_expected", file))
        .expect("Could not open file with expected_output");

    let out = Command::cargo_bin("loplparser")
        .unwrap()
        .arg("--print")
        .arg(file)
        .output()
        .expect("Could not collect output of command.");
    let stdout = str::from_utf8(&out.stdout).unwrap();

    if stdout != expected_output {
        panic!("Result is:\n'{}'\nEXPECTED:\n'{}'", stdout, expected_output);
    }
}
