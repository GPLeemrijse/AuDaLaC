use crate::ast::*;
use std::collections::HashMap;

// All steps in schedule have been defined
// Numeric operators always between: Nat Nat, Nat Int, Int Int
// Unique step names in Structs
// No declared variables in steps with the same name as parameters

#[derive(Debug, PartialEq, Eq)]
pub enum ValidationError {
    StructDefinedTwice(ErrorContext),
    StepDefinedTwice(ErrorContext),
    ParameterDefinedTwice(ErrorContext, String), //parameter name
    VariableAlreadyDeclared(ErrorContext, String), //var name
    UndefinedType(ErrorContext, String),         //attempted type name
    UndefinedField(ErrorContext, String, String), //parent name, field name
}

#[derive(Debug, PartialEq, Eq)]
pub struct ErrorContext {
    struct_name: Option<String>,
    step_name: Option<String>,
}

impl ErrorContext {
    fn from_block_context(context: &BlockEvaluationContext) -> ErrorContext {
        let mut struct_name: Option<String> = None;
        let mut step_name: Option<String> = None;
        if let Some(s) = context.current_struct_name {
            struct_name = Some(s.clone());
        }
        if let Some(s) = context.current_step_name {
            step_name = Some(s.clone());
        }

        return ErrorContext {
            struct_name: struct_name,
            step_name: step_name,
        };
    }
}

// Since we are using references to the AST datastructure,
// we introduce the 'ast lifetime.
/// An organised collection of references to the `Program` AST data
#[derive(Debug, PartialEq, Eq)]
struct StaticProgramData<'ast> {
    struct_data: HashMap<&'ast String, StaticStructData<'ast>>,
    schedule: &'ast Schedule,
}

/// An organised collection of references to the `LoplStruct` AST data
#[derive(Debug, Eq, PartialEq)]
struct StaticStructData<'ast> {
    name: &'ast String,
    parameters: HashMap<&'ast String, &'ast Type>,
    steps: HashMap<&'ast String, &'ast Vec<Box<Stat>>>,
}

struct BlockEvaluationContext<'ast> {
    current_struct_name: Option<&'ast String>,
    current_step_name: Option<&'ast String>,
    structs: &'ast HashMap<&'ast String, StaticStructData<'ast>>,
    vars: Vec<HashMap<&'ast String, &'ast Type>>,
    errors: Vec<ValidationError>,
}

impl<'eval, 'ast> BlockEvaluationContext<'ast> {
    fn add_to_var_scope(&'eval mut self, s: &'ast String, t: &'ast Type) {
        self.vars.last_mut().unwrap().insert(s, t);
    }

    fn push_var_scope(&mut self) {
        self.vars.push(HashMap::new());
    }

    fn pop_var_scope(&mut self) {
        self.vars.pop();
    }
}

pub fn validate_ast<'ast>(ast: &'ast Program) -> Vec<ValidationError> {
    match extract_static_program_data(ast) {
        Err(e) => return e,
        Ok(static_data) => return check_declared_before_use(&static_data),
    }
}

/// Organises a `Program` AST into an easily accessible `StaticProgramData` struct
/// Returns an error result when:
/// - a struct is defined twice
/// - a parameter is defined twice for the same struct
/// - a step is defined twice for the same struct
fn extract_static_program_data<'ast>(
    ast: &'ast Program,
) -> Result<StaticProgramData, Vec<ValidationError>> {
    let mut errors: Vec<ValidationError> = Vec::new();

    let mut program_data = StaticProgramData {
        struct_data: HashMap::new(),
        schedule: &ast.schedule,
    };
    for s in &ast.structs {
        let mut parameters_map: HashMap<&'ast String, &'ast Type> = HashMap::new();
        for p in &s.parameters {
            if parameters_map.contains_key(&p.0) {
                errors.push(ValidationError::ParameterDefinedTwice(
                    ErrorContext {
                        struct_name: Some(s.name.clone()),
                        step_name: None,
                    },
                    p.0.clone(),
                ));
            } else {
                parameters_map.insert(&p.0, &p.1);
            }
        }

        let mut steps_map: HashMap<&'ast String, &'ast Vec<Box<Stat>>> = HashMap::new();
        for step in &s.steps {
            if steps_map.contains_key(&step.name) {
                errors.push(ValidationError::StepDefinedTwice(ErrorContext {
                    struct_name: Some(s.name.clone()),
                    step_name: Some(step.name.clone()),
                }));
            } else {
                steps_map.insert(&step.name, &step.statements);
            }
        }

        let struct_data = StaticStructData {
            name: &s.name,
            parameters: parameters_map,
            steps: steps_map,
        };
        if program_data.struct_data.contains_key(struct_data.name) {
            errors.push(ValidationError::StructDefinedTwice(ErrorContext {
                struct_name: Some(struct_data.name.clone()),
                step_name: None,
            }));
        } else {
            program_data
                .struct_data
                .insert(struct_data.name, struct_data);
        }
    }

    return if errors.is_empty() {
        Ok(program_data)
    } else {
        Err(errors)
    };
}

fn check_declared_before_use<'ast>(data: &StaticProgramData<'ast>) -> Vec<ValidationError> {
    let mut context = BlockEvaluationContext {
        current_struct_name: None,
        current_step_name: None,
        structs: &data.struct_data,
        vars: Vec::new(),
        errors: Vec::new(),
    };

    // Test each struct
    for s in data.struct_data.values() {
        context.current_struct_name = Some(s.name);

        // Create a fresh scope for the Struct's parameters
        context.push_var_scope();

        // Add the Struct's parameters to the new scope
        for (s, t) in &s.parameters {
            context.add_to_var_scope(s, t);
        }

        // Test each step
        for (stepname, step) in &s.steps {
            context.current_step_name = Some(stepname);

            // Each step also gets its own context
            context.push_var_scope();

            check_statement_block(step, &mut context);

            context.pop_var_scope();
        }
        context.current_step_name = None;

        // The struct's parameters leave the scope
        context.pop_var_scope();
    }
    context.current_struct_name = None;

    return context.errors;
}

fn check_statement_block<'ast>(
    block: &'ast Vec<Box<Stat>>,
    context: &mut BlockEvaluationContext<'ast>,
) {
    for stmt in block {
        use crate::ast::Stat::*;

        match &**stmt {
            Declaration(t, id, exp) => {
                check_type_is_defined(t, context);
                // id is not used before
                if let Some(_) = get_id_type(&id, &context.vars) {
                    context
                        .errors
                        .push(ValidationError::VariableAlreadyDeclared(
                            ErrorContext::from_block_context(context),
                            id.clone(),
                        ));
                }
                // Typecheck exp with t
                // Make something like validate_exp(exp)
                // Checking defines and var fields etc.
                //todo!();

                context
                    .vars
                    .last_mut()
                    .expect("There should be at least one context.")
                    .insert(id, t);
            }
            Assignment(parts, exp) => {
                let field_type = get_var_type(parts, context);

                // Typecheck field_type with exp
                //todo!();
            }
            IfThen(cond, statements) => {
                // Typecheck cond to be boolean
                //todo!();

                context.push_var_scope();

                check_statement_block(statements, context);

                context.pop_var_scope();
            }
        }
    }
}

// Returns the type of a `Var`: part1.part2.part3
fn get_var_type<'ast>(
    parts: &'ast Vec<String>,
    context: &mut BlockEvaluationContext<'ast>,
) -> Option<&'ast Type> {
    debug_assert!(parts.len() > 0);
    // Start in current scope

    let first_part = get_id_type(&parts[0], &context.vars);
    if let None = first_part {
        // INSERT ERROR!!
        return None;
    }

    if parts.len() == 1 {
        return first_part; // We know it is not None
    }

    let mut search_space: &HashMap<&'ast String, &'ast Type>;

    if let Some(t) = first_part {
        match t {
            Type::NamedType(s) => {
                search_space = &context.structs.get(s).unwrap().parameters;
            }
            _ => {
                context.errors.push(ValidationError::UndefinedField(
                    ErrorContext::from_block_context(context),
                    parts[0].clone(),
                    parts[1].clone(),
                ));
                return None;
            }
        }
    } else {
        unreachable!()
    }

    for i in 1..parts.len() {
        match search_space.get(&parts[i]) {
            None => {
                context.errors.push(ValidationError::UndefinedField(
                    ErrorContext::from_block_context(context),
                    parts[i - 1].clone(),
                    parts[i].clone(),
                ));
                return None;
            }
            Some(t) => {
                if i == parts.len() - 1 {
                    return Some(t);
                } else {
                    match t {
                        Type::NamedType(s) => {
                            search_space = &context.structs.get(s).unwrap().parameters;
                        }
                        _ => {
                            context.errors.push(ValidationError::UndefinedField(
                                ErrorContext::from_block_context(context),
                                parts[i - 1].clone(),
                                parts[i].clone(),
                            ));
                            return None;
                        }
                    }
                }
            }
        }
    }

    unreachable!("This should not be reachable!");
}

fn check_type_is_defined<'ast>(t: &'ast Type, context: &mut BlockEvaluationContext<'ast>) {
    if let Type::NamedType(s) = t {
        if !context.structs.contains_key(s) {
            context.errors.push(ValidationError::UndefinedType(
                ErrorContext::from_block_context(context),
                s.clone(),
            ));
        }
    }
}

fn get_id_type<'ast>(
    id: &'ast String,
    vars: &Vec<HashMap<&'ast String, &'ast Type>>,
) -> Option<&'ast Type> {
    for v in vars {
        if let Some(t) = v.get(id) {
            return Some(t);
        }
    }
    return None;
}

fn check_type(exp: Exp, expected_type: Type, containing_struct_type: Type) {
    use crate::ast::Exp::*;
    use crate::ast::Literal::*;
    use crate::ast::Type::*;
    match exp {
        Lit(NumLit(n)) => match expected_type {
            NatType => {
                assert!(
                    n <= u32::MAX.into(),
                    "{} does not fit in unsigned 32 bits.",
                    n
                );
                assert!(n >= 0, "{} is not a natural number.", n);
            }
            IntType => assert!(
                n >= i32::MIN.into() && n <= i32::MAX.into(),
                "{} does not fit in signed 32 bits.",
                n
            ),
            _ => panic!(
                "Expected type '{:?}', got 'Numeric Literal'.",
                expected_type
            ),
        },
        Lit(BoolLit(_)) => assert!(
            expected_type == BoolType,
            "Expected type '{:?}', got 'Bool'.",
            expected_type
        ),
        Lit(StringLit(_)) => assert!(
            expected_type == StringType,
            "Expected type '{:?}', got 'String'.",
            expected_type
        ),
        Lit(NullLit) => assert!(
            matches!(expected_type, NamedType { .. }),
            "Expected type '{:?}', got 'null'.",
            expected_type
        ),
        Lit(ThisLit) => assert!(
            expected_type == containing_struct_type,
            "Expected type '{:?}', got '{:?}'.",
            expected_type,
            containing_struct_type
        ),
        _ => unimplemented!(),
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::Literal::*;
    use crate::ast::Type::*;
    use crate::ast::*;
    use crate::ast_validator::check_type;
    use crate::ast_validator::extract_static_program_data;
    use crate::ast_validator::validate_ast;
    use crate::ast_validator::ErrorContext;
    use crate::ast_validator::Type;
    use crate::ast_validator::ValidationError;
    use crate::ast_validator::ValidationError::*;
    use crate::ProgramParser;
    use std::collections::HashMap;

    #[test]
    fn test_validate_double_decl() {
        let program_string = r#"
		struct Node (reachable : Bool) {
    		init {
        		Nat n1 := 1;
        		Nat n1 := 2;
        		Nat n2 := 3;
        	}
        	step_node {
        		Nat n2 := 0;
        		Int n3 := 10;
        		if true then {
        			Int n3 := 5;
        		}
        	}
        }
        init < Fix(step_node)
		"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let errors = validate_ast(&program);
        println!("{:#?}", errors);
        assert_eq!(errors.len(), 2);
        assert_eq!(
            errors[0],
            VariableAlreadyDeclared(
                ErrorContext {
                    struct_name: Some("Node".to_string()),
                    step_name: Some("step_node".to_string())
                },
                "n3".to_string()
            )
        );

        assert_eq!(
            errors[1],
            VariableAlreadyDeclared(
                ErrorContext {
                    struct_name: Some("Node".to_string()),
                    step_name: Some("init".to_string())
                },
                "n1".to_string()
            )
        );
    }

    #[test]
    fn test_extract() {
        let program_string = r#"
		struct Node (reachable : Bool) {
    		init {
        		Node s1 := Node(true);
        	}
        	step_node {
        		Nat n := 0;
        	}
        }
        struct Edge (s: Node, t: Node, w: Int) {
        	init {

        	}

        	step_edge {

        	}
        }

        init < Fix(step_edge < step_node)
		"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let static_data = extract_static_program_data(&program).expect("Validation errors found.");
        let struct_data = static_data.struct_data;
        let node_data = struct_data
            .get(&"Node".to_string())
            .expect("Did not catch Node");
        let edge_data = struct_data
            .get(&"Edge".to_string())
            .expect("Did not catch Edge");
        let node_params = &node_data.parameters;
        let edge_params = &edge_data.parameters;

        fn test_param<'ast>(m: &HashMap<&'ast String, &'ast Type>, name: &str, t: Type) {
            assert!(**m.get(&name.to_string()).unwrap() == t);
        }

        test_param(node_params, "reachable", BoolType);
        test_param(edge_params, "s", NamedType("Node".to_string()));
        test_param(edge_params, "t", NamedType("Node".to_string()));
        test_param(edge_params, "w", IntType);

        assert!(node_data.steps.contains_key(&"init".to_string()));
        assert!(node_data.steps.contains_key(&"step_node".to_string()));
        assert!(edge_data.steps.contains_key(&"init".to_string()));
        assert!(edge_data.steps.contains_key(&"step_edge".to_string()));
        assert!(matches!(*static_data.schedule, Schedule::Sequential { .. }));
    }

    #[test]
    fn test_extract_double_struct() {
        let program_string = r#"struct A(){init{}} struct A(){} init"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let static_data_result = extract_static_program_data(&program);
        match static_data_result {
            Ok(r) => panic!("Expected a StructDefinedTwice failure, got: {:?}.", r),
            Err(errors) => {
                assert!(errors.len() == 1);
                assert!(
                    errors[0]
                        == ValidationError::StructDefinedTwice(ErrorContext {
                            struct_name: Some("A".to_string()),
                            step_name: None
                        })
                );
            }
        }
    }

    #[test]
    fn test_extract_double_step() {
        let program_string = r#"struct A(){} struct B(){init {} init {}} init"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let static_data_result = extract_static_program_data(&program);
        match static_data_result {
            Ok(r) => panic!("Expected a StepDefinedTwice failure, got: {:?}.", r),
            Err(errors) => {
                assert!(errors.len() == 1);
                assert!(
                    errors[0]
                        == ValidationError::StepDefinedTwice(ErrorContext {
                            struct_name: Some("B".to_string()),
                            step_name: Some("init".to_string())
                        })
                );
            }
        }
    }

    #[test]
    fn test_extract_double_param() {
        let program_string = r#"struct A(param1: Int, param1: Nat){init{}} init"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let static_data_result = extract_static_program_data(&program);
        match static_data_result {
            Ok(r) => panic!("Expected a ParameterDefinedTwice failure, got: {:?}.", r),
            Err(errors) => {
                assert!(errors.len() == 1);
                assert!(
                    errors[0]
                        == ValidationError::ParameterDefinedTwice(
                            ErrorContext {
                                struct_name: Some("A".to_string()),
                                step_name: None,
                            },
                            "param1".to_string()
                        )
                );
            }
        }
    }

    #[test]
    fn test_check_type_literals() {
        check_type(
            Exp::Lit(NumLit(10)),
            NatType,
            NamedType("something".to_string()),
        );
        check_type(
            Exp::Lit(NumLit(-10)),
            IntType,
            NamedType("something".to_string()),
        );
    }

    #[test]
    #[should_panic(expected = "-10 is not a natural number.")]
    fn test_check_type_literals_illegal_1() {
        check_type(
            Exp::Lit(NumLit(-10)),
            NatType,
            NamedType("something".to_string()),
        );
    }

    #[test]
    #[should_panic(expected = "4294967296 does not fit in unsigned 32 bits.")]
    fn test_check_type_literals_illegal_2() {
        check_type(
            Exp::Lit(NumLit((u32::MAX as i64) + 1)),
            NatType,
            NamedType("something".to_string()),
        );
    }

    #[test]
    #[should_panic(expected = "-2147483649 does not fit in signed 32 bits.")]
    fn test_check_type_literals_illegal_3() {
        check_type(
            Exp::Lit(NumLit((i32::MIN as i64) - 1)),
            IntType,
            NamedType("something".to_string()),
        );
    }
}
