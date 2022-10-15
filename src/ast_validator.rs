use crate::ast::*;
use codespan_reporting::diagnostic::Diagnostic;
use codespan_reporting::diagnostic::Label;
use core::ops::Range;

// All steps in schedule have been defined
// Numeric operators always between: Nat Nat, Nat Int, Int Int
// Unique step names in Structs
// No declared variables in steps with the same name as parameters

#[derive(Debug, PartialEq, Eq)]
pub struct ValidationError {
    error_type: ValidationErrorType,
    context: ErrorContext,
    loc: Loc,
}

#[derive(Debug, PartialEq, Eq)]
enum ValidationErrorType {
    StructDefinedTwice(Loc),              //Loc of earlier decl
    StepDefinedTwice(Loc),                //Loc of earlier decl
    ParameterDefinedTwice(String, Loc),   //parameter name, Loc of earlier decl
    VariableAlreadyDeclared(String, Loc), //var name, Loc of earlier decl
    UndefinedType(String),                //attempted type name
    UndefinedField(String, String),       //parent name, field name
    UndefinedStep,                        //Step and struct name already in error_context
}

impl ValidationError {
    pub fn to_diagnostic(&self) -> Diagnostic<()> {
        let mut labels: Vec<Label<()>> = Vec::new();

        labels.push(self.primary());

        if let Some(secondary) = self.secondary() {
            labels.push(secondary);
        }

        return Diagnostic::error()
            .with_message(self.message())
            .with_labels(labels);
    }

    fn secondary(&self) -> Option<Label<()>> {
        use ValidationErrorType::*;
        match &self.error_type {
            StructDefinedTwice(l) => {
                Some(Label::secondary((), l.0..l.1).with_message("Previous declaration here."))
            }
            StepDefinedTwice(l) => {
                Some(Label::secondary((), l.0..l.1).with_message("Previous declaration here."))
            }
            ParameterDefinedTwice(_, l) => {
                Some(Label::secondary((), l.0..l.1).with_message("Previous declaration here."))
            }
            VariableAlreadyDeclared(_, l) => {
                Some(Label::secondary((), l.0..l.1).with_message("Previous declaration here."))
            }
            _ => None,
        }
    }

    fn primary(&self) -> Label<()> {
        return Label::primary((), self.loc()).with_message(self.label());
    }

    fn loc(&self) -> Range<usize> {
        return self.loc.0..self.loc.1;
    }

    fn label(&self) -> String {
        use ValidationErrorType::*;
        match &self.error_type {
            StructDefinedTwice(..) => format!(
                "Struct {} defined twice.",
                self.context.struct_name.as_ref().unwrap()
            ),
            StepDefinedTwice(..) => format!(
                "Step {} defined twice.",
                self.context.step_name.as_ref().unwrap()
            ),
            ParameterDefinedTwice(p, _) => format!("Parameter {} defined twice.", p),
            VariableAlreadyDeclared(v, _) => format!("Variable {} defined twice.", v),
            UndefinedType(t) => format!("Undefined type {}.", t),
            UndefinedField(f, t) => format!("Undefined field {} of {}.", f, t),
            UndefinedStep => {
                if let Some(n) = &self.context.struct_name {
                    format!(
                        "The struct {} does not have a step {} defined.",
                        n,
                        self.context.step_name.as_ref().unwrap()
                    )
                } else {
                    format!(
                        "The step {} is not defined for any struct.",
                        self.context.step_name.as_ref().unwrap()
                    )
                }
            }
        }
    }

    fn message(&self) -> &str {
        use ValidationErrorType::*;
        match self.error_type {
            StructDefinedTwice(..) => "Struct defined twice.",
            StepDefinedTwice(..) => "Step defined twice.",
            ParameterDefinedTwice(..) => "Parameter defined twice.",
            VariableAlreadyDeclared(..) => "Variable defined twice.",
            UndefinedType(..) => "Undefined type.",
            UndefinedField(..) => "Undefined field.",
            UndefinedStep => "Undefined step",
        }
    }
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

#[derive(Debug)]
struct BlockEvaluationContext<'ast> {
    current_struct_name: Option<&'ast String>,
    current_step_name: Option<&'ast String>,
    structs: &'ast Vec<Box<LoplStruct>>,
    vars: Vec<Vec<(String, Type, Loc)>>,
    errors: Vec<ValidationError>,
}

impl<'eval, 'ast> BlockEvaluationContext<'ast> {
    fn add_to_var_scope(&'eval mut self, s: &'ast String, t: &'ast Type, l: &'ast Loc) {
        if let Some(v) = self.vars.last_mut() {
            v.push((s.clone(), t.clone(), l.clone()));
        } else {
            panic!(
                "Error: there was no variable scope at location {:?}. Context: {:#?}",
                l, self
            );
        }
    }

    fn push_var_scope(&mut self) {
        self.vars.push(Vec::new());
    }

    fn pop_var_scope(&mut self) {
        if let None = self.vars.pop() {
            panic!("Popped non-existing variable scope. Context: {:#?}", self);
        }
    }
}

pub fn validate_ast<'ast>(ast: &'ast Program) -> Vec<ValidationError> {
    let mut context = BlockEvaluationContext {
        current_struct_name: None,
        current_step_name: None,
        structs: &ast.structs,
        vars: Vec::new(),
        errors: Vec::new(),
    };

    // All Structs must have a unique name
    check_uniqueness_of_structs(&ast.structs, &mut context);

    // Push all struct types to the context
    context.push_var_scope();
    ast.structs
        .iter()
        .for_each(|strct| context.add_to_var_scope(&strct.name, &Type::NamedType(strct.name), &strct.loc));

    // Test each struct
    for s in &ast.structs {
        context.current_struct_name = Some(&s.name);

        // Test if all parameters are unique
        check_uniqueness_of_parameters(&s.parameters, &mut context);

        // Test if all steps are unique
        check_uniqueness_of_steps(&s.steps, &mut context);

        // Create a fresh scope for the Struct's parameters
        context.push_var_scope();
        s.parameters
            .iter()
            .for_each(|(n, t, l)| context.add_to_var_scope(n, t, l));

        // Test each step
        for step in &s.steps {
            context.current_step_name = Some(&step.name);

            // Each step also gets its own context
            context.push_var_scope();

            check_statement_block(&step.statements, &mut context);

            context.pop_var_scope();
        }
        context.current_step_name = None;

        // The struct's parameters leave the scope
        context.pop_var_scope();
    }
    context.current_struct_name = None;
    
    // The struct types leave the scope
    context.pop_var_scope();

    // Make sure the schedule is well defined
    check_schedule(&ast.schedule, &mut context);

    return context.errors;
}

fn check_schedule<'ast>(schedule: &'ast Schedule, context: &mut BlockEvaluationContext<'ast>) {
    use crate::ast::Schedule::*;
    match schedule {
        StepCall(step_name, loc) => {
            if !context
                .structs
                .iter()
                .any(|strct| strct.steps.iter().any(|stp| stp.name == *step_name))
            {
                context.errors.push(ValidationError {
                    error_type: ValidationErrorType::UndefinedStep,
                    context: ErrorContext {
                        step_name: Some(step_name.clone()),
                        struct_name: None,
                    },
                    loc: *loc,
                });
            }
        }
        TypedStepCall(struct_name, step_name, loc) => {
            let strct = context.structs.iter().find(|s| s.name == *struct_name);
            if let Some(s) = strct {
                if !s.steps.iter().any(|stp| stp.name == *step_name) {
                    context.errors.push(ValidationError {
                        error_type: ValidationErrorType::UndefinedStep,
                        context: ErrorContext {
                            step_name: Some(step_name.clone()),
                            struct_name: Some(struct_name.clone()),
                        },
                        loc: *loc,
                    });
                }
            } else {
                context.errors.push(ValidationError {
                    error_type: ValidationErrorType::UndefinedType(struct_name.clone()),
                    context: ErrorContext {
                        step_name: Some(step_name.clone()),
                        struct_name: Some(struct_name.clone()),
                    },
                    loc: *loc,
                });
            }
        }
        Sequential(s1, s2, _) => {
            check_schedule(s1, context);
            check_schedule(s2, context);
        }
        Fixpoint(s, _) => check_schedule(s, context),
    }
}

fn check_uniqueness_of_parameters<'ast>(
    params: &'ast Vec<(String, Type, Loc)>,
    context: &mut BlockEvaluationContext<'ast>,
) {
    for (idx, (pname, _, ploc)) in params.iter().enumerate() {
        if let Some((n, _, l)) = params[0..idx].iter().find(|p| p.0 == *pname) {
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::ParameterDefinedTwice(n.clone(), *l),
                context: ErrorContext::from_block_context(&context),
                loc: *ploc,
            });
        }
    }
}

fn check_uniqueness_of_structs<'ast>(
    structs: &'ast Vec<Box<LoplStruct>>,
    context: &mut BlockEvaluationContext<'ast>,
) {
    for (idx, cur_struct) in structs.iter().enumerate() {
        context.current_struct_name = Some(&cur_struct.name);
        if let Some(s) = structs[0..idx].iter().find(|s1| s1.name == cur_struct.name) {
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::StructDefinedTwice(s.loc),
                context: ErrorContext::from_block_context(&context),
                loc: cur_struct.loc,
            });
        }
    }
}

fn check_uniqueness_of_steps<'ast>(
    steps: &'ast Vec<Box<Step>>,
    context: &mut BlockEvaluationContext<'ast>,
) {
    for (idx, step) in steps.iter().enumerate() {
        context.current_step_name = Some(&step.name);
        if let Some(s) = steps[0..idx].iter().find(|s1| s1.name == step.name) {
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::StepDefinedTwice(s.loc),
                context: ErrorContext::from_block_context(&context),
                loc: step.loc,
            });
        }
    }
}

fn check_statement_block<'ast>(
    block: &'ast Vec<Box<Stat>>,
    context: &mut BlockEvaluationContext<'ast>,
) {
    for stmt in block {
        use crate::ast::Stat::*;

        match &**stmt {
            Declaration(t, id, exp, loc) => {

                // If a named type is used, make sure it exists
                if let Type::NamedType(s) = t {
                    if let None = get_type_from_context(s, context) {
                        todo!();
                    }
                }
                
                // Make sure id is not used before
                if let Some((_, l)) = get_type_from_context(&id, context) {
                    context.errors.push(ValidationError {
                        error_type: ValidationErrorType::VariableAlreadyDeclared(id.clone(), l),
                        context: ErrorContext::from_block_context(context),
                        loc: *loc,
                    });
                }
                // Typecheck exp with t
                // Make something like validate_exp(exp)
                // Checking defines and var fields etc.
                //todo!();

                context.add_to_var_scope(id, t, loc)
            }
            Assignment(parts, exp, loc) => {
                let field_type = get_var_type(parts, context, loc);

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

/// Returns the type of a `Var`: part1.part2.part3, and the location of the original definition
fn get_var_type<'ast>(
    parts: &'ast Vec<String>,
    context: &mut BlockEvaluationContext<'ast>,
    loc: &'ast Loc,
) -> Option<(Type, Loc)> {
    debug_assert!(
        parts.len() > 0,
        "The length of `parts` should be at least 1."
    );

    let mut found_type: Option<(Type, Loc)> = get_type_from_context(&parts[0], context);
    if parts.len() == 1 {
        return match found_type {
            None => {
                context.errors.push(ValidationError {
                    error_type: ValidationErrorType::UndefinedType(parts[0].clone()),
                    context: ErrorContext::from_block_context(context),
                    loc: *loc,
                });
                None
            }
            Some(t) => Some(t),
        };
    }
    for (idx, id) in parts[1..].iter().enumerate() {
        match found_type {
            Some((Type::NamedType(s), _)) => {
                found_type = get_type_from_scope(
                    id,
                    &context
                        .structs
                        .iter()
                        .find(|st| st.name == *s)
                        .unwrap()
                        .parameters,
                );
            }
            _ => {
                context.errors.push(ValidationError {
                    error_type: ValidationErrorType::UndefinedField(
                        parts[idx - 1].clone(),
                        parts[idx].clone(),
                    ),
                    context: ErrorContext::from_block_context(context),
                    loc: *loc,
                });
            }
        }
    }
    return found_type;
}

/// returned Loc is the location where the variable or struct was declared
fn get_type_from_context<'ast>(
    id: &'ast String,
    context: &mut BlockEvaluationContext<'ast>,
) -> Option<(Type, Loc)> {
    for scope in &context.vars {
        if let Some((t, l)) = get_type_from_scope(id, scope) {
            return Some((t.clone(), l));
        }
    }
    return None;
}

/// returned Loc is the location where the variable or struct was declared 
fn get_type_from_scope<'ast>(
    id: &'ast String,
    scope: &Vec<(String, Type, Loc)>,
) -> Option<(Type, Loc)> {
    for (var_name, var_type, loc) in scope {
        if var_name == id {
            return Some((var_type.clone(), *loc));
        }
    }
    return None;
}

fn get_expr_type<'ast>(expr: &'ast Exp, context: &mut BlockEvaluationContext<'ast>) -> Option<Type> {
    use crate::ast::Exp::*;
    match expr {
        BinOp(l, code, r, loc) => todo!(),
        UnOp(code, e, loc) => todo!(),
        Constructor(id, exps, loc) => todo!(),
        Var(parts, loc) => {
            if let Some((t, loc)) = get_var_type(parts, context, loc) {
                return Some(t);
            } else {
                return None
            }
        },
        Lit(lit, loc) => todo!(),
    }
}


#[cfg(test)]
mod tests {
    use crate::ast::Literal::*;
    use crate::ast::Type::*;
    use crate::ast::*;
    use crate::ast_validator::validate_ast;
    use crate::ast_validator::ErrorContext;
    use crate::ast_validator::ValidationError;
    use crate::ast_validator::ValidationErrorType::*;
    use crate::ProgramParser;

    #[test]
    fn test_validate_double_var_decl() {
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
            ValidationError {
                error_type: VariableAlreadyDeclared("n1".to_string(), (59, 71)),
                context: ErrorContext {
                    struct_name: Some("Node".to_string()),
                    step_name: Some("init".to_string())
                },
                loc: (82, 94)
            }
        );
        assert_eq!(
            errors[1],
            ValidationError {
                error_type: VariableAlreadyDeclared("n3".to_string(), (183, 196)),
                context: ErrorContext {
                    struct_name: Some("Node".to_string()),
                    step_name: Some("step_node".to_string())
                },
                loc: (233, 245)
            }
        );
    }

    #[test]
    fn test_validate_double_struct() {
        let program_string = r#"struct A(){init{}} struct A(){} init"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let validation_errors = validate_ast(&program);
        assert!(validation_errors.len() == 1);
        assert_eq!(
            validation_errors[0],
            ValidationError {
                error_type: StructDefinedTwice((0, 18)),
                context: ErrorContext {
                    struct_name: Some("A".to_string()),
                    step_name: None
                },
                loc: (19, 31)
            }
        );
    }

    #[test]
    fn test_validate_double_step() {
        let program_string = r#"struct A(){} struct B(){init {} init {}} init"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let validation_errors = validate_ast(&program);

        assert!(validation_errors.len() == 1);
        assert_eq!(
            validation_errors[0],
            ValidationError {
                error_type: StepDefinedTwice((24, 31)),
                context: ErrorContext {
                    struct_name: Some("B".to_string()),
                    step_name: Some("init".to_string())
                },
                loc: (32, 39)
            }
        );
    }

    #[test]
    fn test_validate_double_param() {
        let program_string = r#"struct A(param1: Int, param1: Nat){init{}} init"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let validation_errors = validate_ast(&program);

        assert!(validation_errors.len() == 1);
        assert_eq!(
            validation_errors[0],
            ValidationError {
                error_type: ParameterDefinedTwice("param1".to_string(), (9, 20)),
                context: ErrorContext {
                    struct_name: Some("A".to_string()),
                    step_name: None,
                },
                loc: (22, 33)
            }
        );
    }
}
