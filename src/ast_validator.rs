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
    StructDefinedTwice(Loc),                        //Loc of earlier decl
    StepDefinedTwice(Loc),                          //Loc of earlier decl
    ParameterDefinedTwice(String, Loc),             //parameter name, Loc of earlier decl
    VariableAlreadyDeclared(String, Loc),           //var name, Loc of earlier decl
    UndefinedType(String),                          //attempted type name
    UndefinedField(String, String),                 //parent name, field name
    UndefinedStep,                                  //Step and struct name already in error_context
    InvalidNumberOfArguments(usize, usize),         //Expected number, supplied number
    TypeMismatch(Type, Type),                       //Expected type, gotten type
    InvalidTypesForOperator(Type, BinOpcode, Type), //lhs type, operator, rhs type
    NoNullLiteralForType(Option<Type>),             // Type of attempted null literal
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
            StructDefinedTwice(l)
            | StepDefinedTwice(l)
            | ParameterDefinedTwice(_, l)
            | VariableAlreadyDeclared(_, l) => {
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
            InvalidNumberOfArguments(e, r) => format!(
                "The struct {} takes {} arguments, while {} were given.",
                self.context.struct_name.as_ref().unwrap(),
                e,
                r
            ),
            TypeMismatch(t1, t2) => format!("Expected type '{}', but got '{}'", t1, t2),
            InvalidTypesForOperator(l, o, r) => format!(
                "The operator {:?} can not be applied to a LHS of type {} and a RHS of type {}",
                o, l, r
            ),
            NoNullLiteralForType(None) => {
                format!("The null literal is not defined in this context.")
            }
            NoNullLiteralForType(Some(t)) => {
                format!("The null literal is not defined for type {}.", t)
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
            InvalidNumberOfArguments(..) => "Invalid number of arguments supplied.",
            TypeMismatch(..) => "An invalid type has been given.",
            InvalidTypesForOperator(..) => "Operator can not be applied to given types.",
            NoNullLiteralForType(..) => "The null literal is not defined in this context.",
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
            Declaration(decl_type, id, exp, loc) => {
                if type_is_defined(decl_type, context, loc.clone()) {
                    // Make sure id is not used before
                    if let Some((_, l)) = get_type_from_context(&id, context) {
                        context.errors.push(ValidationError {
                            error_type: ValidationErrorType::VariableAlreadyDeclared(id.clone(), l),
                            context: ErrorContext::from_block_context(context),
                            loc: *loc,
                        });
                    } else {
                        // t is defined and id has not been used before
                        if let Some(exp_type) =
                            get_expr_type(exp, &Some(decl_type.clone()), context)
                        {
                            if !exp_type.can_be_coerced_to_type(decl_type) {
                                context.errors.push(ValidationError {
                                    error_type: ValidationErrorType::TypeMismatch(
                                        decl_type.clone(),
                                        exp_type,
                                    ),
                                    context: ErrorContext::from_block_context(context),
                                    loc: *loc,
                                });
                            } else {
                                context.add_to_var_scope(id, decl_type, loc);
                            }
                        } // else: type of expression could not be deduced
                    }
                }
            }
            Assignment(parts, exp, loc) => {
                if let Some((var_type, _)) = get_var_type(parts, context, loc) {
                    if let Some(exp_type) = get_expr_type(exp, &Some(var_type.clone()), context) {
                        if !exp_type.can_be_coerced_to_type(&var_type) {
                            context.errors.push(ValidationError {
                                error_type: ValidationErrorType::TypeMismatch(var_type, exp_type),
                                context: ErrorContext::from_block_context(context),
                                loc: *loc,
                            });
                        }
                    } // else: type of expression could not be deduced
                }
            }
            IfThen(cond, statements, cond_loc) => {
                if let Some(cond_type) = get_expr_type(cond, &Some(Type::BoolType), context) {
                    // Booleans have no Null value
                    if cond_type != Type::BoolType {
                        context.errors.push(ValidationError {
                            error_type: ValidationErrorType::TypeMismatch(
                                Type::BoolType,
                                cond_type,
                            ),
                            context: ErrorContext::from_block_context(context),
                            loc: *cond_loc,
                        });
                    }
                }

                // Regardless of if the guard is of boolean type, we check the statements
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

    let mut found_type: Option<(Type, Loc)>;
    if parts[0] == "this" {
        found_type = Some((
            Type::NamedType(context.current_struct_name.unwrap().clone()),
            (0, 0),
        ));
    } else {
        found_type = get_type_from_context(&parts[0], context);
    }

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

fn type_is_defined<'ast>(t: &Type, context: &mut BlockEvaluationContext<'ast>, loc: Loc) -> bool {
    if let Type::NamedType(s) = t {
        if !context.structs.iter().any(|st| st.name == *s) {
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::UndefinedType(s.clone()),
                context: ErrorContext::from_block_context(context),
                loc: loc,
            });
            return false;
        }
    }
    return true;
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

fn get_expr_type<'ast>(
    expr: &'ast Exp,
    preferred_type: &Option<Type>,
    context: &mut BlockEvaluationContext<'ast>,
) -> Option<Type> {
    use crate::ast::Exp::*;
    use crate::ast::Literal::*;
    use crate::ast::Type::*;

    match expr {
        BinOp(l, code, r, loc) => {
            let l_type: Option<Type> = get_expr_type(l, preferred_type, context);
            let r_type: Option<Type> = get_expr_type(r, preferred_type, context);

            if l_type == None || r_type == None {
                return None;
            }
            let bin_type =
                get_binop_expr_type(l_type.as_ref().unwrap(), code, r_type.as_ref().unwrap());
            if bin_type == None {
                context.errors.push(ValidationError {
                    error_type: ValidationErrorType::InvalidTypesForOperator(
                        l_type.unwrap().clone(),
                        code.clone(),
                        r_type.unwrap().clone(),
                    ),
                    context: ErrorContext::from_block_context(context),
                    loc: *loc,
                });
            }
            return bin_type;
        }
        UnOp(code, e, loc) => match code {
            UnOpcode::Negation => {
                let e_type = get_expr_type(e, &Some(BoolType), context);
                if let Some(ref t) = e_type {
                    if t != &BoolType {
                        context.errors.push(ValidationError {
                            error_type: ValidationErrorType::TypeMismatch(BoolType, t.clone()),
                            context: ErrorContext::from_block_context(context),
                            loc: *loc,
                        });
                        return None;
                    }
                }
                return e_type;
            }
        },
        Constructor(id, exps, loc) => {
            // Check if `id` is a proper type
            let cons_type = NamedType(id.clone());
            if type_is_defined(&cons_type, context, *loc) {
                // check types of exps with parameters
                let params = &context
                    .structs
                    .iter()
                    .find(|strct| strct.name == *id)
                    .unwrap()
                    .parameters;
                if params.len() != exps.len() {
                    context.errors.push(ValidationError {
                        error_type: ValidationErrorType::InvalidNumberOfArguments(
                            params.len(),
                            exps.len(),
                        ),
                        context: ErrorContext {
                            struct_name: Some(id.clone()),
                            step_name: None,
                        },
                        loc: *loc,
                    });
                }

                for ((_, p_type, _), e) in params.iter().zip(exps) {
                    if let Some(exp_type) = get_expr_type(e, &Some(p_type.clone()), context) {
                        if exp_type != *p_type {
                            context.errors.push(ValidationError {
                                error_type: ValidationErrorType::TypeMismatch(
                                    p_type.clone(),
                                    exp_type,
                                ),
                                context: ErrorContext {
                                    struct_name: Some(id.clone()),
                                    step_name: None,
                                },
                                loc: *loc,
                            });
                        }
                    }
                }
                return Some(cons_type);
            } else {
                return None;
            }
        }
        Var(parts, loc) => get_var_type(parts, context, loc).map(|(t, _)| t),
        Lit(lit, loc) => {
            match lit {
                NatLit(_) => Some(NatType),
                IntLit(_) => Some(IntType),
                BoolLit(_) => Some(BoolType),
                StringLit(_) => Some(StringType),
                NullLit => {
                    // Only named types have null literals
                    match preferred_type {
                        Some(NamedType(s)) => Some(NamedType(s.clone())),
                        t => {
                            context.errors.push(ValidationError {
                                error_type: ValidationErrorType::NoNullLiteralForType(
                                    t.as_ref().map(|a| a.clone()),
                                ),
                                context: ErrorContext::from_block_context(context),
                                loc: *loc,
                            });
                            None
                        }
                    }
                }
                ThisLit => Some(NamedType(context.current_struct_name.unwrap().clone())),
            }
        }
    }
}

fn get_binop_expr_type<'ast>(l: &Type, op: &'ast BinOpcode, r: &Type) -> Option<Type> {
    use crate::ast::BinOpcode::*;
    use crate::ast::Type::*;

    let is_arithmetic = |o: &'ast BinOpcode| match o {
        Plus | Minus | Mult | Div | Mod => true,
        _ => false,
    };
    let is_comparison = |o: &'ast BinOpcode| match o {
        Equals | NotEquals | LessThanEquals | GreaterThanEquals | LessThan | GreaterThan => true,
        _ => false,
    };

    match (l, op, r) {
        (NatType, _, NatType) if is_arithmetic(op) => Some(NatType),
        (IntType, _, IntType) if is_arithmetic(op) => Some(IntType),
        (NatType | IntType, _, NatType | IntType) if is_arithmetic(op) => Some(IntType),
        (NatType | IntType, _, NatType | IntType) if is_comparison(op) => Some(BoolType),
        (StringType, _, StringType) if is_comparison(op) => Some(BoolType),
        (BoolType, _, BoolType) if is_comparison(op) => Some(BoolType),
        (..) => None,
    }
}

#[cfg(test)]
mod tests {
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
