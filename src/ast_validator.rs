use std::collections::HashSet;
use crate::ast::*;
use codespan_reporting::diagnostic::Diagnostic;
use codespan_reporting::diagnostic::Label;
use core::ops::Range;

#[derive(Debug, PartialEq, Eq)]
pub struct ValidationError {
    error_type: ValidationErrorType,
    context: ErrorContext,
    loc: Loc,
}

#[derive(Debug, PartialEq, Eq)]
enum ValidationErrorType {
    StructDefinedTwice(Loc),                        // Loc of earlier decl
    StepDefinedTwice(Loc),                          // Loc of earlier decl
    ParameterDefinedTwice(String, Loc),             // parameter name, Loc of earlier decl
    VariableAlreadyDeclared(String, Loc),           // var name, Loc of earlier decl
    UndefinedType(String),                          // attempted type name
    UndefinedField(String, String),                 // parent name, field name
    UndefinedStep,                                  // Step and struct name already in error_context
    InvalidNumberOfArguments(usize, usize),         // Expected number, supplied number
    TypeMismatch(Type, Type),                       // Expected type, gotten type
    InvalidTypesForOperator(Type, BinOpcode, Type), // lhs type, operator, rhs type
    NoNullLiteralForType(Option<Type>),             // Type of attempted null literal
    ReservedKeyword(String),                        // Name of reserved keyword.
}

impl ValidationError {
    pub fn to_diagnostic(&self) -> Diagnostic<()> {
        let mut labels: Vec<Label<()>> = Vec::new();

        labels.push(self.primary());

        if let Some(secondary) = self.secondary() {
            labels.push(secondary);
        }

        return Diagnostic::error()
            .with_code(self.code())
            .with_message(self.message())
            .with_labels(labels);
    }

    fn code(&self) -> &str {
        use ValidationErrorType::*;
        match self.error_type {
            StructDefinedTwice(..) => "E001",
            StepDefinedTwice(..) => "E002",
            ParameterDefinedTwice(..) => "E003",
            VariableAlreadyDeclared(..) => "E004",
            UndefinedType(..) => "E005",
            UndefinedField(..) => "E006",
            UndefinedStep => "E007",
            InvalidNumberOfArguments(..) => "E008",
            TypeMismatch(..) => "E009",
            InvalidTypesForOperator(..) => "E010",
            NoNullLiteralForType(..) => "E011",
            ReservedKeyword(_) => "E012",
        }
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
        Label::primary((), self.loc()).with_message(self.label())
    }

    fn loc(&self) -> Range<usize> {
        self.loc.0..self.loc.1
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
            TypeMismatch(t1, t2) => format!("Expected type {}, but got {}", t1, t2),
            InvalidTypesForOperator(l, o, r) => format!(
                "The operator {} can not be applied to a LHS of type {} and a RHS of type {}",
                o, l, r
            ),
            NoNullLiteralForType(None) => {
                "The null literal is not defined in this context.".to_string()
            }
            NoNullLiteralForType(Some(t)) => {
                format!("The null literal is not defined for type {}.", t)
            }
            ReservedKeyword(kw) => format!("The token '{}' is a reserved keyword.", kw)
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
            ReservedKeyword(_) => "Used a reserved keyword.",
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

        ErrorContext {
            struct_name,
            step_name,
        }
    }
}

fn is_reserved<'ast>(name : &String) -> bool {
    let reserved_keywords = HashSet::from([
        "fprintf",
        "printf",
    ].map(|e| e.to_string()));

    return reserved_keywords.contains(name);
}
    
fn check_no_reserved_keywords_structs<'ast>(
    structs: &'ast Vec<ADLStruct>,    
    context: &mut BlockEvaluationContext<'ast>
){
    for s in structs {
        if is_reserved(&s.name) {
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::ReservedKeyword(s.name.clone()),
                context: ErrorContext::from_block_context(context),
                loc: s.loc,
            });
        }
    }
}


fn check_no_reserved_keywords_parameters<'ast>(
    params: &'ast [(String, Type, Loc)],
    context: &mut BlockEvaluationContext<'ast>,
){
    for (n, _, l) in params {
        if is_reserved(n) {
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::ReservedKeyword(n.clone()),
                context: ErrorContext::from_block_context(context),
                loc: *l,
            });
        }
    }
}

fn check_no_reserved_keywords_steps<'ast>(
    steps: &'ast Vec<Step>,
    context: &mut BlockEvaluationContext<'ast>,
){
    for s in steps {
        if is_reserved(&s.name) {
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::ReservedKeyword(s.name.clone()),
                context: ErrorContext::from_block_context(context),
                loc: s.loc,
            });
        }
    }
}


#[derive(Debug)]
struct BlockEvaluationContext<'ast> {
    current_struct_name: Option<&'ast String>,
    current_step_name: Option<&'ast String>,
    structs: &'ast Vec<ADLStruct>,
    vars: Vec<Vec<(String, Type, Loc)>>,
    errors: Vec<ValidationError>,
}

impl<'eval, 'ast> BlockEvaluationContext<'ast> {
    fn add_to_var_scope(&'eval mut self, s: &'ast str, t: &'ast Type, l: &'ast Loc) {
        if let Some(v) = self.vars.last_mut() {
            v.push((s.to_owned(), t.clone(), *l));
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
        if self.vars.pop().is_none() {
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

    // All Structs must have a unique name and not reserved keywords
    check_uniqueness_of_structs(&ast.structs, &mut context);
    check_no_reserved_keywords_structs(&ast.structs, &mut context);

    // Test each struct
    for s in &ast.structs {
        context.current_struct_name = Some(&s.name);

        // Test if all parameters are unique and not reserved keywords
        check_uniqueness_of_parameters(&s.parameters, &mut context);
        check_no_reserved_keywords_parameters(&s.parameters, &mut context);

        // Test if all steps are unique and not reserved keywords
        check_uniqueness_of_steps(&s.steps, &mut context);
        check_no_reserved_keywords_steps(&s.steps, &mut context);

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

    context.errors
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
                if step_name != "print" && !s.steps.iter().any(|stp| stp.name == *step_name) {
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
    params: &'ast [(String, Type, Loc)],
    context: &mut BlockEvaluationContext<'ast>,
) {
    for (idx, (pname, _, ploc)) in params.iter().enumerate() {
        if let Some((n, _, l)) = params[0..idx].iter().find(|p| p.0 == *pname) {
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::ParameterDefinedTwice(n.clone(), *l),
                context: ErrorContext::from_block_context(context),
                loc: *ploc,
            });
        }
    }
}

fn check_uniqueness_of_structs<'ast>(
    structs: &'ast Vec<ADLStruct>,
    context: &mut BlockEvaluationContext<'ast>,
) {
    for (idx, cur_struct) in structs.iter().enumerate() {
        context.current_struct_name = Some(&cur_struct.name);
        if let Some(s) = structs[0..idx].iter().find(|s1| s1.name == cur_struct.name) {
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::StructDefinedTwice(s.loc),
                context: ErrorContext::from_block_context(context),
                loc: cur_struct.loc,
            });
        }
    }
}

fn check_uniqueness_of_steps<'ast>(
    steps: &'ast Vec<Step>,
    context: &mut BlockEvaluationContext<'ast>,
) {
    for (idx, step) in steps.iter().enumerate() {
        context.current_step_name = Some(&step.name);
        if let Some(s) = steps[0..idx].iter().find(|s1| s1.name == step.name) {
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::StepDefinedTwice(s.loc),
                context: ErrorContext::from_block_context(context),
                loc: step.loc,
            });
        }
    }
}

fn check_statement_block<'ast>(
    block: &'ast Vec<Stat>,
    context: &mut BlockEvaluationContext<'ast>,
) {
    for stmt in block {
        use crate::ast::Stat::*;

        match stmt {
            Declaration(decl_type, id, exp, loc) => {
                check_declaration(decl_type, id, exp, loc, context);
            }
            Assignment(parts, exp, loc) => {
                check_assignment(parts, exp, loc, context);
            }
            IfThen(cond, statements1, statements2, cond_loc) => {
                check_ifthen(cond, statements1, statements2, cond_loc, context);
            }
        }
    }
}

fn check_declaration<'ast>(
    decl_type : &'ast Type,
    id : &'ast String,
    exp : &'ast Box<Exp>,
    loc : &'ast (usize, usize),
    context: &mut BlockEvaluationContext<'ast>
    ){

    // Throws an error if decl_type is undefined
    if type_is_defined(decl_type, context, *loc) {
        // Make sure id is not used before
        if let Some((_, l)) = get_type_from_context(id, context) {
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::VariableAlreadyDeclared(id.clone(), l),
                context: ErrorContext::from_block_context(context),
                loc: *loc,
            });
        } else {
            // check type of exp
            if let Some(exp_type) =
                get_expr_type(exp, context)
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
                    if is_reserved(id) {
                        context.errors.push(ValidationError {
                            error_type: ValidationErrorType::ReservedKeyword(id.clone()),
                            context: ErrorContext::from_block_context(context),
                            loc: *loc,
                        });
                    } else {
                        // Everything is okay
                        context.add_to_var_scope(id, decl_type, loc);
                    }
                }
            } // else: type of expression could not be deduced
        }
    }
}

fn check_assignment<'ast>(
    parts : &'ast Box<Exp>,
    exp : &'ast Box<Exp>,
    loc : &'ast Loc,
    context: &mut BlockEvaluationContext<'ast>
    ){
    // Can the type of LHS be determined?
    if let Some((var_type, _)) = get_var_type(parts, context, loc) {
        // Can the type of RHS be determined?
        if let Some(exp_type) = get_expr_type(exp, context) {
            // Can RHS be converted to LHS?
            if !exp_type.can_be_coerced_to_type(&var_type) {
                context.errors.push(ValidationError {
                    error_type: ValidationErrorType::TypeMismatch(var_type.clone(), exp_type),
                    context: ErrorContext::from_block_context(context),
                    loc: *loc,
                });
            }
        } // else: type of expression could not be deduced
    }
}

fn check_ifthen<'ast>(
    cond : &'ast Box<Exp>,
    statements_true : &'ast Vec<Stat>,
    statements_false : &'ast Vec<Stat>,
    cond_loc : &'ast Loc,
    context: &mut BlockEvaluationContext<'ast>
    ){
    if let Some(cond_type) = get_expr_type(cond, context) {
        // Booleans have no Null value
        if cond_type != Type::Bool {
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::TypeMismatch(
                    Type::Bool,
                    cond_type,
                ),
                context: ErrorContext::from_block_context(context),
                loc: *cond_loc,
            });
        }
    }

    // Regardless of if the guard is of boolean type, we check the statements
    context.push_var_scope();
    check_statement_block(statements_true, context);
    context.pop_var_scope();

    context.push_var_scope();
    check_statement_block(statements_false, context);
    context.pop_var_scope();
}


/// Returns the type of a `Var`: part1.part2.part3, and the location of the original definition
fn get_var_type<'ast>(
    parts_exp: &'ast Exp,
    context: &mut BlockEvaluationContext<'ast>,
    loc: &'ast Loc,
) -> Option<(Type, Loc)> {
    
    let parts = parts_exp.get_parts();

    // Get type of first part by looking in the current context 
    let mut found_type: Option<(Type, Loc)>;
    found_type = get_type_from_context(&parts[0], context);
    
    let mut idx = 1; // start at 1 as first part is already done
    while idx < parts.len() {
        let id = &parts[idx];
        if let Some((Type::Named(s), _)) = found_type {
            found_type = get_type_from_scope(
                id,
                &context
                .structs
                .iter()
                .find(|st| st.name == *s)
                .unwrap()
                .parameters
            );
        } else {
            break
        }
        idx += 1;
    }

    if let None = found_type {
        if idx == 1 { // i.e. first part was None type hence idx not incremented
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::UndefinedType(parts[0].clone()),
                context: ErrorContext::from_block_context(context),
                loc: *loc,
            });
        } else {
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::UndefinedField(
                    parts[idx-2].clone(),
                    parts[idx-1].clone(),
                ),
                context: ErrorContext::from_block_context(context),
                loc: *loc,
            });
        }
    }


    return found_type
}

fn type_is_defined<'ast>(t: &Type, context: &mut BlockEvaluationContext<'ast>, loc: Loc) -> bool {
    if let Type::Named(s) = t {
        if !context.structs.iter().any(|st| st.name == *s) {
            context.errors.push(ValidationError {
                error_type: ValidationErrorType::UndefinedType(s.clone()),
                context: ErrorContext::from_block_context(context),
                loc,
            });
            return false;
        }
    }
    true
}

/// returned Loc is the location where the variable or struct was declared
fn get_type_from_context<'ast>(
    id: &'ast String,
    context: &mut BlockEvaluationContext<'ast>,
) -> Option<(Type, Loc)> {
    for scope in &context.vars {
        if let Some((t, l)) = get_type_from_scope(id, scope) {
            return Some((t, l));
        }
    }
    None
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
    None
}

fn get_expr_type<'ast>(
    expr: &'ast Exp,
    context: &mut BlockEvaluationContext<'ast>,
) -> Option<Type> {
    use crate::ast::Exp::*;
    use crate::ast::Literal::*;
    use crate::ast::Type::*;

    match expr {
        BinOp(l, code, r, loc) => {
            let l_type: Option<Type> = get_expr_type(l, context);
            let r_type: Option<Type> = get_expr_type(r, context);

            if l_type == None || r_type == None {
                return None;
            }
            let bin_type =
                get_binop_expr_type(l_type.as_ref().unwrap(), code, r_type.as_ref().unwrap());
            if bin_type == None {
                context.errors.push(ValidationError {
                    error_type: ValidationErrorType::InvalidTypesForOperator(
                        l_type.unwrap(),
                        code.clone(),
                        r_type.unwrap(),
                    ),
                    context: ErrorContext::from_block_context(context),
                    loc: *loc,
                });
            }
            bin_type
        }
        UnOp(code, e, loc) => match code {
            UnOpcode::Negation => {
                let e_type = get_expr_type(e, context);
                if let Some(ref t) = e_type {
                    if t != &Bool {
                        context.errors.push(ValidationError {
                            error_type: ValidationErrorType::TypeMismatch(Bool, t.clone()),
                            context: ErrorContext::from_block_context(context),
                            loc: *loc,
                        });
                        return None;
                    }
                }
                e_type
            }
        },
        Constructor(id, exps, loc) => {
            // Check if `id` is a proper type
            let cons_type = Named(id.clone());
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
                    if let Some(exp_type) = get_expr_type(e, context) {
                        if !exp_type.can_be_coerced_to_type(p_type) {
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
                Some(cons_type)
            } else {
                None
            }
        }
        Var(_, loc) => get_var_type(expr, context, loc).map(|(t, _)| t),
        Lit(lit, _) => {
            match lit {
                NatLit(_) => Some(Nat),
                IntLit(_) => Some(Int),
                BoolLit(_) => Some(Bool),
                StringLit(_) => Some(String),
                NullLit => Some(Null),
                ThisLit => Some(Named(context.current_struct_name.unwrap().clone())),
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

    let is_ordering = |o: &'ast BinOpcode| match o {
        LessThanEquals | GreaterThanEquals | LessThan | GreaterThan => true,
        _ => false,
    };

    let is_equality = |o: &'ast BinOpcode| match o {
        Equals | NotEquals => true,
        _ => false,
    };

    let is_boolean_logic = |o: &'ast BinOpcode| match o {
        And | Or => true,
        _ => false,
    };

    match (l, op, r) {
        (Nat, _, Nat) if is_arithmetic(op) => Some(Nat),
        (Int, _, Int) if is_arithmetic(op) => Some(Int),
        (Nat | Int, _, Nat | Int) if is_arithmetic(op) => Some(Int),
        (Nat | Int, _, Nat | Int) if is_ordering(op) => Some(Bool),
        (Nat | Int, _, Nat | Int) if is_equality(op) => Some(Bool),

        (String, _, String) if is_equality(op) => Some(Bool),
        (Bool, _, Bool) if is_equality(op) => Some(Bool),
        (Bool, _, Bool) if is_boolean_logic(op) => Some(Bool),
        (Named(..)|Null, _, Named(..)|Null) if is_equality(op) || is_ordering(op) => Some(Bool),
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


    #[test]
    fn test_validate_var_type_undef_first_part() {
        let program_string = r#"struct A(a : Int, b : A){init{b.a := 1; c.a := 2;}} init"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let validation_errors = validate_ast(&program);
        assert!(validation_errors.len() == 1);
        assert_eq!(
            validation_errors[0],
            ValidationError {
                error_type: UndefinedType("c".to_string()),
                context: ErrorContext {
                    struct_name: Some("A".to_string()),
                    step_name: Some("init".to_string())
                },
                loc: (40, 49)
            }
        );
    }

    #[test]
    fn test_validate_var_type_undef_second_part() {
        let program_string = r#"struct A(a : Int, b : A){init{b.a := 1; b.c := 2;}} init"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let validation_errors = validate_ast(&program);
        assert!(validation_errors.len() == 1);
        assert_eq!(
            validation_errors[0],
            ValidationError {
                error_type: UndefinedField("b".to_string(), "c".to_string()),
                context: ErrorContext {
                    struct_name: Some("A".to_string()),
                    step_name: Some("init".to_string())
                },
                loc: (40, 49)
            }
        );
    }

    #[test]
    fn test_validate_var_type_undef_third_part() {
        let program_string = r#"struct A(a : Int, b : A){init{b.a := 1; b.b.c := 2;}} init"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let validation_errors = validate_ast(&program);
        assert!(validation_errors.len() == 1);
        assert_eq!(
            validation_errors[0],
            ValidationError {
                error_type: UndefinedField("b".to_string(), "c".to_string()),
                context: ErrorContext {
                    struct_name: Some("A".to_string()),
                    step_name: Some("init".to_string())
                },
                loc: (40, 51)
            }
        );
    }

    #[test]
    fn test_validate_reserved_structs() {
        let program_string = r#"struct printf(a : Int, b : A){init{}} init"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let validation_errors = validate_ast(&program);
        assert!(validation_errors.len() == 1);
        assert_eq!(
            validation_errors[0],
            ValidationError {
                error_type: ReservedKeyword("printf".to_string()),
                context: ErrorContext {
                    struct_name: Some("printf".to_string()),
                    step_name: None
                },
                loc: (0, 37)
            }
        );
    }

    #[test]
    fn test_validate_reserved_params() {
        let program_string = r#"struct A(fprintf : Int){init{}} init"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let validation_errors = validate_ast(&program);
        assert!(validation_errors.len() == 1);
        assert_eq!(
            validation_errors[0],
            ValidationError {
                error_type: ReservedKeyword("fprintf".to_string()),
                context: ErrorContext {
                    struct_name: Some("A".to_string()),
                    step_name: None
                },
                loc: (9, 22)
            }
        );
    }

    #[test]
    fn test_validate_reserved_steps() {
        let program_string = r#"struct A(b : Int){printf{}} printf"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let validation_errors = validate_ast(&program);
        assert!(validation_errors.len() == 1);
        assert_eq!(
            validation_errors[0],
            ValidationError {
                error_type: ReservedKeyword("printf".to_string()),
                context: ErrorContext {
                    struct_name: Some("A".to_string()),
                    step_name: Some("printf".to_string())
                },
                loc: (18, 26)
            }
        );
    }

    #[test]
    fn test_validate_reserved_vars() {
        let program_string = r#"struct A(b : Int){init{ Int printf := 1;}} init"#;
        let program = ProgramParser::new()
            .parse(program_string)
            .expect("ParseError.");
        let validation_errors = validate_ast(&program);
        assert!(validation_errors.len() == 1);
        assert_eq!(
            validation_errors[0],
            ValidationError {
                error_type: ReservedKeyword("printf".to_string()),
                context: ErrorContext {
                    struct_name: Some("A".to_string()),
                    step_name: Some("init".to_string())
                },
                loc: (24, 40)
            }
        );
    }
}
