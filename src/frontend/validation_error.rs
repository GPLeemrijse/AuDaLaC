use core::ops::Range;
use codespan_reporting::diagnostic::Label;
use codespan_reporting::diagnostic::Diagnostic;
use crate::frontend::ast::BinOpcode;
use crate::frontend::ast::Type;
use crate::frontend::ast::Loc;

#[derive(Debug, PartialEq, Eq)]
pub struct ValidationError {
    pub error_type: ValidationErrorType,
    pub context: ErrorContext,
    pub loc: Loc,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ValidationErrorType {
    StructDefinedTwice(Loc),                        // Loc of earlier decl
    StepDefinedTwice(Loc),                          // Loc of earlier decl
    ParameterDefinedTwice(String, Loc),             // parameter name, Loc of earlier decl
    VariableAlreadyDeclared(String, Loc),           // var name, Loc of earlier decl
    UndefinedType(String),                          // attempted type name
    UndefinedVariable(String),                           // attempted var name
    UndefinedParameter(String, String),                 // parent name, field name
    UndefinedStep,                                  // Step and struct name already in error_context
    InvalidNumberOfArguments(usize, usize),         // Expected number, supplied number
    TypeMismatch(Type, Type),                       // Expected type, gotten type
    InvalidTypesForOperator(Type, BinOpcode, Type), // lhs type, operator, rhs type
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
            UndefinedParameter(..) => "E006",
            UndefinedStep => "E007",
            UndefinedVariable(..) => "E008",
            InvalidNumberOfArguments(..) => "E009",
            TypeMismatch(..) => "E010",
            InvalidTypesForOperator(..) => "E011",
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
            UndefinedVariable(t) => format!("Undefined variable {}.", t),
            UndefinedParameter(f, t) => format!("Undefined parameter {} of {}.", t, f),
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
            ReservedKeyword(kw) => format!("The token '{}' is a reserved keyword.", kw),
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
            UndefinedVariable(..) => "Undefined variable.",
            UndefinedParameter(..) => "Undefined parameter.",
            UndefinedStep => "Undefined step",
            InvalidNumberOfArguments(..) => "Invalid number of arguments supplied.",
            TypeMismatch(..) => "An invalid type has been given.",
            InvalidTypesForOperator(..) => "Operator can not be applied to given types.",
            ReservedKeyword(_) => "Used a reserved keyword.",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ErrorContext {
    pub struct_name: Option<String>,
    pub step_name: Option<String>,
}