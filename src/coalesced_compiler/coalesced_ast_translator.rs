use crate::parser::ast::Literal;
use crate::parser::ast::Type;

pub fn as_c_type(t: &Type) -> String {
    use Type::*;
    match t {
        Named(..) => "RefType".to_string(),
        String => "StringType".to_string(),
        Nat => "NatType".to_string(),
        Int => "IntType".to_string(),
        Bool => "BoolType".to_string(),
        Null => "RefType".to_string(),
    }
}

pub fn as_type_enum(t: &Type) -> String {
    use Type::*;
    match t {
        Named(..) => "Ref".to_string(),
        String => "String".to_string(),
        Nat => "Nat".to_string(),
        Int => "Int".to_string(),
        Bool => "Bool".to_string(),
        Null => "Ref".to_string(),
    }
}

pub fn as_printf(t: &Type) -> String {
    use Type::*;
    match t {
        Named(_) => "%u",
        String => "%s",
        Nat => "%u",
        Int => "%d",
        Bool => "%u",
        Null => "%u",
    }
    .to_string()
}

pub fn as_c_literal(l: &Literal) -> String {
    use Literal::*;
    match l {
        NatLit(n) => format!("{}", n),
        IntLit(i) => format!("{}", i),
        BoolLit(b) => format!("{}", if *b { "true" } else { "false" }),
        StringLit(s) => format!("\"{}\"", s),
        NullLit => "0".to_string(), // index based!
        ThisLit => "self".to_string(),
    }
}
