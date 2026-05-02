"""
Q-Lang v0.4 — recursive-descent parser (SPEC §4).

Design
------
* One function per grammar production.
* No lookahead deeper than one token (peek()); predictive parsing.
* Every non-terminal consumes exactly the tokens it owns and nothing
  more; the caller orchestrates separators.
* Unknown or deferred syntax raises ``QLangSyntaxError`` as soon as
  it's recognised, with the offending token's line + col.
* Deferred keywords (``if``, ``for``, ``fn``, ...) raise a message
  naming the feature (SPEC §3.2).

Public API
----------
    from parser_v04 import parse
    program = parse(source)                # raises QLangSyntaxError
    program.decls                          # tuple[Decl, ...]
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ast_nodes_v04 import (  # type: ignore[import-not-found]
    BinaryOp,
    Call,
    ExperimentDef,
    ExpClause,
    GivenClause,
    GivenParam,
    Identifier,
    InvariantClause,
    LetStmt,
    NumberLiteral,
    PipeExpr,
    PrintStmt,
    Program,
    ResultClause,
    SimulateExpr,
    StringLiteral,
    UnaryOp,
    UncertaintyExpr,
    UnitAtom,
    UnitConversion,
    UnitExpr,
)
from errors_v04 import QLangSyntaxError  # type: ignore[import-not-found]
from lexer_v04 import Token, tokenize  # type: ignore[import-not-found]


# Deferred-feature token kinds → human-readable feature names.
_DEFERRED_FEATURE_NAMES: dict[str, str] = {
    "DEFERRED_IF": "if",
    "DEFERRED_ELSE": "else",
    "DEFERRED_ELIF": "elif",
    "DEFERRED_WHILE": "while",
    "DEFERRED_FOR": "for",
    "DEFERRED_FN": "fn / user-defined function",
    "DEFERRED_DEF": "def / user-defined function",
    "DEFERRED_PROVE": "prove",
    "DEFERRED_REASON": "reason",
    "DEFERRED_GENERATE": "generate",
    "DEFERRED_OPTIMIZE": "optimize",
    "DEFERRED_MATCH": "match",
    "DEFERRED_RETURN": "return",
    "DEFERRED_YIELD": "yield",
    "DEFERRED_IMPORT": "import",
    "DEFERRED_CLASS": "class",
    "DEFERRED_LAMBDA": "lambda",
}


class Parser:
    """Recursive-descent parser over a list of tokens."""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # ================================================================
    # Token helpers
    # ================================================================

    def peek(self, offset: int = 0) -> Token:
        i = self.pos + offset
        if i >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[i]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.kind != "EOF":
            self.pos += 1
        return tok

    def check(self, kind: str) -> bool:
        return self.peek().kind == kind

    def match(self, *kinds: str) -> Optional[Token]:
        if self.peek().kind in kinds:
            return self.advance()
        return None

    def expect(self, kind: str, context: str = "") -> Token:
        tok = self.peek()
        if tok.kind != kind:
            where = f" in {context}" if context else ""
            raise QLangSyntaxError(
                f"expected {kind}{where}, got {tok.kind} ({tok.value!r})",
                line=tok.line,
                col=tok.col,
            )
        return self.advance()

    def skip_newlines(self) -> None:
        while self.check("NEWLINE"):
            self.advance()

    def _reject_deferred(self) -> None:
        """If the current token is a DEFERRED_*, raise naming the
        feature.  Called at every place where a fresh statement or
        expression can start."""
        tok = self.peek()
        if tok.kind in _DEFERRED_FEATURE_NAMES:
            feature = _DEFERRED_FEATURE_NAMES[tok.kind]
            raise QLangSyntaxError(
                f"syntax {feature!r} is deferred to a future Q-Lang "
                f"version and rejected in v0.4 (SPEC §3.2)",
                line=tok.line,
                col=tok.col,
                feature=feature,
            )

    # ================================================================
    # Top level
    # ================================================================

    def parse_program(self) -> Program:
        decls: List = []
        self.skip_newlines()
        while not self.check("EOF"):
            decls.append(self.parse_decl())
            self.skip_newlines()
        return Program(decls=tuple(decls))

    def parse_decl(self):
        self._reject_deferred()
        tok = self.peek()
        if tok.kind == "LET":
            return self.parse_let_stmt()
        if tok.kind == "PRINT":
            return self.parse_print_stmt()
        if tok.kind == "EXPERIMENT":
            return self.parse_experiment()
        raise QLangSyntaxError(
            f"unexpected token {tok.kind} ({tok.value!r}); "
            f"expected 'let', 'print', or 'experiment'",
            line=tok.line,
            col=tok.col,
        )

    # ================================================================
    # let / print
    # ================================================================

    def parse_let_stmt(self) -> LetStmt:
        let_tok = self.expect("LET")
        name_tok = self.expect("IDENT", "let binding")
        self.expect("ASSIGN", "let binding")
        value = self.parse_expression()
        self._consume_statement_end()
        return LetStmt(
            name=name_tok.value,
            value=value,
            line=let_tok.line,
            col=let_tok.col,
        )

    def parse_print_stmt(self) -> PrintStmt:
        print_tok = self.expect("PRINT")
        value = self.parse_expression()
        self._consume_statement_end()
        return PrintStmt(
            value=value,
            line=print_tok.line,
            col=print_tok.col,
        )

    def _consume_statement_end(self) -> None:
        """A top-level statement must end with NEWLINE or EOF."""
        if self.check("NEWLINE"):
            self.advance()
        elif not self.check("EOF") and not self.check("RBRACE"):
            tok = self.peek()
            raise QLangSyntaxError(
                f"expected end of statement, got {tok.kind} ({tok.value!r})",
                line=tok.line,
                col=tok.col,
            )

    # ================================================================
    # experiment
    # ================================================================

    def parse_experiment(self) -> ExperimentDef:
        exp_tok = self.expect("EXPERIMENT")
        name_tok = self.expect("IDENT", "experiment declaration")
        self.expect("LBRACE", "experiment body")
        self.skip_newlines()

        clauses: List[ExpClause] = []
        while not self.check("RBRACE") and not self.check("EOF"):
            self._reject_deferred()
            tok = self.peek()
            if tok.kind == "GIVEN":
                clauses.append(self.parse_given_clause())
            elif tok.kind == "LET":
                clauses.append(self.parse_let_stmt())
            elif tok.kind == "INVARIANT":
                clauses.append(self.parse_invariant_clause())
            elif tok.kind == "RESULT":
                clauses.append(self.parse_result_clause())
            else:
                raise QLangSyntaxError(
                    f"unexpected token {tok.kind} ({tok.value!r}) in "
                    f"experiment body; expected one of "
                    f"'given:', 'let', 'invariant:', 'result:'",
                    line=tok.line,
                    col=tok.col,
                )
            self.skip_newlines()
        self.expect("RBRACE", "experiment body")

        return ExperimentDef(
            name=name_tok.value,
            clauses=tuple(clauses),
            line=exp_tok.line,
            col=exp_tok.col,
        )

    def parse_given_clause(self) -> GivenClause:
        given_tok = self.expect("GIVEN")
        self.expect("COLON", "given clause")
        params: List[GivenParam] = []
        params.append(self._parse_given_param())
        while self.check("COMMA") or self.check("NEWLINE"):
            saved = self.pos
            if self.check("COMMA"):
                self.advance()
                self.skip_newlines()
            else:
                self.skip_newlines()
            # A fresh param must start with an IDENT; otherwise we've
            # hit the next clause and should stop.
            if not self.check("IDENT"):
                self.pos = saved  # rewind
                break
            params.append(self._parse_given_param())
        return GivenClause(
            params=tuple(params),
            line=given_tok.line,
            col=given_tok.col,
        )

    def _parse_given_param(self) -> GivenParam:
        name_tok = self.expect("IDENT", "given parameter")
        unit: Optional[UnitExpr] = None
        if self.check("IN"):
            self.advance()
            unit = self.parse_unit_literal()
        return GivenParam(
            name=name_tok.value,
            unit=unit,
            line=name_tok.line,
            col=name_tok.col,
        )

    def parse_invariant_clause(self) -> InvariantClause:
        inv_tok = self.expect("INVARIANT")
        self.expect("COLON", "invariant clause")
        start_pos = self.pos
        expr = self.parse_expression()
        end_pos = self.pos
        # Grab the source slice (for error messages)
        src_text = " ".join(
            t.value
            for t in self.tokens[start_pos:end_pos]
            if t.kind not in ("NEWLINE", "EOF")
        )
        self._consume_statement_end()
        return InvariantClause(
            expression=expr,
            source_text=src_text,
            line=inv_tok.line,
            col=inv_tok.col,
        )

    def parse_result_clause(self) -> ResultClause:
        res_tok = self.expect("RESULT")
        self.expect("COLON", "result clause")
        expr = self.parse_expression()
        self._consume_statement_end()
        return ResultClause(
            expression=expr,
            line=res_tok.line,
            col=res_tok.col,
        )

    # ================================================================
    # Expressions (precedence climbs from low to high)
    # ================================================================

    def parse_expression(self):
        return self.parse_pipe()

    def parse_pipe(self):
        left = self.parse_comparison()
        while self.check("PIPE"):
            pipe_tok = self.advance()
            right = self._parse_call_for_pipe()
            left = PipeExpr(
                left=left,
                right=right,
                line=pipe_tok.line,
                col=pipe_tok.col,
            )
        return left

    def parse_comparison(self):
        """Comparison operators — lower precedence than +/-/*/ but higher
        than pipe.  Non-associative: we parse at most one chain.
        Supports <, <=, >, >=, ==, !=."""
        left = self.parse_add()
        cmp_kinds = ("LT", "LE", "GT", "GE", "EQ", "NE")
        if self.peek().kind in cmp_kinds:
            op_tok = self.advance()
            right = self.parse_add()
            left = BinaryOp(
                op=op_tok.value,
                left=left,
                right=right,
                line=op_tok.line,
                col=op_tok.col,
            )
        return left

    def _parse_call_for_pipe(self) -> Call:
        tok = self.peek()
        if tok.kind != "IDENT":
            raise QLangSyntaxError(
                f"pipe operator |> requires a call on the right, "
                f"got {tok.kind} ({tok.value!r})",
                line=tok.line,
                col=tok.col,
            )
        name = self.advance()
        self.expect("LPAREN", f"call to {name.value}")
        args, kwargs = self._parse_argument_list("RPAREN")
        self.expect("RPAREN", f"call to {name.value}")
        return Call(
            callee=name.value,
            args=tuple(args),
            kwargs=tuple(kwargs),
            line=name.line,
            col=name.col,
        )

    def parse_add(self):
        left = self.parse_mul()
        while self.check("PLUS") or self.check("MINUS"):
            op_tok = self.advance()
            right = self.parse_mul()
            left = BinaryOp(
                op=op_tok.value,
                left=left,
                right=right,
                line=op_tok.line,
                col=op_tok.col,
            )
        return left

    def parse_mul(self):
        left = self.parse_pow()
        while self.check("STAR") or self.check("SLASH"):
            op_tok = self.advance()
            right = self.parse_pow()
            left = BinaryOp(
                op=op_tok.value,
                left=left,
                right=right,
                line=op_tok.line,
                col=op_tok.col,
            )
        return left

    def parse_pow(self):
        left = self.parse_unc()
        if self.check("POW"):
            op_tok = self.advance()
            right = self.parse_pow()  # right-associative
            left = BinaryOp(
                op="**",
                left=left,
                right=right,
                line=op_tok.line,
                col=op_tok.col,
            )
        return left

    def parse_unc(self):
        left = self.parse_unary()
        if self.check("PLUSMINUS"):
            pm_tok = self.advance()
            right = self.parse_unary()
            return UncertaintyExpr(
                value=left,
                uncertainty=right,
                line=pm_tok.line,
                col=pm_tok.col,
            )
        return left

    def parse_unary(self):
        if self.check("MINUS"):
            minus_tok = self.advance()
            operand = self.parse_unary()
            node = UnaryOp(
                op="-",
                operand=operand,
                line=minus_tok.line,
                col=minus_tok.col,
            )
        else:
            node = self.parse_primary()
        # Optional 'in [unit]' unit conversion applies to any unary.
        if self.check("IN"):
            in_tok = self.advance()
            unit = self.parse_unit_literal()
            node = UnitConversion(
                expr=node,
                target_unit=unit,
                line=in_tok.line,
                col=in_tok.col,
            )
        return node

    def parse_primary(self):
        self._reject_deferred()
        tok = self.peek()

        if tok.kind == "NUMBER":
            self.advance()
            unit: Optional[UnitExpr] = None
            if self.check("LBRACKET"):
                unit = self.parse_unit_literal()
            return NumberLiteral(
                text=tok.value,
                unit=unit,
                line=tok.line,
                col=tok.col,
            )

        if tok.kind == "STRING":
            self.advance()
            return StringLiteral(
                value=tok.value,
                line=tok.line,
                col=tok.col,
            )

        if tok.kind == "SIMULATE":
            return self.parse_simulate_expr()

        if tok.kind == "LPAREN":
            self.advance()
            expr = self.parse_expression()
            self.expect("RPAREN", "parenthesised expression")
            return expr

        if tok.kind == "IDENT":
            name = self.advance()
            if self.check("LPAREN"):
                self.advance()
                args, kwargs = self._parse_argument_list("RPAREN")
                self.expect("RPAREN", f"call to {name.value}")
                return Call(
                    callee=name.value,
                    args=tuple(args),
                    kwargs=tuple(kwargs),
                    line=name.line,
                    col=name.col,
                )
            return Identifier(
                name=name.value,
                line=name.line,
                col=name.col,
            )

        raise QLangSyntaxError(
            f"unexpected token {tok.kind} ({tok.value!r}) in expression",
            line=tok.line,
            col=tok.col,
        )

    def _parse_argument_list(self, end_kind: str):
        """Shared by normal calls and simulate blocks.  Accepts a
        mix of positional and kwargs (IDENT ':' expr).  Kwargs must
        come after any positional."""
        args: List = []
        kwargs: List[Tuple[str, object]] = []
        self.skip_newlines()
        if self.check(end_kind):
            return args, kwargs
        while True:
            self.skip_newlines()
            # Kwarg?  Look ahead: IDENT COLON <expr>
            if self.check("IDENT") and self.peek(1).kind == "COLON":
                name = self.advance()
                self.advance()  # ':'
                value = self.parse_expression()
                kwargs.append((name.value, value))
            else:
                if kwargs:
                    tok = self.peek()
                    raise QLangSyntaxError(
                        "positional argument after keyword argument",
                        line=tok.line,
                        col=tok.col,
                    )
                args.append(self.parse_expression())
            self.skip_newlines()
            if self.check("COMMA"):
                self.advance()
                self.skip_newlines()
                # Trailing comma before closer is fine.
                if self.check(end_kind):
                    break
                continue
            break
        return args, kwargs

    def parse_simulate_expr(self) -> SimulateExpr:
        sim_tok = self.expect("SIMULATE")
        dom_tok = self.expect("IDENT", "simulate domain name")
        self.expect("LBRACE", "simulate block")
        self.skip_newlines()
        _, kwargs = self._parse_argument_list("RBRACE")
        self.skip_newlines()
        self.expect("RBRACE", "simulate block")
        return SimulateExpr(
            domain=dom_tok.value,
            kwargs=tuple(kwargs),
            line=sim_tok.line,
            col=sim_tok.col,
        )

    # ================================================================
    # Unit literals
    # ================================================================

    def parse_unit_literal(self) -> UnitExpr:
        lb = self.expect("LBRACKET", "unit literal")
        atoms: List[UnitAtom] = []
        atoms.append(self._parse_unit_atom(1))
        while self.check("STAR") or self.check("SLASH"):
            op = self.advance()
            sign = 1 if op.value == "*" else -1
            # Next atom's exponent is multiplied by sign
            next_atom = self._parse_unit_atom(sign)
            atoms.append(next_atom)
        self.expect("RBRACKET", "unit literal")
        return UnitExpr(
            atoms=tuple(atoms),
            line=lb.line,
            col=lb.col,
        )

    def _parse_unit_atom(self, outer_sign: int) -> UnitAtom:
        name_tok = self.expect("IDENT", "unit atom")
        exponent = 1
        if self.check("CARET"):
            self.advance()
            # Exponent: optional leading minus, then integer NUMBER.
            neg = 1
            if self.check("MINUS"):
                self.advance()
                neg = -1
            num_tok = self.expect("NUMBER", "unit exponent")
            try:
                exponent = neg * int(num_tok.value)
            except ValueError:
                raise QLangSyntaxError(
                    f"unit exponent must be an integer, got {num_tok.value!r}",
                    line=num_tok.line,
                    col=num_tok.col,
                )
        return UnitAtom(
            name=name_tok.value,
            exponent=outer_sign * exponent,
            line=name_tok.line,
            col=name_tok.col,
        )


# ─────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────


def parse(source: str) -> Program:
    tokens = tokenize(source)
    return Parser(tokens).parse_program()


def parse_tokens(tokens: List[Token]) -> Program:
    return Parser(tokens).parse_program()
