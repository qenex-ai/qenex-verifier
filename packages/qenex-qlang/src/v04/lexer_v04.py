"""
Q-Lang v0.4 — lexer (SPEC §3, §4).

Turns a source string into a flat token list.  The lexer is strict:
any character or sequence that is not part of SPEC §4's grammar is
rejected with a ``QLangSyntaxError`` that carries the 1-indexed
line and column of the offending character.

Token model
-----------
Each token is a plain dataclass with four fields:

    kind:   short string ('NUMBER', 'IDENT', 'PLUS', ...)
    value:  the literal source string (for NUMBER, IDENT, STRING) or
            the canonical form of the operator (e.g. '+', '**', '|>')
    line:   1-indexed source line
    col:    1-indexed source column

Tokens are emitted in source order with a synthetic ``NEWLINE`` at
the end of every non-blank statement line (see §4 grammar) and a
final ``EOF`` token.  Blank lines and comments never emit a
``NEWLINE`` token — the parser sees only meaningful newlines.

Keywords
--------
v0.4 keywords are listed in ``KEYWORDS``.  Any identifier spelling
matching a keyword is emitted as its specific token kind (e.g.
``'let'`` → ``'LET'``).  The grammar in SPEC §4 treats these as
reserved words.

Deferred-feature keywords (SPEC §3.2) — ``if``, ``else``, ``while``,
``for``, ``fn``, ``def``, ``prove``, ``reason``, ``generate``,
``optimize``, ``match`` — are tokenised as their deferred-feature
kinds so the parser can produce a ``QLangSyntaxError`` that names
the specific forbidden construct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from errors_v04 import QLangSyntaxError  # type: ignore[import-not-found]


# ─────────────────────────────────────────────────────────────────────
# Token type
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    line: int
    col: int

    def __repr__(self) -> str:  # short, debug-friendly
        return f"{self.kind}({self.value!r})@{self.line}:{self.col}"


# ─────────────────────────────────────────────────────────────────────
# Keywords
# ─────────────────────────────────────────────────────────────────────

# v0.4 keywords — each produces a dedicated token kind.
KEYWORDS: dict[str, str] = {
    "let": "LET",
    "print": "PRINT",
    "experiment": "EXPERIMENT",
    "given": "GIVEN",
    "invariant": "INVARIANT",
    "result": "RESULT",
    "simulate": "SIMULATE",
    "in": "IN",
}

# v0.5+ keywords — tokenised but the parser rejects with a message
# that names the deferred feature (SPEC §3.2).
DEFERRED_KEYWORDS: dict[str, str] = {
    "if": "DEFERRED_IF",
    "else": "DEFERRED_ELSE",
    "elif": "DEFERRED_ELIF",
    "while": "DEFERRED_WHILE",
    "for": "DEFERRED_FOR",
    "fn": "DEFERRED_FN",
    "def": "DEFERRED_DEF",
    "prove": "DEFERRED_PROVE",
    "reason": "DEFERRED_REASON",
    "generate": "DEFERRED_GENERATE",
    "optimize": "DEFERRED_OPTIMIZE",
    "match": "DEFERRED_MATCH",
    "return": "DEFERRED_RETURN",
    "yield": "DEFERRED_YIELD",
    "import": "DEFERRED_IMPORT",
    "class": "DEFERRED_CLASS",
    "lambda": "DEFERRED_LAMBDA",
}


# ─────────────────────────────────────────────────────────────────────
# Single-character punctuation
# ─────────────────────────────────────────────────────────────────────


SINGLE_CHAR_TOKENS: dict[str, str] = {
    "+": "PLUS",
    "-": "MINUS",
    "*": "STAR",  # may combine to '**' — handled in lexer
    "/": "SLASH",
    "(": "LPAREN",
    ")": "RPAREN",
    "{": "LBRACE",
    "}": "RBRACE",
    "[": "LBRACKET",
    "]": "RBRACKET",
    ",": "COMMA",
    ":": "COLON",
    "^": "CARET",
}


# ─────────────────────────────────────────────────────────────────────
# Lexer
# ─────────────────────────────────────────────────────────────────────


class Lexer:
    """One-pass character stream → token list."""

    def __init__(self, source: str) -> None:
        self.src = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []
        # Whether the last emitted token was a NEWLINE (or we're at
        # start-of-file).  We collapse consecutive NEWLINEs and we do
        # not emit a NEWLINE from a blank line or a comment line.
        self._at_logical_sol = True

    # ---- public API ----

    def tokenize(self) -> List[Token]:
        while self.pos < len(self.src):
            ch = self.src[self.pos]

            # Newline
            if ch == "\n":
                self._emit_newline()
                self._advance_line()
                continue

            # Whitespace (space, tab, carriage return)
            if ch in " \t\r":
                self._advance(1)
                continue

            # Comment to end of line
            if ch == "#":
                self._skip_to_eol()
                continue

            # String literal
            if ch == '"':
                self._read_string()
                continue

            # Number (NUMBER is signed by the parser via unary-minus,
            # so here we only match unsigned digits).
            if ch.isdigit() or (ch == "." and self._peek_next_is_digit()):
                self._read_number()
                continue

            # Identifier or keyword
            if ch.isalpha() or ch == "_":
                self._read_identifier()
                continue

            # Multi-character operators first
            if ch == "|" and self._peek(1) == ">":
                self._emit("PIPE", "|>")
                self._advance(2)
                continue

            if ch == "*" and self._peek(1) == "*":
                self._emit("POW", "**")
                self._advance(2)
                continue

            if ch == "+" and self._peek(1) == "/" and self._peek(2) == "-":
                self._emit("PLUSMINUS", "+/-")
                self._advance(3)
                continue

            if ch == "<" and self._peek(1) == "=":
                self._emit("LE", "<=")
                self._advance(2)
                continue

            if ch == ">" and self._peek(1) == "=":
                self._emit("GE", ">=")
                self._advance(2)
                continue

            if ch == "=" and self._peek(1) == "=":
                self._emit("EQ", "==")
                self._advance(2)
                continue

            if ch == "!" and self._peek(1) == "=":
                self._emit("NE", "!=")
                self._advance(2)
                continue

            # Single comparison operators
            if ch == "<":
                self._emit("LT", "<")
                self._advance(1)
                continue
            if ch == ">":
                self._emit("GT", ">")
                self._advance(1)
                continue

            # Assignment
            if ch == "=":
                self._emit("ASSIGN", "=")
                self._advance(1)
                continue

            # Single-char punctuation
            if ch in SINGLE_CHAR_TOKENS:
                self._emit(SINGLE_CHAR_TOKENS[ch], ch)
                self._advance(1)
                continue

            # Unknown character — strict rejection
            raise QLangSyntaxError(
                f"unexpected character {ch!r}",
                line=self.line,
                col=self.col,
            )

        # Trailing logical newline + EOF
        self._emit_newline()
        self.tokens.append(Token("EOF", "", self.line, self.col))
        return self.tokens

    # ---- scanning helpers ----

    def _peek(self, offset: int = 0) -> str:
        i = self.pos + offset
        return self.src[i] if i < len(self.src) else ""

    def _peek_next_is_digit(self) -> bool:
        return self._peek(1).isdigit()

    def _advance(self, n: int) -> None:
        self.pos += n
        self.col += n

    def _advance_line(self) -> None:
        self.pos += 1
        self.line += 1
        self.col = 1

    def _skip_to_eol(self) -> None:
        while self.pos < len(self.src) and self.src[self.pos] != "\n":
            self._advance(1)

    # ---- token emission ----

    def _emit(self, kind: str, value: str) -> None:
        self.tokens.append(Token(kind, value, self.line, self.col))
        self._at_logical_sol = False

    def _emit_newline(self) -> None:
        """Emit NEWLINE if the last meaningful token wasn't one."""
        if self._at_logical_sol:
            return
        self.tokens.append(Token("NEWLINE", "\n", self.line, self.col))
        self._at_logical_sol = True

    # ---- literal readers ----

    def _read_number(self) -> None:
        start_line, start_col = self.line, self.col
        start = self.pos

        # Integer part
        while self.pos < len(self.src) and self.src[self.pos].isdigit():
            self._advance(1)

        # Fractional part
        if self._peek() == "." and self._peek(1).isdigit():
            self._advance(1)
            while self.pos < len(self.src) and self.src[self.pos].isdigit():
                self._advance(1)
        elif self._peek() == "." and not self._peek(1).isdigit():
            # allow "3." form: digit+ "." with nothing after
            self._advance(1)

        # Exponent
        if self._peek().lower() == "e":
            self._advance(1)
            if self._peek() in "+-":
                self._advance(1)
            if not self._peek().isdigit():
                raise QLangSyntaxError(
                    "malformed number: missing exponent digits",
                    line=self.line,
                    col=self.col,
                )
            while self.pos < len(self.src) and self.src[self.pos].isdigit():
                self._advance(1)

        text = self.src[start : self.pos]
        self.tokens.append(Token("NUMBER", text, start_line, start_col))
        self._at_logical_sol = False

    def _read_string(self) -> None:
        start_line, start_col = self.line, self.col
        # Skip opening "
        self._advance(1)
        buf: list[str] = []
        while self.pos < len(self.src) and self.src[self.pos] != '"':
            ch = self.src[self.pos]
            if ch == "\n":
                raise QLangSyntaxError(
                    "unterminated string literal",
                    line=start_line,
                    col=start_col,
                )
            if ch == "\\":
                nxt = self._peek(1)
                if nxt == "n":
                    buf.append("\n")
                    self._advance(2)
                    continue
                if nxt == "t":
                    buf.append("\t")
                    self._advance(2)
                    continue
                if nxt == "\\":
                    buf.append("\\")
                    self._advance(2)
                    continue
                if nxt == '"':
                    buf.append('"')
                    self._advance(2)
                    continue
                if nxt == "":
                    raise QLangSyntaxError(
                        "unterminated string literal",
                        line=start_line,
                        col=start_col,
                    )
                # Unknown escape — keep both chars verbatim
                buf.append("\\")
                buf.append(nxt)
                self._advance(2)
                continue
            buf.append(ch)
            self._advance(1)
        if self.pos >= len(self.src):
            raise QLangSyntaxError(
                "unterminated string literal",
                line=start_line,
                col=start_col,
            )
        # Skip closing "
        self._advance(1)
        self.tokens.append(Token("STRING", "".join(buf), start_line, start_col))
        self._at_logical_sol = False

    def _read_identifier(self) -> None:
        start_line, start_col = self.line, self.col
        start = self.pos
        while self.pos < len(self.src) and (
            self.src[self.pos].isalnum() or self.src[self.pos] == "_"
        ):
            self._advance(1)
        text = self.src[start : self.pos]

        if text in KEYWORDS:
            kind = KEYWORDS[text]
        elif text in DEFERRED_KEYWORDS:
            kind = DEFERRED_KEYWORDS[text]
        else:
            kind = "IDENT"

        self.tokens.append(Token(kind, text, start_line, start_col))
        self._at_logical_sol = False


# ─────────────────────────────────────────────────────────────────────
# Convenience entry point
# ─────────────────────────────────────────────────────────────────────


def tokenize(source: str) -> List[Token]:
    """Lex ``source`` into a token list.  Raises ``QLangSyntaxError``
    on any invalid character or malformed literal."""
    return Lexer(source).tokenize()
