from dataclasses import dataclass
from choco.constant_folding import BinaryExprRewriter as FoldBinaryExprRewriter
from choco.constant_folding import EffectfulBinaryExprRewriter as FoldEffectfulBinaryExprRewriter
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from choco.dialects.choco_flat import *
from choco.dialects.choco_type import *


@dataclass
class LiteralRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
        self, literal: Literal, rewriter: PatternRewriter
    ) -> None:
        if len(literal.results[0].uses) == 0:
            rewriter.replace_op(literal, [], [None])
        return


@dataclass
class BinaryExprRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
        self, expr: BinaryExpr, rewriter: PatternRewriter
    ) -> None:
        if expr.op.data == "//":
            return
        if len(expr.results[0].uses) == 0:
            rewriter.replace_op(expr, [], [None])
        return

@dataclass
class UnaryExprRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
            self, expr: UnaryExpr, rewriter: PatternRewriter
    ) -> None:
        if len(expr.results[0].uses) == 0:
            rewriter.replace_op(expr, [], [None])
        return

@dataclass
class IndexStringRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
            self, index_string: IndexString, rewriter: PatternRewriter
    ) -> None:
        if len(index_string.results[0].uses) == 0:
            rewriter.replace_op(index_string, [], [None])
        return

@dataclass
class ListExprRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
        self, list: ListExpr, rewriter: PatternRewriter
    ) -> None:
        if len(list.results[0].uses) == 0:
            rewriter.replace_op(list, [], [None])
        return

@dataclass
class LoadRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
            self, load: IndexString, rewriter: PatternRewriter
    ) -> None:
        if len(load.results[0].uses) == 0:
            rewriter.replace_op(load, [], [None])
        return

@dataclass
class WhileRewriter(RewritePattern):
    def is_bool_literal(self, op: Operand):
        return (isinstance(op, Literal)) and ( isinstance(op.value, BoolAttr))

    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
            self, while_op: While, rewriter: PatternRewriter
    ) -> None:

        if self.is_bool_literal(while_op.cond.block.ops.first):
            if while_op.cond.block.ops.first.value.data == False:
                rewriter.replace_op(while_op, [], [])
                pass

@dataclass
class CallExprRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
            self, call_expr: CallExpr, rewriter: PatternRewriter
    ) -> None:
        if len(call_expr.results[0].uses) == 0:
            rewriter.replace_op(call_expr, [], [None])
        return

@dataclass
class GetAddressRewriter(RewritePattern):
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
            self, get_addr: GetAddress, rewriter: PatternRewriter
    ) -> None:
        if len(get_addr.results[0].uses) == 0:
            rewriter.replace_op(get_addr, [], [None])
        return


@dataclass
class IfRewriter(RewritePattern):
    def is_bool_literal(self, op: Operand):
        return isinstance(op.cond.op, Literal) and isinstance(op.cond.op.value, BoolAttr)

    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
            self, if_val : If, rewriter: PatternRewriter
    ) -> None:

        if self.is_bool_literal(if_val):

            if if_val.cond.op.value.data == True:

                replacement_ops = []
                for op in if_val.then.ops:
                    op.detach()
                    replacement_ops.append(op)
                rewriter.replace_op(if_val, replacement_ops, [])

            if if_val.cond.op.value.data == False:
                replacement_ops = []
                for op in if_val.orelse.ops:
                    op.detach()
                    replacement_ops.append(op)
                rewriter.replace_op(if_val, replacement_ops, [])
        return

class ChocoFlatDeadCodeElimination(ModulePass):
    name = "choco-flat-dead-code-elimination"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        fold_walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    FoldBinaryExprRewriter(),
                    FoldEffectfulBinaryExprRewriter(),

                    LiteralRewriter(),
                    BinaryExprRewriter(),
                    ListExprRewriter(),
                    UnaryExprRewriter(),
                    LoadRewriter(),
                    IndexStringRewriter(),
                    IfRewriter(),
                    WhileRewriter()
                ]
            ),
            walk_reverse=True,
        )

        dead_code_walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LiteralRewriter(),
                    BinaryExprRewriter(),
                    ListExprRewriter(),
                    UnaryExprRewriter(),
                    LoadRewriter(),
                    IndexStringRewriter(),
                    IfRewriter(),
                    WhileRewriter()
                ]
            ),
            walk_reverse=True,
        )
        fold_walker.rewrite_module(op)
        dead_code_walker.rewrite_module(op)

