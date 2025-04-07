# type: ignore
from dataclasses import dataclass

from xdsl.dialects.builtin import IntegerAttr, ModuleOp
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
class BinaryExprRewriter(RewritePattern):
    def is_integer_literal(self, op: Operation):
        return isinstance(op, Literal) and isinstance(op.value, IntegerAttr)

    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
        self, expr: BinaryExpr, rewriter: PatternRewriter
    ) -> None:

        if expr.op.data == "+":
            if self.is_integer_literal(expr.lhs.op) and self.is_integer_literal(
                    expr.rhs.op
            ):
                lhs_value = expr.lhs.op.value.parameters[0].data
                rhs_value = expr.rhs.op.value.parameters[0].data
                resolve_op = lhs_value + rhs_value
                new_constant = Literal.get(resolve_op)
                rewriter.replace_op(expr, [new_constant])
        if expr.op.data == "-":
            if both_sides_int_literals(expr.lhs.op,expr.rhs.op):
                lhs_value = expr.lhs.op.value.parameters[0].data
                rhs_value = expr.rhs.op.value.parameters[0].data
                resolve_op = lhs_value - rhs_value
                new_constant = Literal.get(resolve_op)
                rewriter.replace_op(expr, [new_constant])

        if expr.op.data == "*":
            if self.is_integer_literal(expr.lhs.op):
                lhs_value = expr.lhs.op.value.parameters[0].data
                if lhs_value == 0:
                    rewriter.replace_op(expr, Literal.get(0))

            elif self.is_integer_literal(expr.rhs.op):
                rhs_value = expr.rhs.op.value.parameters[0].data
                if rhs_value == 0:
                    rewriter.replace_op(expr, Literal.get(0))

            elif both_sides_int_literals(expr.lhs.op,expr.rhs.op):
                lhs_value = expr.lhs.op.value.parameters[0].data
                rhs_value = expr.rhs.op.value.parameters[0].data
                resolve_op = lhs_value * rhs_value
                new_constant = Literal.get(resolve_op)
                rewriter.replace_op(expr, [new_constant])

        if expr.op.data == "/":
            if both_sides_int_literals(expr.lhs.op, expr.rhs.op):
                if rhs_value != 0:
                    lhs_value = expr.lhs.op.value.parameters[0].data
                    rhs_value = expr.rhs.op.value.parameters[0].data
                    resolve_op = lhs_value / rhs_value
                    new_constant = Literal.get(resolve_op)
                    rewriter.replace_op(expr, [new_constant])

        if expr.op.data == "<":
            if both_sides_int_literals(expr.lhs.op,expr.rhs.op):
                lhs_value = expr.lhs.op.value.parameters[0].data
                rhs_value = expr.rhs.op.value.parameters[0].data
                resolve_op = Literal.get(lhs_value < rhs_value)
                rewriter.replace_op(expr, [resolve_op])


        if expr.op.data == ">":
            if both_sides_int_literals(expr.lhs.op,expr.rhs.op):
                lhs_value = expr.lhs.op.value.parameters[0].data
                rhs_value = expr.rhs.op.value.parameters[0].data
                resolve_op = Literal.get(lhs_value > rhs_value)
                rewriter.replace_op(expr, [resolve_op])

        if expr.op.data == "<=":
            if both_sides_int_literals(expr.lhs.op,expr.rhs.op):
                lhs_value = expr.lhs.op.value.parameters[0].data
                rhs_value = expr.rhs.op.value.parameters[0].data
                resolve_op = Literal.get(lhs_value <= rhs_value)
                rewriter.replace_op(expr, [resolve_op])

        if expr.op.data == ">=":
            if both_sides_int_literals(expr.lhs.op, expr.rhs.op):
                lhs_value = expr.lhs.op.value.parameters[0].data
                rhs_value = expr.rhs.op.value.parameters[0].data
                resolve_op = Literal.get(lhs_value >= rhs_value)
                rewriter.replace_op(expr, [resolve_op])

        if expr.op.data == "==":
            if both_sides_int_literals(expr.lhs.op, expr.rhs.op):
                lhs_value = expr.lhs.op.value.parameters[0].data
                rhs_value = expr.rhs.op.value.parameters[0].data
                resolve_op = Literal.get(lhs_value == rhs_value)
                rewriter.replace_op(expr, [resolve_op])


        if expr.op.data == "!=":
            if both_sides_int_literals(expr.lhs.op, expr.rhs.op):
                lhs_value = expr.lhs.op.value.parameters[0].data
                rhs_value = expr.rhs.op.value.parameters[0].data
                resolve_op = Literal.get(lhs_value != rhs_value)
                rewriter.replace_op(expr, [resolve_op])

        if expr.op.data == "%":
            if both_sides_int_literals(expr.lhs.op, expr.rhs.op):
                lhs_value = expr.lhs.op.value.parameters[0].data
                rhs_value = expr.rhs.op.value.parameters[0].data
                resolve_op = Literal.get(lhs_value % rhs_value)
                rewriter.replace_op(expr, [resolve_op])
        return

def both_sides_int_literals(lhs_op, rhs_op):
    if is_integer_literal(lhs_op) and is_integer_literal(rhs_op):
        return True
    else:
        return False

def is_integer_literal(op: Operation):
    return isinstance(op, Literal) and isinstance(op.value, IntegerAttr)


@dataclass
class EffectfulBinaryExprRewriter(RewritePattern):
    def is_bool_literal(self, op: Operation):
        return isinstance(op, Literal) and isinstance(op.value, BoolAttr)
    def is_variable(self, op: Operation):
        return isinstance(op, Load)
    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
            self, expr: EffectfulBinaryExpr, rewriter: PatternRewriter
    ) -> None:
        if expr.op.data == "and":
            if self.is_bool_literal(expr.lhs.ops.first):
                if expr.lhs.ops.first.value.data == True:
                    if self.is_variable(expr.rhs.ops.first):
                        l_var = expr.rhs.ops.first
                        idex_mem : Alloc = l_var.memloc.op
                        new_l_var = Load.create(operands=[idex_mem.results[0]], result_types=[choco_type.bool_type])
                        rewriter.replace_op(expr, new_l_var,[new_l_var.result])

            if self.is_bool_literal(expr.rhs.ops.first):
                if expr.rhs.ops.first.value.data == True:
                    if self.is_variable(expr.lhs.ops.first):
                        l_var = expr.lhs.ops.first
                        idex_mem : Alloc = l_var.memloc.op
                        new_l_var = Load.create(operands=[idex_mem.results[0]], result_types=[choco_type.bool_type])
                        rewriter.replace_op(expr, new_l_var,[new_l_var.result])

        if expr.op.data == "or":

            if self.is_bool_literal(expr.lhs.ops.first):
                if expr.lhs.ops.first.value.data == False:
                    if self.is_variable(expr.rhs.ops.first):
                        l_var = expr.rhs.ops.first
                        idex_mem: Alloc = l_var.memloc.op
                        new_l_var = Load.create(operands=[idex_mem.results[0]], result_types=[choco_type.bool_type])
                        rewriter.replace_op(expr, new_l_var, [new_l_var.result])

            if self.is_bool_literal(expr.rhs.ops.first):
                if expr.rhs.ops.first.value.data == False:
                    if self.is_variable(expr.lhs.ops.first):
                        l_var = expr.lhs.ops.first
                        idex_mem: Alloc = l_var.memloc.op
                        new_l_var = Load.create(operands=[idex_mem.results[0]], result_types=[choco_type.bool_type])
                        rewriter.replace_op(expr, new_l_var, [new_l_var.result])


class ChocoFlatConstantFolding(ModulePass):
    name = "choco-flat-constant-folding"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    BinaryExprRewriter(),
                    EffectfulBinaryExprRewriter()
                ]
            )
        )

        walker.rewrite_module(op)
