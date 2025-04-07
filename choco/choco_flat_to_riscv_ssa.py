# type: ignore

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import (
    MLContext,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from choco.dialects.choco_flat import *
from riscv.ssa_dialect import *


class LiteralPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Literal, rewriter: PatternRewriter):
        value = op.value
        if isinstance(value, IntegerAttr):
            load_value_op = LIOp(op.value)
            rewriter.replace_op(op, [load_value_op])
            return

        if isinstance(value, BoolAttr):
            if value.data:
                load_value_op = LIOp(1)
            else:
                load_value_op = LIOp(0)
            rewriter.replace_op(op, load_value_op)
            return

        if isinstance(value, NoneAttr):
            load_none = LIOp(0)
            rewriter.replace_op(op, load_none)
            return

        if isinstance(value, StringAttr):
            load_length_op = LIOp(len(value.data))
            rewriter.insert_op_before_matched_op(load_length_op)

            # get total length
            load_one_op = LIOp(1)
            rewriter.insert_op_before_matched_op(load_one_op)
            total_length_op = AddOp(load_length_op, load_one_op)
            rewriter.insert_op_before_matched_op(total_length_op)

            # get total bytes
            load_byte_size_op = LIOp(4)
            rewriter.insert_op_before_matched_op(load_byte_size_op)
            total_length_bytes_op = MULOp(total_length_op, load_byte_size_op)
            rewriter.insert_op_before_matched_op(total_length_bytes_op)

            # create pointer
            alloc = CallOp("_malloc", total_length_bytes_op)
            rewriter.insert_op_before_matched_op(alloc)

            # write to string
            store_len = SWOp(load_length_op, alloc, 0)
            rewriter.insert_op_before_matched_op(store_len)
            for i in range(len(value.data)):
                ptr = (i + 1) * 4

                char_val = LIOp(ord(value.data[i]))
                rewriter.insert_op_before_matched_op(char_val)

                store = SWOp(char_val, alloc, ptr)
                rewriter.insert_op_before_matched_op(store)

            # gets value of alloc into a register
            get_alloc_op = AddIOp(alloc,0)
            rewriter.replace_op(op, [get_alloc_op])
            return
        return


class CallPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallExpr, rewriter: PatternRewriter):
        if op.func_name.data == "len":
            zero = LIOp(0)
            maybe_fail = BEQOp(op.args[0], zero, f"_error_len_none")
            read_size = LWOp(op.args[0], 0)
            rewriter.replace_op(op, [zero, maybe_fail, read_size])
            return

        call = CallOp(op.func_name, op.args, has_result=bool(len(op.results)))
        rewriter.replace_op(op, [call])


class AllocPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, alloc_op: Alloc, rewriter: PatternRewriter):
        alloc = AllocOp()
        rewriter.replace_op(alloc_op, alloc)


class StorePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, store_op: Store, rewriter: PatternRewriter):
        store = SWOp(store_op.value, store_op.memloc, 0)
        rewriter.replace_op(store_op, store)


class LoadPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, load_op: Load, rewriter: PatternRewriter):
        load = LWOp(load_op.memloc, 0)
        rewriter.replace_op(load_op, [load])


class UnaryExprPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, unary_op: UnaryExpr, rewriter: PatternRewriter):
        if unary_op.op.data == "not":
            one_op = LIOp(1)
            rewriter.insert_op_before_matched_op(one_op)
            oxor_op = XOROp(one_op, unary_op.value)
            rewriter.replace_op(unary_op,  oxor_op)
            return
        if unary_op.op.data == "-":
            load_zero_op = LIOp(0)
            rewriter.insert_op_before_matched_op(load_zero_op)
            negative_value_op = SubOp(load_zero_op, unary_op.value)
            rewriter.replace_op(unary_op, negative_value_op)
            return


class BinaryExprPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, bin_op: BinaryExpr, rewriter: PatternRewriter):
        lhs = bin_op.lhs
        rhs = bin_op.rhs
        ssa_ops = []
        match bin_op.op.data:
            case "%":
                rem_op = REMOp(lhs, rhs)
                ssa_ops = [rem_op]
            case "<":
                lt_op = SLTOp(lhs, rhs)
                ssa_ops = [lt_op]
            case ">":
                gt_op = SLTOp(rhs, lhs)
                ssa_ops = [gt_op]
            case "+":
                add_op = AddOp(lhs, rhs)
                ssa_ops = [add_op]
            case "-":
                sub_op = SubOp(lhs, rhs)
                ssa_ops = [sub_op]
            case "*":
                mul_op = MULOp(lhs, rhs)
                ssa_ops = [mul_op]
            case "//":
                div_op = DIVOp(lhs, rhs)
                ssa_ops = [div_op]

            case "==":
                eq_op = XOROp(lhs, rhs)
                one_op = LIOp(1)
                slt_op = SLTUOp(eq_op, one_op)
                ssa_ops = [eq_op, one_op, slt_op]
            case "is":
                eq_op = XOROp(lhs, rhs)
                one_op = LIOp(1)
                slt_op = SLTUOp(eq_op, one_op)
                ssa_ops = [eq_op, one_op, slt_op]
            case "!=":
                not_eq_op = XOROp(lhs, rhs)
                zero_op = LIOp(0)
                slt_op = SLTUOp(zero_op, not_eq_op)
                ssa_ops = [not_eq_op, zero_op, slt_op]
            case "<=":
                le_op = SLTOp(rhs, lhs)
                slt_op = XORIOp(le_op, 1)
                ssa_ops = [le_op, slt_op]
            case ">=":
                ge_op = SLTOp(lhs, rhs)
                slt_op = XORIOp(ge_op, 1)
                ssa_ops = [ge_op, slt_op]

        rewriter.replace_op(bin_op, ssa_ops)
        return


class IfPattern(RewritePattern):
    block_counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, if_op: If, rewriter: PatternRewriter):
        zero_op = LIOp(0)
        rewriter.insert_op_before_matched_op(zero_op)

        beq_op = BEQOp(if_op.cond, zero_op, "else_block_no_" + str(self.block_counter))
        rewriter.insert_op_before_matched_op(beq_op)

        # then block
        rewriter.inline_block_before_matched_op(if_op.then)

        # even though if_after_ is empty you need it to skip the else label
        jump_op = JOp("if_after_" + str(self.block_counter))
        rewriter.insert_op_before_matched_op(jump_op)

        else_block = LabelOp("else_block_no_" + str(self.block_counter))
        rewriter.insert_op_before_matched_op(else_block)
        rewriter.inline_block_before_matched_op(if_op.orelse)

        next_label = LabelOp("if_after_" + str(self.block_counter))
        rewriter.insert_op_before_matched_op(next_label)

        rewriter.erase_matched_op()
        self.block_counter += 1
        return


class AndPattern(RewritePattern):
    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, and_op: EffectfulBinaryExpr, rewriter: PatternRewriter):
        if and_op.op.data == "and":
            lhs_val = and_op.lhs.ops.last.value
            rhs_val = and_op.rhs.ops.last.value

            saved_result = AllocOp()
            rewriter.insert_op_before_matched_op(saved_result)

            # evaluate and save lhs
            lhs_label = LabelOp("and_lhs_" + str(self.counter))
            rewriter.insert_op_before_matched_op(lhs_label)
            rewriter.inline_block_before_matched_op(and_op.lhs)
            sw_op = SWOp(lhs_val, saved_result, 0)
            rewriter.insert_op_before_matched_op(sw_op)

            # if LHS is False short circuit out and escape to after
            zero_op = LIOp(0)
            rewriter.insert_op_before_matched_op(zero_op)
            beq_op = BEQOp(lhs_val, zero_op, "and_after_" + str(self.counter))
            rewriter.insert_op_before_matched_op(beq_op)

            # else continue with rhs
            rhs_label = LabelOp("and_rhs_" + str(self.counter))
            rewriter.insert_op_before_matched_op(rhs_label)
            rewriter.inline_block_before_matched_op(and_op.rhs)

            # get return value
            and_opp = ANDOp(lhs_val, rhs_val)
            rewriter.insert_op_before_matched_op(and_opp)
            store_result_op = SWOp(and_opp, saved_result, 0)
            rewriter.insert_op_before_matched_op(store_result_op)

            # return to program execution after and return result
            after_label = LabelOp("and_after_" + str(self.counter))
            rewriter.insert_op_before_matched_op(after_label)
            result = LWOp(saved_result, 0)
            rewriter.replace_op(and_op, result)

            self.counter += 1
            return


class OrPattern(RewritePattern):
    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, or_op: EffectfulBinaryExpr, rewriter: PatternRewriter):
        if or_op.op.data == "or":
            lhs_val = or_op.lhs.ops.last.value
            rhs_val = or_op.rhs.ops.last.value

            saved_result = AllocOp()
            rewriter.insert_op_before_matched_op(saved_result)


            # evaluate and save lhs result
            lhs_label = LabelOp("or_lhs_" + str(self.counter))
            rewriter.insert_op_before_matched_op(lhs_label)
            rewriter.inline_block_before_matched_op(or_op.lhs)
            store_result_op = SWOp(lhs_val, saved_result, 0)
            rewriter.insert_op_before_matched_op(store_result_op)

            # branch if False check rhs else short circuit
            one_op = LIOp(1)
            rewriter.insert_op_before_matched_op(one_op)
            beq_op = BEQOp(lhs_val, one_op, "or_after_" + str(self.counter))
            rewriter.insert_op_before_matched_op(beq_op)

            # evaluate rhs
            rhs_label = LabelOp("or_rhs_" + str(self.counter))
            rewriter.insert_op_before_matched_op(rhs_label)
            rewriter.inline_block_before_matched_op(or_op.rhs)

            # evaluate or
            or_opp = OROp(lhs_val, rhs_val)
            rewriter.insert_op_before_matched_op(or_opp)
            store_result_op = SWOp(or_opp, saved_result, 0)
            rewriter.insert_op_before_matched_op(store_result_op)

            # return to program execution after and return result
            after_label = LabelOp("or_after_" + str(self.counter))
            rewriter.insert_op_before_matched_op(after_label)
            load_result_op = LWOp(saved_result, 0)
            rewriter.replace_op(or_op, [load_result_op])

            self.counter += 1
            return
# this is for 1 line if statements


class IfExprPattern(RewritePattern):
    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, if_op: IfExpr, rewriter: PatternRewriter):
        # same idea as

        or_else = if_op.or_else.ops.last
        then = if_op.then.ops.last

        save_value_ptr = AllocOp()
        rewriter.insert_op_before_matched_op(save_value_ptr)

        # branch and don't evaluate the then block if false
        zero_op = LIOp(0)
        rewriter.insert_op_before_matched_op(zero_op)
        branch_op = BEQOp(if_op.cond, zero_op, "if_expr_else_" + str(self.counter))
        rewriter.insert_op_before_matched_op(branch_op)

        # else evaluate then block and skip else
        rewriter.inline_block_before_matched_op(if_op.then)
        save_then_result_op = SWOp(then.value, save_value_ptr, 0)
        rewriter.insert_op_before_matched_op(save_then_result_op)
        jump_op = JOp("if_expr_after_" + str(self.counter))
        rewriter.insert_op_before_matched_op(jump_op)

        # evaluate else block
        else_label = LabelOp("if_expr_else_" + str(self.counter))
        rewriter.insert_op_before_matched_op(else_label)
        rewriter.inline_block_before_matched_op(if_op.or_else)
        save_else_result_op = SWOp(or_else.value, save_value_ptr, 0)
        rewriter.insert_op_before_matched_op(save_else_result_op)

        # after block
        after_label = LabelOp("if_expr_after_" + str(self.counter))
        rewriter.insert_op_before_matched_op(after_label)

        load_op = LWOp(save_value_ptr, 0)
        rewriter.replace_op(if_op, load_op)

        self.counter += 1
        return


class WhilePattern(RewritePattern):
    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, while_op: While, rewriter: PatternRewriter):
        cond = while_op.cond.ops.last

        while_condition_ptr = AllocOp()
        rewriter.insert_op_before_matched_op(while_condition_ptr)

        # START
        start = LabelOp("while_start_" + str(self.counter))
        rewriter.insert_op_before_matched_op(start)

        rewriter.inline_block_before_matched_op(while_op.cond)


        save_result_instruction = SWOp(cond.value, while_condition_ptr, 0)
        rewriter.insert_op_before_matched_op(save_result_instruction)

        zero_op = LIOp(0)
        rewriter.insert_op_before_matched_op(zero_op)

        while_condition_result = LWOp(while_condition_ptr,0)
        rewriter.insert_op_before_matched_op(while_condition_result)
        beq_op = BEQOp(while_condition_result, zero_op, "while_after_" + str(self.counter))
        rewriter.insert_op_before_matched_op(beq_op)


        # BODY, while start can implicitly go into while body
        body = LabelOp("while_body_" + str(self.counter))
        rewriter.insert_op_before_matched_op(body)
        rewriter.inline_block_before_matched_op(while_op.body)
        jump_op = JOp("while_start_" + str(self.counter))
        rewriter.insert_op_before_matched_op(jump_op)

        # AFTER
        after = LabelOp("while_after_" + str(self.counter))
        rewriter.insert_op_before_matched_op(after)

        rewriter.erase_matched_op()

        self.counter += 1
        return


class ListExprPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, list_expr: ListExpr, rewriter: PatternRewriter):

        # get length of list
        num_elems = LIOp(len(list_expr.elems))
        rewriter.insert_op_before_matched_op(num_elems)

        # get total length of list including length
        one_op = LIOp(1)
        rewriter.insert_op_before_matched_op(one_op)
        total_length_op = AddOp(num_elems, one_op)
        rewriter.insert_op_before_matched_op(total_length_op)

        # get total space required
        four_op = LIOp(4)
        rewriter.insert_op_before_matched_op(four_op)
        total_space_op = MULOp(total_length_op, four_op)
        rewriter.insert_op_before_matched_op(total_space_op)

        # allocate the required space
        list_ptr = CallOp("_malloc", [total_space_op])
        rewriter.insert_op_before_matched_op(list_ptr)
        store_len = SWOp(num_elems, list_ptr, 0)
        rewriter.insert_op_before_matched_op(store_len)

        # allocate individual elements
        for i in range(len(list_expr.elems)):
            index_offset = (i + 1) * 4
            store = SWOp(list_expr.elems[i], list_ptr, index_offset)
            rewriter.insert_op_before_matched_op(store)

        # do this to load the value of list_ptr, not the value at list_ptr
        load_ptr_val = AddIOp(list_ptr, 0)
        rewriter.replace_op(list_expr, [load_ptr_val])
        return


class GetAddressPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, get_address: GetAddress, rewriter: PatternRewriter):

        # first check it is a valid index
        zero_op = LIOp(0)
        rewriter.insert_op_before_matched_op(zero_op)
        none_check_op = BEQOp(get_address.value, zero_op, "_list_index_none")
        rewriter.insert_op_before_matched_op(none_check_op)
        oob_below_check_op = BLTOp(get_address.index, zero_op, "_list_index_oob")
        rewriter.insert_op_before_matched_op(oob_below_check_op)

        # get index (account for first value being length)
        one_op = LIOp(1)
        rewriter.insert_op_before_matched_op(one_op)
        index_op = AddOp(get_address.index, one_op)
        rewriter.insert_op_before_matched_op(index_op)

        # check valid index
        list_length_op = LWOp(get_address.value, 0)
        rewriter.insert_op_before_matched_op(list_length_op)
        oob_above_check = BGEOp(get_address.index, list_length_op, "_list_index_oob")
        rewriter.insert_op_before_matched_op(oob_above_check)

        # calculate offset in bytes
        four_op = LIOp(4)
        rewriter.insert_op_before_matched_op(four_op)
        offset = MULOp(index_op, four_op)
        rewriter.insert_op_before_matched_op(offset)
        addr = AddOp(get_address.value, offset)
        rewriter.replace_op(get_address,addr)
        return


class IndexStringPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, indexString: IndexString, rewriter: PatternRewriter):

        load_four_op = LIOp(4)
        rewriter.insert_op_before_matched_op(load_four_op)
        load_one_op = LIOp(1)
        rewriter.insert_op_before_matched_op(load_one_op)

        # one accounts for the first index being length
        index_plus_one_op = AddOp(indexString.index, load_one_op)
        rewriter.insert_op_before_matched_op(index_plus_one_op)

        # address offset
        offset = MULOp(index_plus_one_op, load_four_op)
        rewriter.insert_op_before_matched_op(offset)

        # actual address = base + offset
        addr = AddOp(indexString.value, offset)
        rewriter.insert_op_before_matched_op(addr)

        # we must create a new string of length 2 (first index length (1) and second index char)
        length = LIOp(8)
        rewriter.insert_op_before_matched_op(length)
        new_string_ptr = CallOp("_malloc", [length])
        rewriter.insert_op_before_matched_op(new_string_ptr)

        # store length of 1, don't include length value itself, at offset 0 in new string
        one = LIOp(1)
        rewriter.insert_op_before_matched_op(one)
        store_length = SWOp(one, new_string_ptr, 0)
        rewriter.insert_op_before_matched_op(store_length)

        # store char at index 1
        char = LWOp(addr, 0)
        rewriter.insert_op_before_matched_op(char)
        store_char = SWOp(char, new_string_ptr, 4)
        rewriter.insert_op_before_matched_op(store_char)

        new_ptr = AllocOp()
        store_pointer = SWOp(new_string_ptr, new_ptr, 0)

        rewriter.replace_op(indexString, [new_ptr, store_pointer], new_ptr.results)


class YieldPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, get_address: Yield, rewriter: PatternRewriter):
        rewriter.erase_matched_op()


class FuncDefPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, func: FuncDef, rewriter: PatternRewriter):
        new_func = FuncOp.create(
            result_types=[],
            properties={"func_name": StringAttr(func.func_name.data)},
        )

        new_region = rewriter.move_region_contents_to_new_regions(func.func_body)
        new_func.add_region(new_region)
        for arg in new_region.blocks[0].args:
            rewriter.modify_block_argument_type(arg, RegisterType())

        rewriter.replace_op(func, [new_func])


class ReturnPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, ret: Return, rewriter: PatternRewriter):
        return_op = ReturnOp(ret.value)
        rewriter.replace_op(ret, return_op)
        return


class ChocoFlatToRISCVSSA(ModulePass):
    name = "choco-flat-to-riscv-ssa"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LiteralPattern(),
                    CallPattern(),
                    UnaryExprPattern(),
                    BinaryExprPattern(),
                    StorePattern(),
                    LoadPattern(),
                    AllocPattern(),
                    IfPattern(),
                    AndPattern(),
                    OrPattern(),
                    IfExprPattern(),
                    WhilePattern(),
                    ListExprPattern(),
                    GetAddressPattern(),
                    IndexStringPattern(),
                    FuncDefPattern(),
                    ReturnPattern(),
                ]
            ),
            apply_recursively=True,
        )

        walker.rewrite_module(op)

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    YieldPattern(),
                ]
            ),
            apply_recursively=True,
        )

        walker.rewrite_module(op)
