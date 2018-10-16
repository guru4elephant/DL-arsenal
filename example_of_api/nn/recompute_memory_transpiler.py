import sys
import os
from paddle.fluid import core 
class RecomputeTranspiler(object):
    def __init__(self, memory_pool, checkpoints, weight_dict):
        self.pool_vars = memory_pool
        self.alloced_vars = {}
        for key in range(len(self.pool_vars)):
            self.alloced_vars[key] = 0
        self.checkpoints = {}
        for cp in checkpoints:
            group = cp.split("$")
            self.checkpoints[group[1]] = group[0]
        self.weight_dict = weight_dict
        self.max_var_num = 0
        self.renamed_dict = {}
        self.var_size_dict = {}

    def print_var_by_id(self, id):
        out_str = "  vars {\n" + \
                  '''    name: "%d_rep_var"\n''' % id + \
                  "    type {\n" + \
                  "    type: LOD_TENSOR\n" + \
                  "      lod_tensor {\n" + \
                  "        tensor {\n" + \
                  "          data_type: FP32\n" + \
                  "          dims: 256\n" + \
                  "        }\n" + \
                  "      }\n" + \
                  "    }\n" + \
                  "  }\n"
        return out_str
    
    def print_op(self, op):
        out_str = "  ops{\n"
        lines = str(op).split("\n")
        for line in lines:
            out_str += "    " + line + "\n"
        out_str += "  }\n"
        return out_str
    
    def print_var(self, var):
        out_str = "  vars {\n"
        lines = str(var).split("\n")
        for line in lines:
            out_str += "    " + line + "\n"
        out_str += "  }\n"
        return out_str

    def get_optimized_program(self, start_prog, main_prog, forward_partitions,
                              backward_partitions, opt_ops,
                              main_vars):
        pass

    def dump_optimized_program(self, start_prog, main_prog, forward_partitions,
                               backward_partitions, opt_ops,
                               main_vars, start_prog_file, main_prog_file):
        fout = open(start_prog_file, "w")
        fout.write(str(start_prog))
        fout.write("\n")
        
        fout = open(main_prog_file, "w")
        fout.write("blocks {\n")
        fout.write("  idx: 0\n")
        fout.write("  parent_idx: -1\n")
        name_keys = main_prog.current_block().vars.keys()
        local_vars = main_prog.current_block().vars
        for name in name_keys:
            if name not in self.renamed_dict:
                out_str = self.print_var(local_vars[name])
                fout.write(out_str)
                fout.write("\n")
        for key in self.alloced_vars:
            if self.alloced_vars[key] > 0:
                fout.write(self.print_var_by_id(key))
                fout.write("\n")
        for part in forward_partitions:
            for op in part:
                fout.write(self.print_op(op))
                fout.write("\n")
        for part in backward_partitions:
            for op in part:
                fout.write(self.print_op(op))
                fout.write("\n")
        for op in opt_ops:
            fout.write(self.print_op(op))
            fout.write("\n")
        fout.write("}\n")
            
    def get_optimized_program(self):
        self.startup_program = None
        self.main_program = None

    def get_max_var_num(self):
        return self.max_var_num

    def get_nearest_var(self, var_name, used_idx):
        var_size = self.var_size_dict[var_name]
        min_diff = 1e+10
        min_index = -1
        for key in self.alloced_vars:
            if key in used_idx:
                continue
            #diff = abs(var_size - self.alloced_vars[key])
            diff = self.alloced_vars[key] - var_size
            if diff < min_diff and diff >= 0:
                min_diff = diff
                min_index = key
        if min_index == -1:
            for key in self.alloced_vars:
                if key in used_idx:
                    continue
                if self.alloced_vars[key] == 0:
                    min_index = key
                    break

        if self.alloced_vars[min_index] < var_size:
            self.alloced_vars[min_index] = var_size
        return min_index
        
    def rewrite_ops(self, ops):
        # heuristic rewrite algorithm
        # if a var is not checkpoint: 
        #   if current var has been rewritten: reuse the rewritten var
        #   else:
        #     1) find the nearest var by size and rewrite current var
        #     2) update the size of the rewritten var by current var's size
        #     3) put the rewritten var into used var dict
        pool_idx = 0
        local_dict = {}
        used_idx = {}
        for op in ops:
            for i, name in enumerate(op.input_arg_names):
                if name not in self.checkpoints and name not in self.weight_dict:
                    if name in local_dict:
                        op._rename_input(op.input_arg_names[i], local_dict[name])
                    else:
                        self.renamed_dict[name] = 1
                        pool_idx = self.get_nearest_var(name, used_idx)
                        used_idx[pool_idx] = 1
                        op._rename_input(op.input_arg_names[i], self.pool_vars[pool_idx])
                        local_dict[name] = self.pool_vars[pool_idx]

            for i, name in enumerate(op.output_arg_names):
                if name not in self.checkpoints and name not in self.weight_dict:
                    if name in local_dict:
                        op._rename_output(op.output_arg_names[i], local_dict[name])
                    else:
                        self.renamed_dict[name] = 1
                        pool_idx = self.get_nearest_var(name, used_idx)
                        used_idx[pool_idx] = 1
                        op._rename_output(op.output_arg_names[i], self.pool_vars[pool_idx])
                        local_dict[name] = self.pool_vars[pool_idx]
        total_size = 0
        for key in self.alloced_vars:
            if self.alloced_vars[key] > 0:
                total_size += self.alloced_vars[key]
                print("var%d: %d" % (key, self.alloced_vars[key]))
        print("total size: %d" % total_size)
        return pool_idx

    def get_total_var_size_before_opt(self):
        total_size = 0
        for key in self.var_size_dict:
            total_size += self.var_size_dict[key]
        return total_size

    def get_total_var_size_after_opt(self):
        total_size = 0
        for key in self.var_size_dict:
            if key not in self.renamed_dict:
                total_size += self.var_size_dict[key]

        for key in self.alloced_vars:
            total_size += self.alloced_vars[key] 

        return total_size

    def set_opt_ops(self, opt_ops):
        self.opt_ops = opt_ops

    def is_partition_op(self, op, op_index, ref_table):
        is_partition = True
        for name in op.output_arg_names:
            if name not in self.checkpoints:
                is_partition = False
            else:
                print(name + " is checkpoint")
            if ref_table[op_index]["output"][name] > 0:
                print(name + " ref count " + str(ref_table[op_index]["output"][name]))
                is_partition = False
            else:
                print(name + " ref count " + str(ref_table[op_index]["output"][name]))
        return is_partition

    def is_opt_op(self, op):
        if op.type in self.opt_ops:
            return True
        return False

    def is_backward_op(self, op):
        is_backward = False
        for name in op.output_arg_names:
            if "@GRAD" in name:
                is_backward = True
        for name in op.input_arg_names:
            if "@GRAD" in name:
                is_backward = True
        if "_grad" in op.type:
            is_backward = True
        if "read" in op.type:
            is_backward = False
        return is_backward

    def volumn(self, var_desc, batch):
        val = reduce(lambda x, y: x * y, var_desc.shape, 1)
        if val < 0:
            return abs(val * batch)
        else:
            return val
        
    def get_all_var_size(self, program, batch):
        all_vars = program.current_block().vars
        ks = all_vars.keys()
        total_size = 0

        for key in ks:
            if "READER" in str(all_vars[key].type):
                continue
            self.var_size_dict[key] = self.volumn(all_vars[key], batch)
            total_size += self.var_size_dict[key]

        print("before optimization: %d" % total_size)

    def _make_ref_table(self, program):
        all_ops = program.current_block().ops
        ref_dict = {}
        op_index = len(all_ops) - 1
        ref_table = {}
        for op in all_ops[::-1]:
            ref_table[op_index] = {"input":{}, "output":{}}
            for name in op.output_arg_names:
                if name in ref_dict:
                    ref_dict[name] += 1
                else:
                    ref_dict[name] = 0
                ref_table[op_index]["output"][name] = ref_dict[name]
            '''
            for name in op.input_arg_names:
                if name in ref_dict:
                    ref_dict[name] += 1
                else:
                    ref_dict[name] = 0
                ref_table[op_index]["input"][name] = ref_dict[name]
            '''
            op_index -= 1
        return ref_table

    def get_forward_partitions2(self, forward_program):
        ref_table = self._make_ref_table(forward_program)
        forward_partitions = []
        fp = []
        all_ops = forward_program.current_block().ops
        for i, op in enumerate(all_ops):
            # get forward partitions, skip optimization op and backward op
            if self.is_opt_op(op):
                continue
            if self.is_backward_op(op):
                continue
            fp.append(op)
            if self.is_partition_op(op, i, ref_table):
                print("is partition op ")
                print(op.output_arg_names)
                forward_partitions.append(fp)
                fp = []
        if fp != []:
            forward_partitions.append(fp)
        for part in forward_partitions[:-1]:
            self.rewrite_ops(part)
        return forward_partitions[:-1]

    def get_backward_partitions2(self, backward_program):
        forward_partitions = []
        backward_partitions = []
        fp = []
        bp = []
        ref_table = self._make_ref_table(backward_program)
        all_ops = backward_program.current_block().ops
        for i, op in enumerate(all_ops):
            if self.is_opt_op(op) or self.is_backward_op(op):
                continue
            fp.append(op)
            if self.is_partition_op(op, i, ref_table):
                forward_partitions.append(fp)
                fp = []
        if fp != []:
            forward_partitions.append(fp)
        ff_partition_idx = len(forward_partitions) - 1
        bp = forward_partitions[ff_partition_idx]
        for i, op in enumerate(all_ops):
            if self.is_opt_op(op):
                continue
            if not self.is_backward_op(op):
                continue
            bp.append(op)
            if self.is_partition_op(op, i, ref_table):
                backward_partitions.append(bp)
                ff_partition_idx -= 1
                bp = forward_partitions[ff_partition_idx]
        if bp != []:
            backward_partitions.append(bp)
        for part in backward_partitions:
            self.rewrite_ops(part)
        return backward_partitions
                    
    def get_forward_partitions(self, forward_program):
        forward_partitions = []
        backward_partitions =[]
        fp = []
        bp = []
        for op in forward_program.current_block().ops:
            if op.type in self.opt_ops:
                continue
            is_backward = False
            for name in op.output_arg_names:
                if "@GRAD" in name:
                    is_backward = True
            for name in op.input_arg_names:
                if "@GRAD" in name:
                    is_backward = True
            if "_grad" in op.type:
                is_backward = True
            if is_backward:
                continue
            is_partition = False
            for name in op.output_arg_names:
                if name in self.checkpoints and op.type == self.checkpoints[name]:
                    is_partition = True
                    break
            fp.append(op)
            if is_partition:
                forward_partitions.append(fp)
                fp = []
        if fp != []:
            forward_partitions.append(fp)
        for part in forward_partitions[:-1]:
            self.rewrite_ops(part)
        return forward_partitions[:-1]

    def get_backward_partitions(self, backward_program):
        forward_partitions = []
        backward_partitions =[]
        fp = []
        bp = []
        forward_op = []
        backward_op = []
        for op in backward_program.current_block().ops:
            if op.type in self.opt_ops:
                continue
            is_backward = False
            for name in op.output_arg_names:
                if "@GRAD" in name:
                    is_backward = True
            for name in op.input_arg_names:
                if "@GRAD" in name:
                    is_backward = True
            if "_grad" in op.type:
                is_backward = True
            if is_backward:
                backward_op.append(op)
                continue
            forward_op.append(op)
            is_partition = False
            for name in op.output_arg_names:
                if name in self.checkpoints and op.type == self.checkpoints[name]:
                    print("*****************: " + name + "*************" + op.type)
                    is_partition = True
            fp.append(op)
            if is_partition:
                forward_partitions.append(fp)
                fp = []

        print("checkpoint num: " + str(len(self.checkpoints)))

        print("%%%")
        for i, fop in enumerate(forward_op):
            print(str(i) + " : " + fop.type)
        
        print("%%%")
        for i, bop in enumerate(backward_op):
            print(str(i) + " : " + bop.type)

        if fp != []:
            forward_partitions.append(fp)        
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@  " + str(len(forward_partitions)))
        ff_partition_idx = len(forward_partitions) - 1
        bp = forward_partitions[ff_partition_idx]
        
        for op in backward_program.current_block().ops:
            if op.type in self.opt_ops:
                continue
            is_backward = False
            for name in op.output_arg_names:
                if "@GRAD" in name:
                    is_backward = True
            for name in op.input_arg_names:
                if "@GRAD" in name:
                    is_backward = True
            if "_grad" in op.type:
                is_backward = True
            if not is_backward:
                continue
            
            is_partition = False
            for name in op.output_arg_names:
                #if name in self.checkpoints and op.type == self.checkpoints[name]:
                if name in self.checkpoints:
                    print("*****************: " + name + "*************" + op.type)
                    is_partition = True
                    
            bp.append(op)
            
            if is_partition:
                backward_partitions.append(bp)
                ff_partition_idx -= 1
                bp = forward_partitions[ff_partition_idx]
        if bp != []:
            backward_partitions.append(bp)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@  " + str(len(backward_partitions)))
        for part in backward_partitions:
            self.rewrite_ops(part)
        return backward_partitions
