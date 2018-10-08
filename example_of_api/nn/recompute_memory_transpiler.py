import sys
import os

class RecomputeTranspiler(object):
    def __init__(self, memory_pool, checkpoints, weight_dict):
        self.pool_vars = memory_pool
        self.checkpoints = {}
        for cp in checkpoints:
            group = cp.split("$")
            self.checkpoints[group[1]] = group[0]
        print self.checkpoints
        self.weight_dict = weight_dict
        self.max_var_num = 0
        self.renamed_dict = {}

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
        for i in range(self.max_var_num):
            fout.write(self.print_var_by_id(i))
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
        print "optimized program called"

    def get_max_var_num(self):
        return self.max_var_num
        
    def rewrite_ops(self, ops):
        pool_idx = 0
        local_dict = {}
        for op in ops:
            for i, name in enumerate(op.input_arg_names):
                if name not in self.checkpoints and name not in self.weight_dict:
                    if name in local_dict:
                        op.rename_input(op.input_arg_names[i], local_dict[name])
                    else:
                        self.renamed_dict[name] = 1
                        op.rename_input(op.input_arg_names[i], self.pool_vars[pool_idx])
                        local_dict[name] = self.pool_vars[pool_idx]
                        pool_idx += 1
            for i, name in enumerate(op.output_arg_names):
                if name not in self.checkpoints and name not in self.weight_dict:
                    if name in local_dict:
                        op.rename_output(op.output_arg_names[i], local_dict[name])
                    else:
                        self.renamed_dict[name] = 1
                        op.rename_output(op.output_arg_names[i], self.pool_vars[pool_idx])
                        local_dict[name] = self.pool_vars[pool_idx]
                        pool_idx += 1
        if pool_idx > self.max_var_num:
            self.max_var_num = pool_idx + 1
        return pool_idx

    def set_opt_ops(self, opt_ops):
        self.opt_ops = opt_ops
    
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
                continue
            is_partition = False
            for name in op.output_arg_names:
                if name in self.checkpoints and op.type == self.checkpoints[name]:
                    is_partition = True
            fp.append(op)
            if is_partition:
                forward_partitions.append(fp)
                fp = []
                
        if fp != []:
            forward_partitions.append(fp)        
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
                if name in self.checkpoints and op.type == self.checkpoints[name]:
                    is_partition = True
                    
            bp.append(op)
            
            if is_partition:
                backward_partitions.append(bp)
                ff_partition_idx -= 1
                bp = forward_partitions[ff_partition_idx]
        if bp != []:
            backward_partitions.append(bp)
        for part in backward_partitions:
            self.rewrite_ops(part)
        return backward_partitions
