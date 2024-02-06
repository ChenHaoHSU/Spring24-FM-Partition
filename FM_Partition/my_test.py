import os
from student_impl.eid_ch48458 import FM_Partition
eid = "ch48458"
benchmark_path = "benchmarks/example_1.txt"
output_root = "output"
output_root = os.path.join(output_root, eid)
if not os.path.isdir(output_root):
    os.mkdir(output_root)

output_path = os.path.join(output_root, os.path.basename(benchmark_path))
solver = FM_Partition()
solver.read_graph(benchmark_path)
solution = solver.solve()
profiling = (0, 0) # ignore runtime and memory for now
solver.dump_output_file(*solution, *profiling, output_path)
   
