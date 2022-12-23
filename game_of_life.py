import pyopencl as cl
import numpy

import random

from time import time

WIDTH = 4096*2
HEIGHT = 2160*2
ITERATIONS = 100

kernelsource = f"""
uint index(uint x, uint y, uint w, uint h){{
    x = x % w;
    y = y % h;
    return y * w + x;
}}
__kernel void iterate(
    __constant uchar *currState,
    __global uchar *nextState){{
	uint x = get_global_id(0);
	uint y = get_global_id(1);
    uint w = get_global_size(0);
    uint h = get_global_size(1);

    uchar s = currState[index(x, y, w, h)];
    uchar c = !!currState[index(x-1, y-1, w, h)] + !!currState[index(x, y-1, w, h)] + !!currState[index(x+1, y-1, w, h)];
    c += !!currState[index(x-1, y, w, h)] + !!s + !!currState[index(x+1, y, w, h)];
    c += !!currState[index(x-1, y+1, w, h)] + !!currState[index(x, y+1, w, h)] + !!currState[index(x+1, y+1, w, h)];
    nextState[index(x, y, w, h)] = (uchar)(c==3) * (s + 1) + (uchar)(c == 4) * (s + !!s);
//    if (c == 3) {{
//        nextState[index(x, y, w, h)] = s + 1;
//    }}else if (c == 4) {{
//        nextState[index(x, y, w, h)] = s + !!s;
//    }}else {{
//        nextState[index(x, y, w, h)] = 0;
//    }}
}}
"""

context = cl.create_some_context()
queue = cl.CommandQueue(context)

program = cl.Program(context, kernelsource).build()

h_a = numpy.random.randint(2, size=WIDTH*HEIGHT, dtype=numpy.uint8)

h_b = numpy.empty(WIDTH*HEIGHT, dtype=numpy.uint32)

d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
rtime = time()

iter = program.iterate
iter.set_scalar_arg_dtypes([None, None])

for _ in range(int(ITERATIONS/2)):
    iter(queue, (WIDTH, HEIGHT), None, d_a, d_b)
    iter(queue, (WIDTH, HEIGHT), None, d_b, d_a)

queue.finish()
rtime = time() - rtime

print(f"Kernels completed in {rtime}s.")

rtime = time()
cl.enqueue_copy(queue, h_a, d_a)
queue.finish()
rtime = time() - rtime
print(f"Unload from accelerator completed in {rtime}s.")

cl.enqueue_copy(queue, h_a, d_b)
queue.finish()

#h_a = numpy.max(h_a) - h_a # Invert for black cells on white field
h_a = h_a - numpy.min(h_a) # Normalize (and set max in PGM below)

f = open("output.pgm", "w")
f.write(f"P5\n{WIDTH} {HEIGHT}\n{numpy.max(h_a)}\n")
h_a.tofile(f)

f.close()
