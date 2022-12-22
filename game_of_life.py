import pyopencl as cl
import numpy

import random

from time import time

WIDTH = 1024
HEIGHT = 1024
# Actually runs two iterations per count so this 200. This improves contrast (PPM max is still set at 100).
ITERATIONS = 100

kernelsource = f"""
unsigned int index(unsigned int x, unsigned int y, unsigned int w, unsigned int h){{
//    if (x == -1) x = w-1;
//    if (y == -1) y = h-1;
    x = x % w;
    y = y % h;
    return y * h + x;
}}
__kernel void iterate(
    __global unsigned int *currState,
    __global unsigned int *nextState){{
	int x = get_global_id(0);
	int y = get_global_id(1);
    int w = get_global_size(0);
    int h = get_global_size(1);

    int i = x + y*h;
    unsigned int s = currState[i];
    unsigned int c = !!currState[index(x-1, y-1, w, h)] + !!currState[index(x, y-1, w, h)] + !!currState[index(x+1, y-1, w, h)];
    c += !!currState[index(x-1, y, w, h)] + !!s + !!currState[index(x+1, y, w, h)];
    c += !!currState[index(x-1, y+1, w, h)] + !!currState[index(x, y+1, w, h)] + !!currState[index(x+1, y+1, w, h)];
    if (c == 3) {{
        nextState[index(x, y, w, h)] = s + 1;
    }}else if (c == 4) {{
        nextState[index(x, y, w, h)] = s + !!s;
    }}else {{
        nextState[index(x, y, w, h)] = 0;
    }}
}}

"""

context = cl.create_some_context()
queue = cl.CommandQueue(context)

program = cl.Program(context, kernelsource).build()

h_a = numpy.random.randint(2, size=WIDTH*HEIGHT, dtype=numpy.uint32)
#h_a = numpy.zeros(WIDTH*HEIGHT, dtype=numpy.uint32)
# Glider
# h_a[0] = 1
# h_a[1] = 1
# h_a[2] = 1
# h_a[2 + WIDTH] = 1
# h_a[1 + 2 * WIDTH] = 1
for x in range(WIDTH):
	for y in range(HEIGHT):
		print(f" {h_a[y*WIDTH + x]}", end = '')
	print()
h_b = numpy.empty(WIDTH*HEIGHT, dtype=numpy.uint32)

d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
rtime = time()

iter = program.iterate
iter.set_scalar_arg_dtypes([None, None])

for _ in range(ITERATIONS):
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

for x in range(WIDTH):
	for y in range(HEIGHT):
		print(f" {h_a[y*WIDTH + x]}", end = '')
	print()
cl.enqueue_copy(queue, h_a, d_b)
queue.finish()
for x in range(WIDTH):
	for y in range(HEIGHT):
		print(f" {h_a[y*WIDTH + x]}", end = '')
	print()
	
f = open("output.ppm", "w")
# Change ITERATIONS to ITERATIONS * 2 lower constract but technically correct...
f.write(f"P3\n{WIDTH} {HEIGHT}\n{ITERATIONS}\n")
for x in range(WIDTH):
	for y in range(HEIGHT):
		f.write(f"{ITERATIONS-h_a[y*WIDTH + x]} {ITERATIONS-h_a[y*WIDTH + x]} {ITERATIONS-h_a[y*WIDTH + x]}\n")
f.close()
