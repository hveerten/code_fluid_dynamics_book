size_double = 8 # size of double in bytes
no_cons = 4
no_prim = 4

ghosts = 2
res_x = 1000
res_y = 1000

gres_x = 2 * ghosts + res_x
gres_y = 2 * ghosts + res_y

memory = gres_x * gres_y * (no_cons + no_prim + no_cons + no_cons) * size_double
memory_KB = memory / 1024.
memory_MB = memory_KB / 1024.
memory_GB = memory_MB / 1024.

print ("size in memory: %i" % memory)
print ("size in memory: %f KB" % memory_KB)
print ("size in memory: %f GB" % memory_GB)

