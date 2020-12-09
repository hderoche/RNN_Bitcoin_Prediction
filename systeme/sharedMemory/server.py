from multiprocessing import shared_memory

sharedMem = shared_memory.SharedMemory(create=True, size= 10)
buffer = sharedMem.buf
buffer[:6] = bytearray([1, 2, 8, 3, 2, 2])

sharedMem2 = shared_memory.SharedMemory(sharedMem.name)
print(bytes(sharedMem2.buf).decode())

sharedMem.close()
sharedMem2.close()
sharedMem.unlink()
