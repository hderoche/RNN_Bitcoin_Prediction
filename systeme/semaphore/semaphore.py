import threading

isParent = '[parent] '
isChild = '[child] '
sem = threading.Semaphore(2)
print(isParent, 'Wait for Child to Print')
sem.acquire()
print(isChild, 'Hello from Child')
print(isChild, 'I am done! Release Semaphore')
sem.release()
print(isParent, 'Child Printed')