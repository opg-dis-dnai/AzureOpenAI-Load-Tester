from util import promptLooper


loop = ["a", "b", "c"]

gen = promptLooper(loop)

for i in range(10):
    print(next(gen))