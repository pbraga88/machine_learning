#!/usr/bin/python3

def test(*x):
    *x += 1
    print(x)
x = 0
test(&x)
print(x)
