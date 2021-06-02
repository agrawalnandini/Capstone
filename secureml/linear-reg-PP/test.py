##############################
# Questions to ask Professor:
# 1. correctness of matrixmulreg?
# 2. dealing with negative shares in truncation: int64tofloat vs truncation (conditional on partyNum vs +ve/-ve share value)
# 3. In matrixmulreg, should we "truncate" every after multiplication? : 
# since multiplying two numbers in the left-shifted form makes their product 
# a twice left-shifted value in the ring.
##############################

from functionalities import functionalities as func
from Config import Config as conf
import numpy as np

def test():
	if(conf.partyNum == 0):
		A = np.array([1,2,3])
		B = np.array([1,1.4,1.7])
		A = func.floattoint64(A)
		B = func.floattoint64(B)
		U = np.array([3,4,5])
		U = func.floattoint64(U)
		V = np.array([2,1,6])
		V = func.floattoint64(V)
		Z = func.floattoint64(np.array([50]))
		B= B.reshape(3,1)
		V= V.reshape(3,1)

	else: 
		A = np.array([1,2,3])
		B = np.array([1,1.4,1.7])
		A = func.floattoint64(A)
		B = func.floattoint64(B)
		U = np.array([2,1,6])
		U = func.floattoint64(U)
		V = np.array([3,4,5])
		V = func.floattoint64(V)
		Z = func.floattoint64(np.array([121]))
		B= B.reshape(3,1)
		V= V.reshape(3,1)

	# E = np.subtract(A,U)
	# F = np.subtract(B,V)

	E = np.array([-3,-1,-5])
	F = np.array([[-3],[-2.2],[-7.6]])
	E = func.floattoint64(E)
	F = func.floattoint64(F)
	mul1 = np.matmul(E,F)
	mul1 = np.uint64(func.int64tofloat(mul1))
	# mul1 = func.truncate(mul1,(1<<13))
	mul2 = np.matmul(A,F)
	mul2 = np.uint64(func.int64tofloat(mul2))
	print("mul2: ",mul2)
	print("mul2 in float: ",func.int64tofloat(mul2))
	# mul2 = func.truncate(mul2,(1<<13))
	mul3 = np.matmul(E,B)
	mul3 = np.uint64(func.int64tofloat(mul3))
	# mul3 = func.truncate(mul3,(1<<13))
	mul0 = np.multiply(func.floattoint64(-1 * conf.partyNum), mul1)
	mul0 = np.uint64(func.int64tofloat(mul0))
	# mul0 = func.truncate(mul0,(1<<13))

	y = np.add(mul0,mul2)
	y = np.add(y,mul3)
	y = np.add(y,Z)
	# y = np.uint64(func.int64tofloat(y))
	# y = func.truncate(y,(1<<13))
	yhat = func.int64tofloat(y)
	print(yhat)

conf.partyNum = 0
print("partyNum: ", conf.partyNum)
test()
conf.partyNum = 1
print("partyNum: ", conf.partyNum)
test()