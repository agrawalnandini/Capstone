from functionalities import functionalities as func
import numpy as np
from Config import Config as conf

n = 6
d = 3

class testmatmul:

	def triplets(A,B):
		if(conf.partyNum==0):
			U = np.array([3,4,5])
			U = U.reshape(1,d)
			V = np.array([2,1,6])
			V = V.reshape(d,1)
			U = func.floattoint64(U)
			V = func.floattoint64(V)
			Z = np.array([50])
			Z = Z.reshape(1,1)
			Z = func.floattoint64(Z)

		else: 
			U = np.array([2,1,6])
			U = U.reshape(1,d)
			V = np.array([3,4,5])
			V = V.reshape(d,1)
			U = func.floattoint64(U)
			V = func.floattoint64(V)
			Z = np.array([121])
			Z = Z.reshape(1,1)
			Z = func.floattoint64(Z)
		
		E = np.array(np.subtract(A,U),dtype=np.uint64)
		F = np.array(np.subtract(B,V),dtype=np.uint64)
		# print("E:")
		# print(F)
		return E,F,V,Z

	def test():
		A = np.array([1, 2, 3])
		B = np.array([1.0, 1.4, 1.7])
		A = A.reshape(1,d)
		B = B.reshape(d,1)
		A = func.floattoint64(A)
		B = func.floattoint64(B)
		print(A)
		print(B)

		E,F,V,Z = testmatmul.triplets(A,B)

		y_hat = func.matrixmul_reg(A,B,E,F,V,Z)
		# print(y_hat.shape)
		print("multiplication: ", (y_hat[0]))