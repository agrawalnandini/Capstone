from Config import Config as conf
import sys
import itertools
import numpy as np
from functionalities import functionalities as func
import random
import math

class linearReg:

	def readData(filename_data,filename_mask):
		# X n*d matrix
		# Y n*1 matrix
		# w d*1 matrix
		# U n*d matrix
		# V d*t matrix
		# V' 1*t matrix mask the difference Y - Y^
		# Z 1*n matrix
		# Z'd*t matrix (U_b transpose * V'[i]) 

		mask=[]
		X=[]    
		Y=[]
		
		i=0
		with open(filename_data,'r+') as f:
			for line in f:
				row=line.split()
				i=i+1
				if(i<=6):
					r = []
					r.append(func.floattoint64(0.5))
					for j in row:
						r.append(np.uint64(j))
					X.append(r)
				else:
					Y.append(np.uint64(row[0].rstrip()))
			f.close()
		

		conf.n = len(Y)
		conf.d = len(X[0])
		conf.t = conf.n		
		
		return X,Y

	def SGDLinear(X,Y,U,V,VDash,Z,ZDash):
		print("Training the model...")
		X = (np.array(X, dtype = np.uint64))
		Y = (np.array(Y, dtype = np.uint64))
		U = np.array(U, dtype = np.uint64)
		V = np.array(V, dtype = np.uint64)
		VDash = np.array(VDash, dtype = np.uint64)
		Z = np.array(Z, dtype = np.uint64)
		ZDash = np.array(ZDash, dtype = np.uint64)

		E1 = np.uint64(np.subtract(X,U))
		E2 = np.uint64(func.reconstruct(E1.tolist()))
		E = np.uint64(np.add(E1,np.array(E2, dtype = np.uint64)))
		
		# randomly initialise weights vector
		weights = np.array((np.random.rand(conf.d)))
		weights = weights.reshape(conf.d,1)
		weights2 = func.reconstruct(weights)

		wts = weights+weights2
		print("Initial (reconstructed) weights: ")
		print(wts)

		weights = np.array(func.floattoint64(weights), dtype = np.uint64)
		# print(weights)

		for e in range(conf.epochs):
			print("EPOCH:", e+1, end="    ")
			loss = 0.0

			for j in range(conf.t): 
				X_B = np.array(X[j:j+conf.batchsize], dtype = np.uint64)
				Y_B = np.array([Y[j:j+conf.batchsize]], dtype = np.uint64).transpose()

				xb2 = func.reconstruct(X_B.tolist())
				xb2 = np.array(xb2,dtype = np.uint64)
				xb = np.add(X_B, np.array(xb2))

				E_B = np.array(E[j:j+conf.batchsize], dtype = np.uint64)
				V_j = np.array([V[:,j]], dtype = np.uint64).transpose()	
				Z_j = np.array([Z[:,j]], dtype = np.uint64).transpose()  	
				Vdash_j = np.array([VDash[:,j]], dtype = np.uint64).transpose()
				Zdash_j = np.array([ZDash[:,j]], dtype = np.uint64).transpose()

				F1 = (np.subtract(weights,V_j))
				F2 = np.array(func.reconstruct(F1.tolist()), dtype = np.uint64)
				F = np.array(np.add(F1,F2), dtype = np.uint64) 

				YB_dash = func.matrixmul_reg(X_B,weights,E_B,F,V_j,Z_j)

				D_B = (np.subtract(YB_dash,Y_B))

				# computing loss
				yb2 = func.reconstruct(Y_B.tolist())
				y = (np.add(Y_B,np.array(yb2, dtype = np.uint64)))
				y = func.int64tofloat(y[0][0])
				
				#################################################### Computing loss ###################################################################
			
				ybdash2 = func.reconstruct(YB_dash.tolist())
				y_hat = (np.add(YB_dash,np.array(ybdash2, dtype = np.uint64)))				
				y_hat = (func.int64tofloat(y_hat[0][0]))
				
				dif = (y_hat - y)
				loss = loss+(dif*dif)

				#######################################################################################################################################

				Fdash_1 = (np.subtract(D_B,Vdash_j))
				Fdash_2 = func.reconstruct(Fdash_1)
				FDash = (np.add(Fdash_1,np.array(Fdash_2, dtype = np.uint64)))

				X_BT = np.array(X_B, dtype = np.uint64).transpose() 
				E_BT = np.array(E_B, dtype = np.uint64).transpose()

				Del_J = func.matrixmul_reg(X_BT,D_B,E_BT,FDash,Vdash_j,Zdash_j) # the partial differentiation of the loss function output - dx1
				
				red = func.floattoint64(conf.alpha)
				for i in range(conf.d):
					Del_J[i][0] = np.multiply(Del_J[i][0],red)
					Del_J[i][0] = np.uint64(func.int64tofloat(Del_J[i][0]))

				weights = ((np.subtract(np.array(weights, dtype = np.uint64),Del_J)))
			
			if e == 0: 
				print("Initial Loss: ",end=" ")
				print(float(loss)/(conf.n))
			elif e==conf.epochs-1:
				print("Final Loss: ",end=" ")
				print(float(loss)/(conf.n))
			else:
				print("Loss: ", float(loss)/(conf.n))

		print("\nTraining complete.")
		print("\nMy share of the final model: ")
		print(weights)
		################# Reconstructed final weights #############################################
		
		weights2 = func.reconstruct(weights.tolist())
		model = (np.add(np.array(weights2, dtype = np.uint64),np.array(weights, dtype = np.uint64)))

		###########################################################################################
		
		return model