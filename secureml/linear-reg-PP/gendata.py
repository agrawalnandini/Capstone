import random
import numpy as np
from Config import Config as conf
from functionalities import functionalities as func
from sklearn.datasets import load_boston

n = 6
d = 2
t = n
batchsize=1

def split_shares(x,p,q):
	x_1 = np.uint64(np.random.uniform(0, (2**conf.l), (p,q)))
	x_2 = np.array(np.subtract(x,x_1), dtype = np.uint64)
	return np.array(x_1, dtype = np.uint64),x_2

def check_uvz():
	mask0 = []
	mask1 = []
	with open("mask0.txt",'r') as f1:
		for line in f1:
			row=line.split()
			row=[int(i, base=10) for i in row]
			mask0.append(row)
		f1.close()

	U0 = np.array(mask0[:n], dtype = np.uint64)
	V0 = np.array(mask0[n:n+d+1], dtype = np.uint64)
	Vdash0 = np.array(mask0[n+d+1: n+d+2], dtype = np.uint64)
	Z0 = np.array(mask0[n+d+2:n+d+3], dtype = np.uint64)
	Zdash0 = np.array(mask0[n+d+3:], dtype = np.uint64)

	with open("mask1.txt",'r') as f2:
		for line in f2:
			row=line.split()
			row=[int(i, base=10) for i in row]
			mask1.append(row)
		f2.close()

	U1 = np.array(mask1[:n], dtype = np.uint64)
	V1 = np.array(mask1[n:n+d+1], dtype = np.uint64)
	Vdash1 = np.array(mask1[n+d+1: n+d+2], dtype = np.uint64)
	Z1 = np.array(mask1[n+d+2:n+d+3], dtype = np.uint64)
	Zdash1 = np.array(mask1[n+d+3:], dtype = np.uint64)

	# add shares
	U = np.add(U0,U1)
	V = np.add(V0,V1)
	z = np.add(Z0,Z1)

	Z = np.zeros((1,t),dtype=np.uint64)
	for i in range(len(U)):
		Z[:,i]= np.array((np.matmul(U[i],V[:,i])), dtype = np.uint64) #multiplying a row of u with a column of v
	
	cmp = (z == Z)
	
def saveData(serverNum,u,v,v_dash=np.array(None),z=np.array(None),z_dash=np.array(None)): #xy

	if serverNum!= 0 and serverNum!= 1:
		datafile='data.txt'
		maskfile='mask.txt'
	else:
		datafile = 'data' + str(serverNum) + '.txt' 
		maskfile = 'mask' + str(serverNum) + '.txt'

	if(v_dash.any()==None and z.any()==None and z_dash.any()==None):
		with open(datafile,'w+') as df:
			np.savetxt(df, u, delimiter=' ',fmt='%d') #x
			np.savetxt(df, v, delimiter='\n',fmt='%d') #y

	else:
		with open(maskfile,'w+') as f:
			np.savetxt(f,u,delimiter=' ',fmt='%d')
			np.savetxt(f,v,delimiter=' ',fmt='%d')
			np.savetxt(f,v_dash,delimiter=' ',fmt='%d')
			np.savetxt(f,z,delimiter=' ',fmt='%d')
			np.savetxt(f,z_dash,delimiter=' ',fmt='%d')

def check_shares(x1,x2,X):
	sx = (np.add(x1,x2))

def main():
	
	print("====================================================")
	print("====================DATA OWNER======================")
	print("====================================================")
	print("\nGenerating additive secret shares of data...")
	X = [[4,1],[2,8],[1,0],[3,2],[1,4],[6,7]]
	X = np.array(X, dtype = np.uint64)
	X = func.floattoint64(X)
	Y = [2,-14,1,-1,-7,-8]
	Y = np.array(Y, dtype = np.uint64)
	
	Y = func.floattoint64(Y)
	Y = Y.reshape(len(Y),1)
	
	X_1,X_2 = split_shares(X,len(Y),d)
	
	Y_1,Y_2 = split_shares(Y,len(Y),1)
	
	check_shares(Y_1,Y_2,Y)
	
	saveData(-1,X,Y) 
	saveData(0,X_1,Y_1)
	saveData(1,X_2,Y_2)
	check_uvz()
	print("Input shares for server 0 stored in data0.txt.")
	print("\nInput shares for server 1 stored in data1.txt.")
	

if __name__ == '__main__':
	main()


#not needed since LHE/OT
	# u,v,v_dash,z,z_dash = generatedata() 
	# u_1,u_2 = split_shares(u,n,d+1)
	# v_1,v_2 = split_shares(v,d+1,t)
	# vdash_1,vdash_2 = split_shares(v_dash,batchsize,t)
	# z_1,z_2 = split_shares(z,1,t)
	# zdash_1,zdash_2 = split_shares(z_dash,d+1,t) 
	# saveData(-1,u,v,v_dash,z,z_dash) 
	# saveData(0,u_1,v_1,vdash_1,z_1,zdash_1) 
	# saveData(1,u_2,v_2,vdash_2,z_2,zdash_2) 

# def generatedata():
	
# 	z=[]
# 	z_dash=[] 
# 	u = np.array(np.random.randint(low = 0, high = 2**(conf.l-1),size = (n,d+1)), dtype = np.uint64)
# 	v = np.array(np.random.randint(low = 0, high = 2**(conf.l-1),size = (d+1,t)), dtype = np.uint64)
# 	v_dash = np.array(np.random.randint(low = 2**(conf.l-1),size = (batchsize,t)), dtype = np.uint64)

# 	z = np.zeros((1,t),dtype=np.uint64)
# 	z_dash= np.zeros((d+1,t),dtype=np.uint64)
   
# 	for i in range(len(u)):
# 		z[:,i]= np.array((np.matmul(u[i],v[:,i])), dtype = np.uint64) 
# 		u_row_tranpose = np.transpose(np.matrix(u[i]))
# 		z_dash[:,i]=np.array(np.matmul(u_row_tranpose,v_dash[:,i]), dtype = np.uint64)  
	
# 	return u,v,v_dash,z,z_dash 