from Config import Config as conf
import sys
from functionalities import functionalities as func
from linearReg import linearReg as linearReg
from offline import offline as off
import numpy as np
from testmatmul import testmatmul as test
from ot_gen_offline import ot_gen_offline as ot

def main():
	conf.partyNum = int(sys.argv[1])
	if conf.partyNum == 0:
		conf.PORT = 8004
		conf.advPORT = 8005
		filename_data='data0.txt'
	else: 
		conf.PORT = 8005
		conf.advPORT = 8004
		filename_data='data1.txt'	

	print("============================================================")
	print("====================SERVER "+str(conf.partyNum)+"================================")
	print("============================================================")

	print("\nReading input data from",filename_data)
	X,Y = linearReg.readData(filename_data,filename_mask)

	
	# U = np.uint64(np.random.uniform(0, (2**conf.l), (conf.n,conf.d)))
	# V = np.uint64(np.random.uniform(0, (2**conf.l), (conf.d,conf.t)))
	U = np.random.rand(conf.n, conf.d)
	V = np.random.rand(conf.d,conf.t)
	print("\nGENERATING BEAVER'S TRIPLETS LHE...")

	Vdash = np.random.rand(conf.batchsize,conf.t)

	# print("My U: ",U)
	# print("My Vdash: ",Vdash)

	# #######Reconstruct to check#########
	# U2 = np.array(func.reconstruct(U.tolist()))
	# U2 = U2.reshape(conf.n,conf.d)
	# u = np.add(U,U2)

	# V2 = np.array(func.reconstruct(V.tolist()))
	# V2 = V2.reshape(conf.d,conf.t)
	# v = np.add(V,V2)	

	# Vdash2 = np.array(func.reconstruct(Vdash.tolist()))
	# Vdash2 = Vdash2.reshape(conf.batchsize,conf.t)
	# vdash = np.add(Vdash,Vdash2)

	# print('U: ',u)
	# print('V:', v)
	# print('Vdash: ',vdash)

	# #######Actual values of Z and Zdash########
	# z = np.zeros((1,conf.t))
	# zdash = np.zeros((conf.d,conf.t))
	# for i in range(len(u)):
	# 	z[:,i]= np.array((np.matmul(u[i],v[:,i])))
	# print("Actual mult z: ", z)
	
	# for i in range(len(u)):
	# 	u_row_tranpose = np.transpose(np.matrix(u[i]))
	# 	zdash[:,i]=np.array(np.matmul(u_row_tranpose,vdash[:,i]))
	
	# print("Actual mult zdash: ", zdash)

	# OT
	# U = func.floattoint64(U)
	# V = func.floattoint64(V)

	######LHE generation of Z and Zdash shares#####
	flag = 0 # generate Z
	Z=off.lhe(np.array(U),np.array(V),flag)
	# Z = ot.trip_gen(np.array(U),np.array(V),flag)
	# # print('my Z: ',Z)

	flag = 1 # generate Zdash
	Zdash=off.lhe(np.array(U),np.array(Vdash),flag)
	# Z = ot.trip_gen(np.array(U),np.array(V),flag)
	
	# #######RECONSTRUCT Z AND ZDASH TO VERIFY########
	# Z2 = np.array(func.reconstruct(Z.tolist()))
	# Z2 = Z2.reshape(1,conf.t)
	# Z_f = np.add(Z,Z2)

	# print("reconstructed Z: ", Z_f)

	# Zdash2 = np.array(func.reconstruct(Zdash.tolist()))
	# Zdash2 = Zdash2.reshape(conf.d,conf.t)
	# Zdash_f = np.add(Zdash,Zdash2)

	# # print("reconstructed Zdash: ", Zdash_f)
	# # print(Zdash)
	print("\nTRIPTLETS GENERATED!")
	# ###### PREPARING MASKS FOR LINEAR REG #######
	U = func.floattoint64(U)
	V = func.floattoint64(V)
	Vdash = func.floattoint64(Vdash)
	Z = func.floattoint64(Z)
	Zdash = func.floattoint64(Zdash)
	
	model = linearReg.SGDLinear(X,Y,U,V,Vdash,Z,Zdash)
	model = model.tolist()
	print("\n\nFINAL (RECONSTRUCTED) MODEL: ")
	
	for i in range(conf.d):
		model[i] = func.int64tofloat(model[i][0])
	print(model)

if __name__ == '__main__':
	main()