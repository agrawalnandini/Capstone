from functionalities import functionalities as func
import numpy as np
import phe as paillier
from Config import Config as conf
import math
import sys

class offline:

	#functions to encrypt and decrypt in the paillier cryptosystem using the public and private key respectively
	def encrypt_vector(public_key, x):
		enc_x = []
		for i in range(len(x)):
			enc_x.append(public_key.encrypt(x[i][0]))
		return np.array(enc_x)

	def decrypt_vector(private_key, x):
		dec_x = []
		for i in range(len(x)):
			dec_x.append(private_key.decrypt(x[i][0]))
		return np.array(dec_x)

	#Linear Homomorphic Encryption for Beaver's Precomputed Triplets generation
	def lhe(U,V, flag=0):
		C=[]
		#generate public key private key using the paillier cryptosystem
		pubkey,privkey = paillier.generate_paillier_keypair()
		
		pubkeyOther = func.reconstruct([pubkey]) #getting the other party's public key
		pubkeyOther=pubkeyOther[0]
		for j in range(conf.t): 
			A = np.array(U[j:j+conf.batchsize])
			if(flag != 0):
				A = A.reshape(conf.d,1)

			B = np.array(V[:,j])
			B = B.reshape(V.shape[0],1)
			
			c_0 = np.matmul(A,B) #A0 X B0
			encrypted_B = offline.encrypt_vector(pubkey, B) #S1 encrypts B1 for A0B1 and S0 encrypts B0 A1B0
		
			other_B = np.array(func.reconstruct(encrypted_B.tolist()))
			other_B = other_B.reshape(V.shape[0],1) #B is d*1 or 1*1 for Vdash
			if (flag == 0):
				c_1=0
				for i in range(V.shape[0]): 
					#internally multiplication happens using exponentiation and addition happens using multiplication in paillier cryptosystem
					c_1 = c_1 + other_B[i][0]*A[0][i] 
				c_1 = np.array(c_1)
				c_1= c_1.reshape(conf.batchsize,1)		
			else :
				c_1 = []
				for i in range(A.shape[0]):
					temp = other_B[0][0] * A[i][0]
					c_1.append([temp])
				c_1 = np.array(c_1)
				c_1 = c_1.reshape(A.shape[0],1) 
			
			random_num = np.array(np.random.random(size=(c_1.shape[0],1))) #random number to mask the product
			encrypted_random = offline.encrypt_vector(pubkeyOther,random_num)
			encrypted_random= encrypted_random.reshape(c_1.shape[0],1)
			c_1 = np.add(c_1,encrypted_random)
			recv = np.array(func.reconstruct(c_1.tolist()))
			recv=recv.reshape(c_1.shape[0],1)
			recv=offline.decrypt_vector(privkey,recv)
			recv=recv.reshape(c_1.shape[0],1)
			random_num = np.multiply(-1,random_num) 
			term = np.add(c_0,recv)
			term = np.add(term,random_num)
			term = term.reshape(term.shape[0],)
			C.append(term.tolist()) #A0B0 + A0B1 + A1BO for S0, A1B1+A0B1+A1B0 for S1
		
		C = np.array(C)
		if flag==0:
			C = C.reshape(c_1.shape[0],conf.t)
		else:
			C = np.transpose(C)
		return C