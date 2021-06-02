from functionalities import functionalities as func
from Config import Config as conf
import math
from charm.toolbox.integergroup import IntegerGroup
from charm.core.math.integer import *
import binascii
from backports.pbkdf2 import pbkdf2_hmac
import numpy as np
import pickle
import time

class ot_gen_offline:
	def KDF(k):
		salt = binascii.unhexlify('aaef2d3f4d77ac66e9c5a6c3d8f921d1')
		# print(k)
		passwd = "".join(str(k)).encode("utf8")
		key = pbkdf2_hmac("sha256", passwd, salt, 50000, 8)
		key = int.from_bytes(key,"big")
		return key;

	def trip_gen(U,V,flag=0):
		start_time = time.time()
		G = IntegerGroup() # do this in config?
		G.paramgen(1024)
		g = G.randomGen()

		# files for communication
		h_filename = str(conf.partyNum)+"_"+"h.txt"
		v_filename = str(conf.partyNum)+"_"+"v.txt"

		for j in range(conf.t):
			A = np.array(U[j:j+conf.batchsize])
			if(flag != 0):
				A = A.reshape(conf.d,1)
				# print("A.shape: ", A.shape)

			B = np.array(V[:,j])
			B = B.reshape(V.shape[0],1)
			c_0 = np.matmul(A,B) # A0xB0 for S0 and A1xB1 for S1
			print("C_0 shape: ", c_0.shape)
			c_1 = []
			c_2 = []

			# A0 x B1 and A1 x B0 // works only for batchsize = 1
			for i in range(A.shape[0]):
				for i1 in range(A.shape[1]):
					r = np.uint64(np.random.uniform(0, (2**conf.l), (conf.l,))).tolist() # in 2^l?
					# print(r)
					f_r = [np.uint64(A[i][i1]*(math.pow(2,p))+r[p])%(2**conf.l) for p in range(conf.l)]
					b = []
					for o in range(B.shape[0]):
						n = ("{0:b}".format(B[o][0])).rjust(64,'0') # bitdecompostion of B_ij
						b = [int(p) for p in n] # convert the bit string to int list

						# (h_i0,h_i1)
						alpha = [G.random(G.q) for i in range(conf.l)]
						h = []
						for l in range(conf.l):
							beta = G.random(G.q)
							h1 = (g**beta)%(G.p)
							if(b[l] == 0):
								h.append((g**alpha[l],h1))
							else:
								h.append((h1,g**alpha[l]))

						# print(type(h[0][0]))
						# send h to the other party
						func.send_file(h,h_filename,14)

						# process received h
						other_h_filename = "other_" + h_filename
						with open(other_h_filename, "r+") as f:
							w = f.readline()
						res = str(w).strip('][').split(', ')
						s = [p.strip('()') for p in res]
						other_h = [(integer(s[z])%(G.p),integer(s[z+1])%(G.p)) for z in range(0,len(s)-1,2)]


						m = G.random(G.q) # random element from Z_R_q
						u = (g**m)
						
						# send and receive u
						other_u = func.send_val([int(u)])					
						other_u = integer(other_u[0])%(G.p)

						k = []
						v = []
						for l in range(conf.l):
							k.append((other_h[l][0]**m,other_h[l][1]**m))
							kdf_k0 = ot_gen_offline.KDF(k[l][0])
							kdf_k1 = ot_gen_offline.KDF(k[l][1])

							v.append((r[l]^kdf_k0,f_r[l]^kdf_k1))
						
						# send and receive v
						other_v = func.send_file(v,v_filename,13)

						# process received u
						other_v_filename = "other_" + v_filename
						with open(other_v_filename, "r+") as f:
							w = f.readline()
						res = str(w).strip('][').split(', ')
						s = [p.strip('()') for p in res]
						other_v = [(integer(s[z])%(G.p),integer(s[z+1])%(G.p)) for z in range(0,len(s)-1,2)]


						fin_k = []
						x = []
						for l in range(conf.l):
							fin_k.append(other_u**alpha[l])
							kdf_k = ot_gen_offline.KDF(fin_k[l])
							x.append((other_v[l][b[l]])^integer(kdf_k))

					sum1 = 0
					for p in range(conf.l):
						# print(type(x[p]))
						sum1 = (integer(sum1)+integer(x[p]))%(2**64)
						# print(sum1)

					sum2 = 0
					for p in range(conf.l):
						sum2 = (sum2 + ((-1)*r[l]))

				c_1.append(int(sum1))
				c_2.append(int(sum2%(2**64)))

		print("Time taken by program: ",time.time() - start_time)
		c_1 = np.array(c_1).reshape(c_0.shape[0],c_0.shape[1])
		c_2 = np.array(c_2).reshape(c_0.shape[0],c_0.shape[1])
		print("C_1 shape: ", c_1.shape)
		print("C_2 shape: ", c_2.shape)

		C = np.array(np.add(c_0,c_1))
		C = np.add(C,c_2)


		return C