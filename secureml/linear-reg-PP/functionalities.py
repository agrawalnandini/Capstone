#!/usr/bin/python
import sys
from Config import Config as conf
import socket
import pickle 
import random
import numpy as np
import math
import tqdm
import os

class functionalities:
	def floattoint64(x):
		#maps values to the uint64 world
		x = np.array(conf.converttoint64*(x), dtype = np.uint64)
		return x

	def int64tofloat(x,scale=1<<conf.precision):
		#maps values from the uint64 world back to its original form
		y=0
		if(x > (2**(conf.l-1))-1):
			x = (2**conf.l) - x
			y = np.uint64(x)
			y = y*(-1)
		else:
			y = np.uint64(x)
			
		return float(y)/(scale)

	def send_file(file_info,filename,sz):
		with open(filename,"w+") as f:
			f.write("".join(str(file_info)))
		filesize = os.path.getsize(filename)
		SEPARATOR = "--"
		BUFFER_SIZE = 4096

		if(conf.partyNum == 0):

			ssock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			ssock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			ssock.bind((conf.IP, conf.PORT))
			ssock.listen()
			while True:
				try:
					client, addr = ssock.accept()
					break
				except:
					continue
	
			client.send(f"{filename}{SEPARATOR}{filesize}".encode())
		
			with open(filename, "rb") as f:
				bytes_read = f.read(filesize)		
				client.sendall(bytes_read)

			received = client.recv(sz).decode()
			fname, fsize = received.split(SEPARATOR)
			b = math.ceil(int(fsize)/BUFFER_SIZE)
	
			with open(str("other_")+filename, "wb") as f:
				bytes_read = client.recv(int(fsize))
				f.write(bytes_read)
				f.flush()	
		
			client.close()
			ssock.close()

		else:
			csock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			while True:
				try:
					csock.connect((conf.advIP,conf.advPORT))
					break
				except: 
					continue
			
			received = csock.recv(sz).decode()
			
			fname, fsize = received.split(SEPARATOR)
			b = math.ceil(int(fsize)/BUFFER_SIZE)
			
			with open(str("other_")+filename, "wb") as f:
				bytes_read = csock.recv(int(fsize))
				f.write(bytes_read)
				f.flush()						
			

			csock.send(f"{filename}{SEPARATOR}{filesize}".encode())
			with open(filename, "rb") as f:
				bytes_read = f.read(filesize)			
				csock.sendall(bytes_read)
			csock.close()

		return

	def send_val(send_info):
		#exchange data between two servers
		#Party 0 - Server , Party 1 - Client
		if(conf.partyNum == 0):
			ssock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			ssock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			ssock.bind((conf.IP, conf.PORT))
			ssock.listen()
			while True:
				try:
					client, addr = ssock.accept()
					break
				except:
					continue
			recv_info = client.recv(4096)
			recv_info = pickle.loads(recv_info)
			client.send(pickle.dumps(send_info))
			client.close()
			ssock.close()
		else: 
			csock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			while True:
				try:
					csock.connect((conf.advIP,conf.advPORT))
					break
				except: 
					continue
			csock.send(pickle.dumps(send_info))
			recv_info = pickle.loads(csock.recv(4096))
			csock.close()
		return recv_info

	def addshares(a, b, mask):
		sendlist = []
		sum1 = (a + b + mask) 
		sendlist.append(sum1)
		sum2 = send_val(sendlist)
		
		return sum1+sum2[0]

	def reconstruct(c):
		C = functionalities.send_val(c)
		return C

	def multiplyshares(a,b,u,v,z):
		sendlist = []
		e = a - u
		f = b - v
		sendlist.append(e)
		sendlist.append(f)
		recv_info = send_val(sendlist)
		E = e + recv_info[0]
		F = f + recv_info[1]
		c = (-1 * conf.partyNum * E * F) + (a * F) + (E * b) + z
		sendlist=[]
		sendlist.append(c)
		C = reconstruct(sendlist)
		return C[0]+c

	def matrixadd(A,B,mask):
		sum1 = np.add(np.array(A),np.array(B))
		sum2 = send_val(sum1.tolist())		

		return (np.add(np.array(sum2)),sum1).tolist()

	def matrixmul(A,B,U,V,Z):
		A = np.array(A)
		B = np.array(B)
		U = np.array(U)
		V = np.array(V)
		
		E = np.subtract(A,U)
		F = np.subtract(B,V)

		sendlist = []
		sendlist.append(E.tolist())
		sendlist.append(F.tolist())
		recv_info = send_val(sendlist)

		E = E + recv_info[0]
		F = F + recv_info[1]

		c = np.add(-1 * conf.partyNum * (np.multiply(E,F)),np.multiply(A*F) + np.multiply(E*B))
		C = reconstruct(c.tolist())

		C = (np.add(np.array(C),c).tolist())
		
		return C

	def truncate(x,scale):
		if(conf.partyNum==0):
			x = x/scale
		else:
			x = (2**conf.l) - x
			y = np.uint64(x)
			x = (y*(-1)/scale)
		return np.uint64(x)

	def addvectors(A,B):
		m,n = A.shape
		C = np.array([[0]*n]*m)
		for i in range(m):
			for j in range(n):
				print(i,j)
				C[i][j] = (A[i][j] + B[i][j])%(2**conf.l)
				print(C[i][j])
		return C

	def matrixmul_reg(A,B,E,F,V,Z):
		# Matrix multiplication using the protocol in the paper
		# A - data pt
		# B - weights
		# E = datapoint - data mask U
		# V - mask of weights for this batch
		# F = weights - weights mask V

		mul1 = np.matmul(E,F)
		for i in range(len(mul1)):
			mul1[i][0] = np.uint64(functionalities.int64tofloat(mul1[i][0]))
		mul2 = np.matmul(A,F)
		for i in range(len(mul2)):
			mul2[i][0] = np.uint64(functionalities.int64tofloat(mul2[i][0]))
		mul3 = np.matmul(E,B)
		for i in range(len(mul3)):
			mul3[i][0] = np.uint64(functionalities.int64tofloat(mul3[i][0]))
		mul0 = np.multiply(functionalities.floattoint64(-1 * conf.partyNum), mul1)
		for i in range(len(mul0)):
			mul0[i][0] = np.uint64(functionalities.int64tofloat(mul0[i][0]))

		Yhat1 = np.add(mul0,mul2)
		Yhat2 = np.add(mul3,Z)
		Yhat = np.add(Yhat1,Yhat2)

		return Yhat