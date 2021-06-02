import socket
import numpy as np

class Config:
	PORT = -1
	partyNum = -1
	IP = socket.gethostbyname(socket.gethostname()) 
	advIP = IP 
	advPORT = -1
	l = 64
	lby2 = 32
	modl = 2**l
	precision = 13
	converttoint64 = (1<<precision)
	trunc_parameter = (1>>precision)
	epochs = 10
	OT = 0
	LHE = 1
	TRIPLET_GEN_MODE = OT
	
	# data specific

	n=-1
	d=-1
	t=-1
	batchsize=1
	alpha = 0.01 # learning rate
	alpha_inv = (1/alpha)
	train = int(506*(80/100))
	test = 506 - train