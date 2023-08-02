import matplotlib.pyplot as plt
import numpy as np
from math import *
import cmath as cm
import random as rd
import pandas as pd



class Load_exp_data:
	"""
	"""
	def __init__(self):	
		path = "data/20221206_aaa_CPRsolo_block1_tbl/meta_data/"
		self.data = pd.read_excel(path+"20221206_aaa_CPRsolo_block1_tbl.xlsx")
		print('data loaded')
		self.nb_frame = 300
		self.nb_pts = 503

	#get coherence lvl for line l
	def get_coh_val(self,l):
		return self.data['rdp_coh'][l]

	def get_timeframe(self,l):
		time_frame = []
		f = 1
		while f<self.nb_frame+1:
			nb = floor(np.log10(f))+1
			val = self.data["frme_ts_"+" "*(3-nb)+str(f)][l]
			if isnan(val):
				break
			time_frame.append(val*1e-6)
			f += 1
		return time_frame

	def get_coh_dir(self,l,size):
		return [self.data['rdp_dir'][l]]*size
	#size is the number of frames

	def get_joy_dir(self,l,size):
		js_dir = np.zeros(size)
		for f in range(1,size+1):
			nb = floor(np.log10(f))+1
			js_dir[f-1] = self.data["js_dir_"+" "*(3-nb)+str(f)][l]
		return list(js_dir)

	def get_joy_ecc(self,l,size):
		js_ecc = np.zeros(size)
		for f in range(1,size+1):
			nb = floor(np.log10(f))+1
			js_ecc[f-1] = self.data["js_ecc_"+" "*(3-nb)+str(f)][l]
		return list(js_ecc)

	def get_point_dir(self,l,size): 
		#x and y are matrices, for each line you have one point and columns are time frames
		x = np.zeros((self.nb_pts,size))
		y = np.zeros((self.nb_pts,size))
		#reading and collecting the data
		for i in range(size):
			data = np.loadtxt('data/20221206_aaa_CPRsolo_block1_tbl/signal/points_coord_'+str(l)+'_'+str(i)+'.txt')
			x[:,i] = data[:,0]
			y[:,i] = data[:,1]
		#calculating the direction at each time step
		dif_x = x[:,1:] - x[:,:-1]
		dif_y = y[:,1:] - y[:,:-1]
		point_dir = np.zeros((size-1,2))
		for t in range(size-1):
			somme_x = 0
			somme_y = 0
			nb = 0 
			for i in range(self.nb_pts):
				if abs(dif_x[i,t])<0.2 and abs(dif_y[i,t])<0.2:
					nb += 1
					somme_x += dif_x[i,t]
					somme_y += dif_y[i,t]
			if nb>0:
				point_dir[t,0] = somme_x/nb
				point_dir[t,1] = somme_y/nb
			else:
				point_dir[t,0] = np.nan
				point_dir[t,1] = np.nan
		theta = []
		for t in range(size-1):
			dx = point_dir[t,0]
			dy = point_dir[t,1]
			if dx>0 and dy>0:
				theta.append(atan(dx/dy))
			elif dx<0 and dy>0:
				theta.append(atan(dx/dy)+2*np.pi)
			elif dy<0 :
				theta.append(atan(dx/dy)+np.pi)
		return list(np.asarray(theta)*180/np.pi)

	#gets succesive dirations with the same coherence level
	def get_coh_block_pt(self):
		pt_dir = {}
		nb_block = self.data.shape[0]//10
		for b in range(nb_block):
			pt_dir[b] = []
			for l in range(10*b,(b+1)*10):
				size = len(self.get_timeframe(l))
				pt_dir[b] += self.get_point_dir(l,size)
		return pt_dir

	#return the joystick direction and eccentricity combined into a complex signal
	#for each coherence block
	def get_coh_block_js(self):
		js_dir = {}
		js_ecc = {}
		res = {}
		nb_block = self.data.shape[0]//10
		for b in range(nb_block):
			js_dir[b] = []
			js_ecc[b] = []
			for l in range(10*b,(b+1)*10):
				size = len(self.get_timeframe(l))
				js_dir[b] += self.get_joy_dir(l,size)
				js_ecc[b] += self.get_joy_ecc(l,size)

			res[b] = np.exp(1j*np.asarray(js_dir[b])*np.pi/180)#*np.asarray(js_ecc[b])
		return res

	#gets the nominal direction for a block of same coherence under complex form 
	def get_nominal_dir(self):
		D_n ={}
		nb_block = self.data.shape[0]//10
		for b in range(nb_block):
			D_n[b] = []
			for l in range(10*b,(b+1)*10):
				D_n[b] += [self.data['rdp_coh'][b*10]*cm.exp(1j*self.data['rdp_dir'][l]*np.pi/180)]*len(self.get_timeframe(l))
			D_n[b] = np.asarray(D_n[b])	
		return D_n

	def clean_nan(self,js,dn):
		dn = dn[~np.isnan(js)]
		js = js[~np.isnan(js)]
		return js, dn 


def D_n_matrix(D_n,n):
	mat = np.zeros((len(D_n),n),dtype=complex)
	for tau in range(n):
		mat[tau,:tau+1] = D_n[:tau+1][::-1]
	for tau in range(n,len(D_n)):
		mat[tau,:] = D_n[tau-n+1:tau+1][::-1]
	return mat

def Ker_mat(mat,Js):
	return np.dot(np.dot(np.linalg.inv(np.dot(mat.T,mat)),mat.T),Js)







