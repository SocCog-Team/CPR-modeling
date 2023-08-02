import matplotlib.pyplot as plt
import numpy as np
from math import *
import cmath as cm
import random as rd



def funcJs1(t):
	res = np.zeros(t.shape)
	for i in range(len(res)):
		if t[i] > 0:
			res[i] = np.exp(-t[i])
	return res

class Generate_data:
	"""
	refresh_rate in Hertz
	"""
	def __init__(self, refresh_rate=120): 
		self.dt_obs = 1000/refresh_rate #sampling duration in ms
		self.gamma_OU = 0.05/self.dt_obs
		self.sigma_OU = np.sqrt(2*self.gamma_OU)
		self.dt_sim = min(1/self.gamma_OU,self.dt_obs)*0.1 # the simulation dt must be lower than observation dt and 1/gamma (in ms)
		self.nb_alpha = 10
		self.nb_frame = 300
		self.T = self.nb_alpha * self.nb_frame * self.dt_obs #the duration of the simulation in ms
		self.coh = 1
		self.epsilon_data = 0.05 #strength of the OU noise in point direction
		self.epsilon_JS = 0.05 #strength of the JS noise
		self.nb_pt = floor(self.T/self.dt_obs) #number of measurements

	def OU_process(self):
		#making an OU process
		nb = floor(self.T/self.dt_sim)
		x = np.zeros(nb)
		eta = np.random.normal(loc=0,scale=1,size=nb-1) # bruit blanc gaussien 
		for t in range(1,nb):
			x[t] = (1-self.gamma_OU*self.dt_sim)*x[t-1] + self.sigma_OU*np.sqrt(self.dt_sim)*eta[t-1]
		return x

	def get_data(self):
		alpha = np.random.random(self.nb_alpha)*np.pi*2
		d1 = self.OU_process()
		d2 = self.OU_process()
		data_dir = np.zeros(self.nb_pt,dtype=complex)
		nom_dir = np.zeros(self.nb_pt,dtype=complex)
		#generating data
		for k,a in enumerate(alpha):
			nom_dir[k*self.nb_frame:(k+1)*self.nb_frame] = self.coh*cm.exp(a*1j)
			data_dir[k*self.nb_frame:(k+1)*self.nb_frame] = self.coh*cm.exp(a*1j)
		data_dir += (d1[::floor(np.round(self.dt_obs/self.dt_sim))] + 1j*d2[::floor(np.round(self.dt_obs/self.dt_sim))]) * self.epsilon_data
		return data_dir, nom_dir

	def convolution(self,A,B):
		t = np.linspace(-self.T/2,self.T/2,len(B))
		return np.convolve(A(t),B,mode = 'same')*self.dt_obs

	def JS_1(self,data_dir,tau=0.5,lag=0): 
		return self.convolution(lambda t:funcJs1((t-lag)/tau)/tau,data_dir)

	def JS_2(self,data_dir,tau=0.5):
		d1 = self.OU_process()
		d2 = self.OU_process()
		js_1 = self.JS_1(data_dir,tau=tau)
		return  js_1 + (d1[::floor(np.round(self.dt_obs/self.dt_sim))] + 1j*d2[::floor(np.round(self.dt_obs/self.dt_sim))]) * self.epsilon_JS 

	def JS_3(self,data_dir,tau=0.5,lag=1):
		d1 = self.OU_process()
		d2 = self.OU_process() 
		js_1 = self.JS_1(data_dir,tau=tau,lag=lag)
		return  js_1 + (d1[::floor(np.round(self.dt_obs/self.dt_sim))] + 1j*d2[::floor(np.round(self.dt_obs/self.dt_sim))]) * self.epsilon_JS 



def autocorr(x):
	result = np.correlate(x, x, mode='full')
	return result[result.size//2:]*2/result.size


def D_n_matrix(D_n,n):
	mat = np.zeros((len(D_n),n),dtype=complex)
	for tau in range(n):
		mat[tau,:tau+1] = D_n[:tau+1][::-1]
	for tau in range(n,len(D_n)):
		mat[tau,:] = D_n[tau-n+1:tau+1][::-1]
	return mat

def Ker_mat(mat,Js):
	return np.dot(np.dot(np.linalg.inv(np.dot(mat.T,mat)),mat.T),Js)

def all_to_ker():
	gen = Generate_data()
	data_dir, D_n = gen.get_data()
	delta_D = data_dir - D_n
	n = 100 #kernel size
	list_t = np.linspace(0,n*gen.dt_obs,n)
	tau = 10*gen.dt_obs #kernel parameter in ms
	lag = 25*gen.dt_obs #kernel parameter in ms 

	J1 = gen.JS_1(data_dir,tau=tau)
	J2 = gen.JS_2(data_dir,tau=tau)
	J3 = gen.JS_3(data_dir,tau=tau,lag=lag)
	#d_n_mat : nominal direction as a matrix
	d_n_mat = D_n_matrix(D_n,n) #D_n[a:b]

	ker1 = Ker_mat(d_n_mat,J1)
	ker2 = Ker_mat(d_n_mat,J2)
	ker3 = Ker_mat(d_n_mat,J3)
	ker_th1 = np.exp(-list_t/tau)/tau
	ker_th2 = np.exp(-list_t/tau)/tau
	ker_th3 = np.exp(-(list_t-lag)/tau)/tau * np.heaviside(list_t-lag,0.5)

	list_ratio = []
	for (k,ker),ker_th in zip(enumerate([ker1,ker2,ker3]),[ker_th1,ker_th2,ker_th3]):
		ratio = np.max(np.abs(ker_th))/np.max(np.abs(ker[1:-1])) # multiplicative ratio that doesn't affect the characteristic time
		list_ratio.append(ratio)

	J_p1 = np.convolve(list_ratio[0]*ker1,D_n,mode='same')*gen.dt_obs
	J_p2 = np.convolve(list_ratio[1]*ker2,D_n,mode='same')*gen.dt_obs
	J_p3 = np.convolve(list_ratio[2]*ker3,D_n,mode='same')*gen.dt_obs

	dJ1 = J1 - J_p1
	dJ2 = J2 - J_p2
	dJ3 = J3 - J_p3

	C1 = np.correlate(dJ1,delta_D,mode='full')[len(dJ1)-1:]
	C2 = np.correlate(dJ2,delta_D,mode='full')[len(dJ2)-1:]
	C3 = np.correlate(dJ3,delta_D,mode='full')[len(dJ3)-1:]
	
	return C1, C2, C3


#auto correlation for t > 0
def autocorr_J_th(sigma,gamma,tau,t):
	tab1 = np.exp(-gamma*t)
	tab2 = np.exp(-t/tau)
	return sigma**2*((tab1+tab2)/(gamma*tau+1)+(tab2-tab1)/(gamma*tau-1))/(2*gamma)

def autocorr_D_th(sigma,gamma,t):
	return sigma**2*np.exp(-gamma*t)/gamma







