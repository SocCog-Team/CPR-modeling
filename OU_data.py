import matplotlib.pyplot as plt
import numpy as np
from math import *
import cmath as cm
import random as rd
from scipy.signal import fftconvolve
import matplotlib.animation as animation



class Animation:
	"""docstring for Animation"""
	def __init__(self,data_dir,c):
		#initialize the figure
		self.fig, self.ax = plt.subplots()
		self.ax.set_aspect('equal')
		self.line, = self.ax.plot([],[])
		self.x = np.real(data_dir)
		self.y = np.imag(data_dir)
		b1 = np.max(abs(self.x))
		b2 = np.max(abs(self.y))
		self.ax.set_xlim(-b1,b1)
		self.ax.set_ylim(-b2,b2)
		self.nb_frame = len(data_dir)
		theta = np.linspace(0,2*np.pi,1000)
		self.ax.plot(c*np.cos(theta),c*np.sin(theta))
	
	#i is the frame number
	def animate(self,i):
		self.line.set_data([0,self.x[i]],[0,self.y[i]])
		return self.line,

	def make_anim(self):
		ani = animation.FuncAnimation(self.fig,self.animate,frames=self.nb_frame,interval=1,blit=True,repeat=False)
		ani.save('/Users/Selma/Desktop/PhD/data/OU_data/animation/OU_data.gif',writer='pillow')




def OU_process(sigma,T):
	# making an OU process
	gamma = sigma**2/2 # theoretical
	dt = 0.1/gamma #time step
	nb = floor(T/dt) #nb of time steps
	x = np.zeros(nb) 
	eta = np.random.normal(loc=0,scale=1,size=nb-1) # bruit blanc gaussien
	for t in range(1,nb):
		x[t] = (1-gamma*dt)*x[t-1] + sigma*np.sqrt(dt)*eta[t-1]
	return x


# generate random data folowing the OU process 
# alpha is the nominal direction (angle in rad)
# noise is the OU process  

def get_data():
	nb_alpha = 10
	alpha = np.random.random(nb_alpha)*np.pi*2
	list_size = np.random.randint(250,300,nb_alpha)
	T = nb_alpha * 300
	sigma = 1
	c = 1
	epsilon = 0.1
	d1 = OU_process(sigma,T)
	d2 = OU_process(sigma,T)
	data_dir = np.zeros(np.sum(list_size),dtype=complex)
	D_n = np.zeros(np.sum(list_size),dtype=complex)
	j = 0
	#generating data
	for a,l in zip(alpha,list_size):
		data_dir[j:j+l] = c*cm.exp(a*1j) + (d1[j:j+l] + 1j*d2[j:j+l]) * epsilon
		D_n[j:j+l] = c*cm.exp(a*1j) 
		j += l
	return data_dir, D_n


def convolution(A,B,T,nb):
	dt = 2*T/nb 
	t = np.linspace(-T,T,nb) 
	conv = np.convolve(A(t),B,mode = 'same')*dt
	return conv


def JS_1(data_dir,T=100,tau=0.5,lag=0): 
	return convolution(lambda t:np.exp(-(t-lag)/tau)*np.heaviside(t-lag,0.5)/tau,data_dir,T,len(data_dir))  


def JS_2(data_dir,T=100,tau=0.5):
	d1 = OU_process(1,40*T)
	d2 = OU_process(1,40*T) 
	eta = 0.02
	js_1 = JS_1(data_dir,T=T,tau=tau)
	return  js_1 + (d1[len(d1)-len(js_1):] + 1j*d2[len(d2)-len(js_1):]) * eta 


def JS_3(data_dir,T=100,tau=0.5,lag=1):
	d1 = OU_process(1,40*T)
	d2 = OU_process(1,40*T) 
	eta = 0.02
	js_1 = JS_1(data_dir,T=T,tau=tau,lag=lag)
	return  js_1 + (d1[len(d1)-len(js_1):] + 1j*d2[len(d2)-len(js_1):]) * eta 


def J_p_fourier(D_n,J): # is not precise 
	J_hat = np.conjugate(np.fft.fft(J))
	D_n_hat = np.conjugate(np.fft.fft(D_n))
	return np.fft.ifft(J_hat/D_n_hat)


def D_n_matrix(D_n,n):
	mat = np.zeros((len(D_n),n),dtype=complex)
	for tau in range(n):
		mat[tau,:tau+1] = D_n[:tau+1][::-1]
	for tau in range(n,len(D_n)):
		mat[tau,:] = D_n[tau-n+1:tau+1][::-1]
	return mat


def Ker_mat(mat,Js):
	return np.dot(np.dot(np.linalg.inv(np.dot(mat.T,mat)),mat.T),Js)








