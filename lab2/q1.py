import math
import numpy as np
from matplotlib import pyplot as plt


def binomialModel(M,S0,K,T,r,sigma,x):
	delta = T/M
	u = d = 0
	if(x==1):
		u = math.exp((sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
		d = math.exp((-sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
	else:
		u = math.exp(sigma*math.sqrt(delta))
		d = math.exp(-sigma*math.sqrt(delta))
	p = (math.exp(r*delta)-d)/(u-d)
	q = (u-math.exp(r*delta))/(u-d)
	priceCall = []
	pricePut = []
	current = S0*(u**M)
	for i in range(M+1):
		priceCall.append(max(0,current-K))
		pricePut.append(max(0,K-current))
		current = current*d/u
	for i in range(M):
		for j in range(M-i):
			priceCall[j] = math.exp(-r*delta)*(p*priceCall[j] + q*priceCall[j+1])
			pricePut[j] = math.exp(-r*delta)*(p*pricePut[j] + q*pricePut[j+1])
	return [priceCall[0],pricePut[0]]



S0 = 100
K = 100
T = 1
M = 100
r = 0.08
sigma = 0.2

prices = [binomialModel(M,S0,K,T,r,sigma,0),binomialModel(M,S0,K,T,r,sigma,1)]
for i in range(2):
    print("Price of European Call option using set {x} = {p}".format(x=(i+1),p=prices[i][0]))
    print("Price of European Put option using set {x} = {p}\n".format(x=(i+1),p=prices[i][1]))



plt.style.use("seaborn")

def twoDplot(x,calls,puts,legend,xlabel,ylabel,title):
	plt.style.use("seaborn")
	plt.plot(x,calls)
	plt.plot(x,puts)
	plt.legend(legend)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()

 	
for st in [0,1]:
    calls = []
    puts = []
    for S in np.arange(0,205,5):
        x = binomialModel(M,S,K,T,r,sigma,st)
        calls.append(x[0])
        puts.append(x[1])
    twoDplot(np.arange(0,205,5),calls,puts,["Call Option","Put Option"],"S0","Price","Price  of Call and Put Option vs S0 using set {st}".format(st = (st+1)))

for st in [0,1]:
	calls = []
	puts = []
	for k in np.arange(0,205,5):
		x = binomialModel(M,S0,k,T,r,sigma,st)
		calls.append(x[0])
		puts.append(x[1])
	twoDplot(np.arange(0,205,5),calls,puts,["Call Option","Put Option"],"K","Price","Price  of Call and Put Option vs K using set {st}".format(st = (st+1)))

for st in [0,1]:
	calls = []
	puts = []
	for R in np.linspace(0,0.2,100):
		x = binomialModel(M,S0,K,T,R,sigma,st)
		calls.append(x[0])
		puts.append(x[1])
	twoDplot(np.linspace(0,0.2,100),calls,puts,["Call Option","Put Option"],"r","Price","Price  of Call and Put Option vs r using set {st}".format(st = (st+1)))

for st in [0,1]:
	calls = []
	puts = []
	for sig in np.linspace(0.1,0.3,100):
		x = binomialModel(M,S0,K,T,r,sig,st)
		calls.append(x[0])
		puts.append(x[1])
	plt.xlabel("sigma")
	plt.ylabel("Price")
	plt.title("Price  of Call Option vs sigma using set {st}".format(st = (st+1)))
	plt.plot(np.linspace(0.1,0.5,100),calls)
	plt.show()
	plt.xlabel("sigma")
	plt.ylabel("Price")
	plt.title("Price  of Put Option vs sigma using set {st}".format(st = (st+1)))
	plt.plot(np.linspace(0.1,0.5,100),puts)
	plt.show()

for K in [95,100,105]:
	for st in [0,1]:
		calls = []
		puts = []
		for m in np.arange(1,201):
			x = binomialModel(m,S0,K,T,r,sigma,st)
			calls.append(x[0])
			puts.append(x[1])
		plt.xlabel("M")
		plt.ylabel("Price")
		plt.title("Price  of Call Option vs M using set {st} and K = {K}".format(st = (st+1),K = K))
		plt.plot(np.arange(1,201),calls)
		plt.show()
		plt.xlabel("M")
		plt.ylabel("Price")
		plt.title("Price  of Put Option vs M using set {st} and k = {K}".format(st = (st+1),K = K))
		plt.plot(np.arange(1,201),puts)
		plt.show()	


sarr = np.arange(0,201,5)
karr = np.arange(0,201,5)
rarr = np.linspace(0,1,50)
sigarr = np.linspace(0.01,1,50)
marr = np.arange(1,201,5)

def plot3D(X,Y,Z1,Z2,st,p1,p2):
	plt.style.use("seaborn")
	ax = plt.axes(projection ='3d')
	ax.scatter(X,Y,Z1)
	ax.set_title("Price of Call Option vs "+p1 + " and "+p2+(" using set {st}".format(st=(st+1))))
	ax.set_xlabel(p1)
	ax.set_ylabel(p2)
	ax.set_zlabel("Price")
	plt.show()
	ax = plt.axes(projection ='3d')
	ax.scatter(X,Y,Z2)
	ax.set_title("Price of Put Option vs "+p1 + " and "+p2+(" using set {st}".format(st=(st+1))))
	ax.set_xlabel(p1)
	ax.set_ylabel(p2)
	ax.set_zlabel("Price")
	plt.show()



for st in [0,1]:
	X,Y = np.meshgrid(sarr,rarr)
	Z1 = np.empty(np.shape(X))
	Z2 = np.empty(np.shape(Y))
	for i in range(np.shape(X)[0]):
		for j in range(np.shape(Y)[1]):
			tmp = binomialModel(M,X[i,j],K,T,Y[i,j],sigma,st)
			Z1[i,j] = tmp[0]
			Z2[i,j] = tmp[1]
	plot3D(X,Y,Z1,Z2,st,"S0","r")
	

for st in [0,1]:
	X,Y = np.meshgrid(sarr,karr)
	Z1 = np.empty(np.shape(X))
	Z2 = np.empty(np.shape(Y))
	for i in range(np.shape(X)[0]):
		for j in range(np.shape(Y)[1]):
			tmp = binomialModel(M,X[i,j],Y[i,j],T,r,sigma,st)
			Z1[i,j] = tmp[0]
			Z2[i,j] = tmp[1]
	plot3D(X,Y,Z1,Z2,st,"S0","K")

for st in [0,1]:
	X,Y = np.meshgrid(sarr,sigarr)
	Z1 = np.empty(np.shape(X))
	Z2 = np.empty(np.shape(Y))
	for i in range(np.shape(X)[0]):
		for j in range(np.shape(Y)[1]):
			tmp = binomialModel(M,X[i,j],K,T,r,Y[i,j],st)
			Z1[i,j] = tmp[0]
			Z2[i,j] = tmp[1]
	plot3D(X,Y,Z1,Z2,st,"S0","sigma")

for st in [0,1]:
	X,Y = np.meshgrid(sarr,marr)
	Z1 = np.empty(np.shape(X))
	Z2 = np.empty(np.shape(Y))
	for i in range(np.shape(X)[0]):
		for j in range(np.shape(Y)[1]):
			tmp = binomialModel(Y[i,j],X[i,j],K,T,r,sigma,st)
			Z1[i,j] = tmp[0]
			Z2[i,j] = tmp[1]
	plot3D(X,Y,Z1,Z2,st,"S0","M")


for st in [0,1]:
	X,Y = np.meshgrid(karr,rarr)
	Z1 = np.empty(np.shape(X))
	Z2 = np.empty(np.shape(Y))
	for i in range(np.shape(X)[0]):
		for j in range(np.shape(Y)[1]):
			tmp = binomialModel(M,S0,X[i,j],T,Y[i,j],sigma,st)
			Z1[i,j] = tmp[0]
			Z2[i,j] = tmp[1]
	plot3D(X,Y,Z1,Z2,st,"K","r")


for st in [0,1]:
	X,Y = np.meshgrid(karr,sigarr)
	Z1 = np.empty(np.shape(X))
	Z2 = np.empty(np.shape(Y))
	for i in range(np.shape(X)[0]):
		for j in range(np.shape(Y)[1]):
			tmp = binomialModel(M,S0,X[i,j],T,r,Y[i,j],st)
			Z1[i,j] = tmp[0]
			Z2[i,j] = tmp[1]
	plot3D(X,Y,Z1,Z2,st,"K","sigma")


for st in [0,1]:
	X,Y = np.meshgrid(karr,marr)
	Z1 = np.empty(np.shape(X))
	Z2 = np.empty(np.shape(Y))
	for i in range(np.shape(X)[0]):
		for j in range(np.shape(Y)[1]):
			tmp = binomialModel(Y[i,j],S0,X[i,j],T,r,sigma,st)
			Z1[i,j] = tmp[0]
			Z2[i,j] = tmp[1]
	plot3D(X,Y,Z1,Z2,st,"K","M")


for st in [0,1]:
	X,Y = np.meshgrid(rarr,np.linspace(0.15,1,50))
	Z1 = np.empty(np.shape(X))
	Z2 = np.empty(np.shape(Y))
	for i in range(np.shape(X)[0]):
		for j in range(np.shape(Y)[1]):
			tmp = binomialModel(M,S0,K,T,X[i,j],Y[i,j],st)
			Z1[i,j] = tmp[0]
			Z2[i,j] = tmp[1]
	plot3D(X,Y,Z1,Z2,st,"r","sigma")
 
for st in [0,1]:
	X,Y = np.meshgrid(rarr,marr)
	Z1 = np.empty(np.shape(X))
	Z2 = np.empty(np.shape(Y))
	for i in range(np.shape(X)[0]):
		for j in range(np.shape(Y)[1]):
			tmp = binomialModel(Y[i,j],S0,K,T,X[i,j],sigma,st)
			Z1[i,j] = tmp[0]
			Z2[i,j] = tmp[1]
	plot3D(X,Y,Z1,Z2,st,"r","M")
 
for st in [0,1]:
	X,Y = np.meshgrid(np.linspace(0.15,1,50),marr)
	Z1 = np.empty(np.shape(X))
	Z2 = np.empty(np.shape(Y))
	for i in range(np.shape(X)[0]):
		for j in range(np.shape(Y)[1]):
			tmp = binomialModel(Y[i,j],S0,K,T,r,X[i,j],st)
			Z1[i,j] = tmp[0]
			Z2[i,j] = tmp[1]
	plot3D(X,Y,Z1,Z2,st,"sigma","M")