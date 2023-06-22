#%%
import numpy as np
import matplotlib.pyplot as plt

delT=0.01
xDIM=30
cutOFF=2.5
SIGMA=1.0
EPS=1.0
maxT=1000
pNUM=900
px=int(np.sqrt(pNUM))
kB=1.0
sysT=1.0
Tau=delT/0.0025
sig=1.
mu=np.round(np.sqrt(3*kB*sysT),3)/pNUM
Urc = 4 * ((1 / cutOFF) ** 12 - (1 / cutOFF) ** 6)
Frc = 48 * (1 / cutOFF)**2 * ((1 / cutOFF)**12 - 0.5 * (1 / cutOFF)**6)

x=np.linspace(1,xDIM-1,num=px)
y=np.linspace(1,xDIM-1,num=px)
X,Y=np.meshgrid(x,y)
pos=np.zeros((pNUM,2))
pos[:,0]=X.ravel()
pos[:,1]=Y.ravel()

vel=np.zeros((pNUM, 2))
vel[:, 0] = np.random.randn(pNUM)
vel[:, 1] = np.random.randn(pNUM)
vx,vy=np.sum(vel,axis=0)
vtmp=np.ones((pNUM))
vel[:,0]=vel[:,0]-vx*vtmp/pNUM
vel[:,1]=vel[:,1]-vy*vtmp/pNUM

plt.scatter(pos[:,0],pos[:,1],color='purple',marker='.')

#%%
accln=np.zeros((pNUM,2))
momentum=[]
temp=[]
totalE=[]

def force_calc(pos,pNUM,xDIM,cutOFF,Urc,Frc):
  forx=np.zeros_like(pos)
  pot_energy=0
  for i in range(len(pos)-1):
    for j in range(i+1,len(pos)):
      dists=pos[i]-pos[j]


      if dists[0]>xDIM/2:
        dists[0]-=xDIM
      elif dists[0]<-xDIM/2:
        dists[0]+=xDIM
      
      if dists[1]>xDIM/2:
        dists[1]-=xDIM
      elif dists[1]<-xDIM/2:
        dists[1]+=xDIM
      

      rij=np.linalg.norm(dists)
      dirn=dists/rij
      
      if rij<=cutOFF:
        f_interaction=48*dirn*(rij**-12 - 0.5* rij**-6)-Frc 
        pot_energy +=  4*(rij**-12 - rij ** -6)- Urc + rij * Frc - cutOFF*Frc
        
        forx[i,:]+=f_interaction 
        forx[j,:]-=f_interaction 

  return forx,pot_energy/pNUM

for te in range(maxT):  
  print(f"Running iteration::::{te}")
  pos += vel  * delT +  accln * delT ** 2
  #pos[:nA+nF,0]=np.clip(pos[:nA+nF,0],1,14)
  pos  = pos % xDIM
  accln_old=accln  
  accln,pe = force_calc(pos,pNUM,xDIM,cutOFF,Urc,Frc)    
  vel += 0.5*(accln+accln_old)  * delT
  
  p_inst=np.nansum(vel,axis=0)
  ke=(np.linalg.norm(p_inst)**2)/(2*pNUM)
  momentum.append(np.linalg.norm(p_inst))
  temp.append(ke)    
  totalE.append(ke+pe)     
  print("plotting...")
  plt.scatter(pos[:,0],pos[:,1],color='purple',marker='.')
  
  plt.xlim(0,30)
  plt.ylim(0,30)
  plt.tight_layout()
  plt.savefig(f'{te}.png')
  plt.close()

plt.subplot(3,1,1)
plt.plot(range(len(momentum)),momentum)
plt.ylabel('momentum')
plt.title('evolution of momentum and temperature')  

plt.subplot(3,1,2)
plt.plot(range(len(temp)),temp)
plt.ylabel('kE(temperature)')

plt.subplot(3,1,3)
plt.plot(range(len(totalE)),totalE)
plt.ylabel('totE Energy')
plt.savefig('details.png')
# %%
