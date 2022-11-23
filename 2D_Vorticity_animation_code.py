#took 2008.8485360145569 Seconds



#t=999.0(all values) with changed dispp for faster output
import time
st=time.time()
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FFMpegWriter
plt.rcParams['animation.ffmpeg_path']='C:\\Users\\Madhav Sirohi\\Downloads\\FFmp\\ffmpeg-2022-07-10-git-846488cca8-essentials_build\\ffmpeg-2022-07-10-git-846488cca8-essentials_build\\bin\\ffmpeg.exe'
plt.rcParams['figure.figsize']=(10,10)

global kx 
global ky 
global Laplace_inv 
global K 
global Dealias
global N

T=1000                  #defining total time
dt=0.1                  # time step
N=384                   # Grid resolution
nu=2.4e-34              # viscosity
r=8                     # part of hyperviscous terms
tmax = T/dt             # Max timesteps

a = 0                   # physical size of the domain
b = 2*np.pi
L = b-a 

dx = L/N                # grid size 


k = 2*np.pi/L*np.arange(-N/2,N/2)   
k=np.fft.fftshift(k) #shifting frequencies


[kx, ky] =  np.meshgrid(k, k) # defining the mesh in wavenumber space
K = np.sqrt( kx**2 + ky**2) 
x_ = a + dx*np.arange(0,N)
[x,y] = np.meshgrid(x_, x_);  # defining the mesh in physical space
Laplace  = -K**2;         # defining Laplacian in K space
Laplace_inv = -(kx**2+ky**2)**-1;  # defining Laplacian inverse in K space
#print(Laplace_inv[0,0])
Laplace_inv[0,0] = 0;

Dealias=(abs(kx)<N/3)*(abs(ky)<N/3) ;

M1 =  np.ones((N,N));

# INITIALIZATION for the vorticity field
balanced_strength_initial=100; 
K_initial= 6;    # initial energy in modes K < K_initial might need changes
theta_bt   = 2*np.pi*random.random((N,N));

psik_bt = balanced_strength_initial*(K<K_initial)*np.exp(1j*theta_bt) ; # stream fn 


zTk = Laplace*psik_bt;
zT = np.real( np.fft.ifft2(zTk) );   # Initial vorticity


def RHS_2DVorticity(zT1):
    psik=Laplace_inv*np.fft.fft2(zT1) ;  
    uTk=- 1j*ky*psik   ; 
    vTk=1j*kx*psik ;  
    uT=np.real(np.fft.ifft2(uTk))   ;  
    vT=np.real(np.fft.ifft2(vTk)) ;
    zTuTk= np.fft.fft2(uT*zT1)  ;  
    zTvTk= np.fft.fft2(vT*zT1); # -ADVECTIVE TERM in the RHS
    Rz=-np.real(np.fft.ifft2(Dealias*(1j*kx*(zTuTk)+1j*ky*(zTvTk))));
    return Rz




M_disip    =nu*K**(2*r) ;  # HTPERVIS TERM
#M      =  M1 + dt.*M_disip   ; 
M= np.ones((N,N))+ dt*M_disip; # HYPERVISC
dispp=10/dt;  # to display


metadata=dict(title='2D Vorticity animation video.mp4', artist='Madhav') #mp4 utilities
writer=FFMpegWriter(fps=15, metadata=metadata)
fig=plt.figure()
ax=plt.axes()
ax.set_xbound(0,6)
ax.set_ybound(0,6)
def plotT(a,b,c):
    cp=plt.contourf(a,b,c,cmap=cm.jet,levels=100)
fig.colorbar(cp)
#  START of Time Integration
display('begun time integration ...')
with writer.saving(fig,'2D Vorticity animation video.mp4',100):
    for i in range(int(tmax)): # START of time loop i
       t=i*dt
       # RK4 intermediate steps 
       zT1 = zT; 
       # stage 1
       Rz = RHS_2DVorticity(zT1);
       qz1  = Rz;
       zT1 = zT + dt*qz1/2;
       # stage 2
       Rz = RHS_2DVorticity(zT1);
       qz2  = Rz;
       zT1 = zT + dt*qz2/2;
       # stage 3
       Rz = RHS_2DVorticity(zT1);
       qz3  = Rz;
       zT1 = zT + dt*qz3;
       # stage 4
       Rz = RHS_2DVorticity(zT1);
       qz4  = Rz;
       zT =  M1*zT + dt*( qz1  + 2*qz2  + 2*qz3  + qz4  )/6  ;
       zTk = np.fft.fft2( zT ) ;
       zTk = zTk/M ; # HYPERVISC
       zT   = np.real( np.fft.ifft2(zTk) ); # the final vorticity field

    
    
       #code for the contour plot
       if (i%(1*dispp)==0):   # condition of display 
           t=i*dt;  
            #Making the figure for the animation
           #fig,ax=plt.subplots(1,1)
           #ax.set_xbound(0,6)
           #ax.set_ybound(0,6)
           ax.set_title('t='+str(t))
           #cp=plt.contourf(x,y,zT,cmap=cm.jet,levels=100)
           plotT(x,y,zT)
           
           writer.grab_frame()
       
                            # saving zT physical field at each 1 times
   #if (i%(1*dispp)==0):   # condition of display
    #   t=i*dt;           # Display the vorticity for gif
        

#python code
   
       #plt.contourf(x,y,zT[:,:])
       #plt.xlabel('x')
       #plt.ylabel('y')
       #plt.show()


        



toc_time = time.time()-st
print(toc_time)
