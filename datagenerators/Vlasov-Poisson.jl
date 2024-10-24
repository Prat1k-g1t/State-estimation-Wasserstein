using ForwardDiff
using LaTeXStrings
using NLsolve
using Plots
using LinearAlgebra
using PyCall
using JLD
using SparseArrays
using DelimitedFiles
using NPZ

# N: nombre d'intervalles de discrétisation dans chaque dimension
#const N = 200;
const Nx = 128; 
const Nv = 128; 

# num_cells: nombre de cellules
const num_cells =  Nx*Nv;

vmin = -10
vmax = 10

const dx = 10*pi/Nx;
const dy = (vmax-vmin)/Nv
const dt = 0.01;

#T1= 0.25
T = 50

const grid1Dx = range(0, 10*pi, length=Nx) |> collect;
const grid1Dv = range(-10, 10, length=Nv) |> collect;

#Retourne l'indice de la case m en fonction des indices (i,j) (1 <= i,j <= N) 
function getNumCase(i,j)
   #return (j-1)*Nx+i;
   return (i-1)*Nv+j;
end

#Retourne l'indice de la case m en fonction des indices (i,j) ( 1<= i,j <= N) 
function getIndicesCase(m)
  # i = m - floor(Int, (m-1)/Nx)*Nx
  # j = floor(Int, (m-i)/Nx) +1 
  j= m - floor(Int, (m-1)/Nv)*Nv
  i = floor(Int, (m-j)/Nv) +1 
   return i,j
end

V = spzeros(num_cells)
for i=1:Nx
    for j=1:Nv
        xi = (i-0.5)*dx
        vj = vmin + (j-0.5)*dy
        m = getNumCase(i,j)
        V[m] = vj
    end
end


Lap = spzeros(Nx,Nx)
for i=1:Nx
        Lap[i,i] = -2.0/(dx*dx)
       
        if (i<Nx)
            mpx = i+1
        else
            mpx =  1
        end
        Lap[i,mpx] = 1.0/(dx*dx)
       
        if (i>1)     
            mmx = i-1
        else
            mmx =  Nx
        end
        Lap[i,mmx] = 1.0/(dx*dx)
end  

Lapnew = Lap[1:(Nx-1), 1:(Nx-1)]

G = spzeros(Nx,Nx)
for i=1:Nx
        G[i,i] = 0
       
        if (i<Nx)
            mpx = i+1
        else
            mpx =  1
        end
        G[i,mpx] = 1.0/(2*dx)
       
        if (i>1)     
            mmx = i-1
        else
            mmx =  Nx
        end
        G[i,mmx] = -1.0/(2*dx)
end  


VGradx = spzeros(num_cells, num_cells)
for i=1:Nx
    for j=1:Nv
        xi= (i-0.5)*dx
        vj= -vmin + (j-0.5)*dy
        m = getNumCase(i,j)
        VGradx[m,m] = 0
       
        if (i<Nx)
            mpx = getNumCase(i+1,j)
        else
            mpx =  getNumCase(1,j)
        end
        VGradx[m,mpx] = 1.0/(2*dx)*vj
       
        if (i>1)     
            mmx = getNumCase(i-1,j)
        else
            mmx =  getNumCase(Nx,j)
        end
        VGradx[m,mmx] = -1.0/(2*dx)*vj
    end
end  

function E(rho)
    rhonew = 1.0 .-rho
    
    rhonew2 = rhonew[1:(Nx-1)]
    Vnew = Lapnew\rhonew2
    
    V = [Vnew;0]
    return G*V
end

function EGradv(E)
    EGradv = spzeros(num_cells, num_cells)
    for i=1:Nx
        for j=1:Nv
            xi= (i-0.5)*dx
            vj= -5 + (j-0.5)*dy
            m = getNumCase(i,j)
            EGradv[m,m] = 0
       
            if (j<Nv)   
                mpy = getNumCase(i,j+1)
            else
                mpy =  getNumCase(i,1)
            end
            EGradv[m,mpy] = 1.0/(2*dy)*E[i]
        
   
            if (j>1)
                mmy = getNumCase(i,j-1)
            else
                mmy =  getNumCase(i,Nv)
            end
            EGradv[m,mmy] = -1.0/(2*dy)*E[i]
    
        end
    end  
    return EGradv
end



Id = sparse(I,num_cells, num_cells);


############################################
#Boucle sur les parametres
#############################################

nint = 1

num_param = 51
v0tab = range(1.5,3.5,length = num_param)

## Imprimer les valeurs des parametres
#s_param=open("../data/VlasovPoisson/params.npz", "w");
npzwrite("../data/VlasovPoisson/params.npz", v0tab)

## Imprimer les valeurs des instants de temps
timetab  = [0; 0]
while(nint*dt < T)
	t = nint*dt
	global timetab = [timetab; t]
	global nint = nint+1
end
#s_time=open("../data/VlasovPoisson/times.npz", "w");
npzwrite("../data/VlasovPoisson/times.npz", timetab)
print(size(timetab))

##Imprimer les valeurs des coordonnees des points de la grille 
pointtab = zeros(num_cells,2)
for m in 1:num_cells
 	i,j = getIndicesCase(m)
	xi = (i-0.5)*dx
	vj= -vmin + (j-0.5)*dy
	pointtab[m,1] = xi
	pointtab[m,2] = vj	
end
#s_point=open("../data/VlasovPoisson/points.npz", "w");
npzwrite("../data/VlasovPoisson/points.npz", pointtab)
print(size(pointtab))



for l in 1:num_param

	v0 = v0tab[l]
	print("l=")
	println(l)
	println(v0)
	U0 = zeros(num_cells);
	#v0 = 1.3
	#v0 = 2.4
	#v0 = 3.0
	beta = 1e-3

	function g(x1,x2)
    		return (1+ beta*cos(0.2*x1)) *(1.0/(2*sqrt(2*pi))*exp(-0.5*(x2-v0)^2) +  1.0/(2*sqrt(2*pi))*exp(-0.5*(x2+v0)^2))
	end

	for i=1:Nx
    		for j=1:Nv
        		xi = (i-0.5)*dx
        		yj = vmin + (j-0.5)*dy
        		m = getNumCase(i,j)
        		println(m)
        		U0[m] = g(xi,yj)
    		end
	end

	U0 = U0/(sum(U0))*Nx*Nv
	1.0/Nx*1.0/Nv*sum(U0)


n=1

Utab = [U0 U0]
Uold0 = U0

while (n*dt < T)
    print("n = ")
    println(n)
    t = n*dt
    
    Uold0tab = reshape(Uold0, Nv,Nx)
    rho = Uold0tab'*((1.0/Nv)*ones(Nv,1))
    
    #Kmat = VGradx + EGradv(E(rho))
    Kmat = VGradx - EGradv(E(rho))
    #Explicite: 
    #Unew = Uold0 + dt*Kmat*Uold0
    
    #Implicite:
    #Cmat = Id -dt*Kmat
    Cmat = Id  + dt*Kmat;
    Unew = Cmat\Uold0;
    Unew = Unew/sum(Unew)*Nx*Nv;
    print(size(Unew))
    Utab = [Utab Unew]
    Uold0 = Unew
    n = n+1
end

##Imprimer Utab dans le fichier
#s_tab=open(string("../data/VlasovPoisson/vals",l,".npz"), "w");
npzwrite(string("../data/VlasovPoisson/vals",l,".npz"), Utab)
#s_tab.close()


end







