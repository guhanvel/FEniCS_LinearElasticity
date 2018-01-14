from __future__ import print_function
from fenics import *

# Scaled variables
L = 1; W = 0.2
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

# Create mesh and define function space
mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 10, 3, 3)
V = VectorFunctionSpace(mesh, 'P', 1)

# Define boundary condition

class Left(SubDomain):
    def inside(self, x, on_boundary):
      return near(x[0], 0.0)
    
class Right(SubDomain):
    def inside(self, x, on_boundary):
#return on_boundary and  abs(x[0]-L) < 0.2 and abs(x[1]-W) < 0.2  and abs(x[1]-W) < 0.2
      return near(x[0], L)
      
left = Left()
right = Right()
sub_domains = MeshFunction("size_t", mesh)
sub_domains.set_all(0)
right.mark(sub_domains, 1)
left.mark(sub_domains, 2)

bc = DirichletBC(V, Constant((0, 0, 0)), left)
ds = Measure("ds")[sub_domains]
#ds = ds(subdomain_data = MeshFunction)

# Define strain and stress

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    #return sym(nabla_grad(u))

def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension()  # space dimension
v = TestFunction(V)
f = Constant((0, 0, 0))

#Load at the right end
T = Constant((0, 0, -10))
a = inner(sigma(u), epsilon(v))*dx
L = dot(T, v)*ds(1)

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution
plot(u, title='Displacement', mode='displacement')

# Plot stress
s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d)  # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s))
V = FunctionSpace(mesh, 'P', 1)
von_Mises = project(von_Mises, V)
plot(von_Mises, title='Stress intensity')

# Compute magnitude of displacement
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)
plot(u_magnitude, 'Displacement magnitude')
print('min/max u:',
      u_magnitude.vector().array().min(),
      u_magnitude.vector().array().max())

# Save solution to file in VTK format
# File('elasticity/displacement.pvd') << u
# File('elasticity/von_mises.pvd') << von_Mises
# File('elasticity/magnitude.pvd') << u_magnitude

# Hold plot
#interactive()

