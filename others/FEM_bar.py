import taichi as ti
import numpy as np
import math

ti.init(arch=ti.cpu)

node_num_x = 40
node_num_y = 8
node_num_total = node_num_x * node_num_y
num_element = (node_num_x - 1) * (node_num_y - 1) * 2

GRAVITY = ti.Vector([0, -9.8])
WIDTH = 900
HEIGHT = 900

spring_stiffness = ti.var(dt=ti.float32, shape=())
damping = 10
particle_mass = 0.1

dt = 1e-4

x = ti.Vector(2, dt=ti.float32, shape=node_num_total, needs_grad = True)
v = ti.Vector(2, dt=ti.float32, shape=node_num_total)
total_energy = ti.var(dt=ti.float32, shape=(), needs_grad=True)

# the indices to vertices
triangles = ti.var(dt=ti.int32, shape=(num_element, 3))

B = ti.Matrix(2, 2, dt=ti.float32, shape=num_element)

# material physical properties
E, nu = 1000, 0.3 # young's modulus, poisson's ratio
la = E * nu / ((1 + nu) * (1 - 2 * nu)) # parameters related to E and nu
mu = E / (2 * (1 + nu))
node_mass = 1
tri_area = 1

def init_mesh():
    # initialize nodes position
    mesh = lambda i, j: i * node_num_y + j
    dx = 1/32

    for i in range(node_num_x):
        for j in range(node_num_y):
            t = mesh(i, j)
            x[t] = [0.1 + i * dx * 0.5, 0.7 + j * dx * 0.5 + i * dx * 0.1]
            v[t] = [0, 0]

    # build mesh
    for i in range(node_num_x - 1):
        for j in range(node_num_y - 1):
            # element id
            eid = (i * (node_num_y - 1) + j) * 2
            triangles[eid, 0] = mesh(i, j)
            triangles[eid, 1] = mesh(i + 1, j)
            triangles[eid, 2] = mesh(i, j + 1)

            eid = (i * (node_num_y - 1) + j) * 2 + 1
            triangles[eid, 0] = mesh(i, j + 1)
            triangles[eid, 1] = mesh(i + 1, j + 1)
            triangles[eid, 2] = mesh(i + 1, j)


@ti.func
def compute_D(i):
    a = triangles[i, 0]
    b = triangles[i, 1]
    c = triangles[i, 2]
    return ti.Matrix.cols([x[b] - x[a], x[c] - x[a]])


@ti.kernel
def compute_B():
    for i in range(num_element):
        B[i] = compute_D(i).inverse()

@ti.kernel
def cal_total_energy():
    for i in range(num_element):
        D = compute_D(i)
        F = D @ B[i]

        # NeoHookean
        I1 = (F @ F.transpose()).trace()
        J = max(0.2, F.determinant())
        element_energy_density = 0.5 * mu * (I1 - 1) - mu * ti.log(J) + 0.5 * la * ti.log(J) ** 2
        total_energy[None] += element_energy_density * tri_area

@ti.kernel
def step():
    for i in range(node_num_total):

        # negative gradient is exactly the force
        v[i] += dt * ((-x.grad[i] / node_mass) + ti.Vector([0, -10])) * math.exp(dt * -6)

        # boundary condition
        if i // node_num_y == 0:
            v[i] = [0, 0]

        x[i] += dt * v[i]

def display():
    # for i in range(num_element):
    #     for j in range(3):
    #         i1 = triangles[i,j%3]
    #         i2 = triangles[i,(j+1)%3]
    #         gui.line(begin=x[i1], end=x[i2], radius=2, color=0x445566)

    gui.text(content=f"energy: {total_energy[None]}", pos=(0, 0.95), font_size=25)
    gui.circles(x.to_numpy(), color=0x9093CB, radius=5)


if __name__ == "__main__":
    init_mesh()
    gui = ti.GUI("FEM-bar", res=(WIDTH, HEIGHT), background_color=0x111111)
    # inverse matrix pre-computation
    compute_B()
    while True:
        for i in range(30):
            with ti.Tape(loss=total_energy):
                cal_total_energy()
            step()
        display()
        gui.show()

