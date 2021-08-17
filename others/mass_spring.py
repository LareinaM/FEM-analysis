import taichi as ti
import numpy as np

ti.init()

NUM_NODE = 31
GRAVITY = ti.Vector([0,-9.8])
WIDTH = 900
HEIGHT = 900


spring_stiffness = ti.var(dt=ti.float32, shape=())
damping = 30
particle_mass = 0.5

dt = 5e-5

x = ti.Vector(2, dt=ti.float64, shape=NUM_NODE)
v = ti.Vector(2, dt=ti.float64, shape=NUM_NODE)
v = ti.Vector(2, dt=ti.float64, shape=NUM_NODE)

rest_length = ti.var(dt=ti.float64, shape=(NUM_NODE, NUM_NODE))

@ti.kernel
def init_line():
    spring_stiffness[None] = 500

    for i in range(NUM_NODE):
        x[i] = [0.2 + 0.6*(i/(NUM_NODE-1)), 0.5]
        v[i] = [0.0, 0.0]

    for i,j in ti.ndrange(NUM_NODE, NUM_NODE):
        if ti.abs(j-i) == 1:
            rest_length[i, j] = (x[i]-x[j]).norm()
        else:
            rest_length[i, j] = 0.0

@ti.kernel
def init_bridge():
    spring_stiffness[None] = 500
    brige_height = 0.1

    for i in range(NUM_NODE):
        x[i] = [0.2 + 0.6 * (i / (NUM_NODE-1)), 0.5 + (i % 2) * brige_height]
        v[i] = [0.0, 0.0]

    for i,j in ti.ndrange(NUM_NODE, NUM_NODE):
        if ti.abs(j-i) == 1 or ti.abs(j-i) == 2:
            rest_length[i, j] = (x[i]-x[j]).norm()
        else:
            rest_length[i, j] = 0.0

@ti.kernel
def step():
    for i in range(NUM_NODE):
        v[i] *= ti.exp(-dt * damping)  # damping
        f = GRAVITY * particle_mass
        for j in range(NUM_NODE):
            if rest_length[i, j] != 0:
                f += spring_stiffness[None] * ((x[i]-x[j]).norm() - rest_length[i, j]) * (x[j] - x[i]).normalized()

        v[i] += f / particle_mass * dt
        x[i] += v[i] * dt

    x[0] = [0.2, 0.5]
    x[NUM_NODE-1] = [0.8, 0.5]


def display():
    for i in range(NUM_NODE):
        for j in range(NUM_NODE):
            if rest_length[i, j] != 0.0:
                gui.line(begin=x[i], end=x[j], radius=2, color=0x445566)

    gui.circles(x.to_numpy(), color=0x9093CB, radius=10)
    gui.text(content=f"stiffness: {spring_stiffness[None]}", pos=(0, 0.95), font_size=25)

if __name__ == "__main__":
    #init_line()
    init_bridge()
    gui = ti.GUI("Spring-mass", res=(WIDTH, HEIGHT), background_color=0x111111)

    while True:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == gui.UP:
                spring_stiffness[None] += 1000
                if spring_stiffness[None] >= 30000:
                    spring_stiffness[None] -= 1000
            else:
                e.key == gui.DOWN
                spring_stiffness[None] -= 1000
                if spring_stiffness[None] <= 0:
                    spring_stiffness[None] = 500

        for i in range(100):
            step()

        display()
        gui.show()

