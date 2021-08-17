import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

WIDTH = 640
HEIGHT = 800

dt = 1e-3

num_balls = 30
ball_radius = 10

x = ti.Vector(2, dt=ti.float32, shape=num_balls)
v = ti.Vector(2, dt=ti.float32, shape=num_balls)
m = ti.var(dt=ti.float32, shape=num_balls)
m.from_numpy(np.ones(shape=num_balls, dtype=np.float32))

gravity = ti.Vector([0, -9.8])

gui = ti.GUI("Bouncing ball", res=(WIDTH, HEIGHT), background_color=0xDDDDDD)

@ti.kernel
def step():
    for i in range(num_balls):
        v[i] += gravity * dt

        # collide with the container
        if x[i].x <= ball_radius / WIDTH or x[i].x >= 1 - ball_radius / WIDTH:
            v[i].x *= -1
        elif x[i].y <= ball_radius / HEIGHT or x[i].y >= 1 - ball_radius / HEIGHT:
            v[i].y *= -1

        # TODO: check collide with the ball
        # TODO: simulate collision among balls

        for j in range(num_balls):
            """
            if x[i].x == x[j].x and x[i].y == x[j].y and not i == j:
                v[i] *= -1
                v[j] *= -1
            """
            dis = (x[i] - x[j]).norm()
            dis_x = abs(x[i].x - x[j].x)
            dis_y = abs(x[i].y - x[j].y)
            if dis_x <= ball_radius*2/WIDTH and dis_y <= ball_radius*2/HEIGHT and not i == j:
                v[i] *= -1
                v[j] *= -1
        
        x[i] += v[i] * dt

rand_position = np.random.rand(num_balls, 2)
rand_position = np.array(rand_position, dtype=np.float32)
x.from_numpy(rand_position)

rand_velocity = np.random.rand(num_balls, 2)
rand_velocity = np.array(rand_velocity, dtype=np.float32) * 5
v.from_numpy(rand_velocity)

for i in range(10000):
    step()
    gui.circles(x.to_numpy(), color=0xFFFFFF, radius=ball_radius)
    gui.show()
