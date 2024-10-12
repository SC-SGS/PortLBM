import csv

import numpy

import matplotlib.pyplot as plot

min_x = 0
min_y = 0
min_time_step = 0

max_x = 0
max_y = 0
max_time_step = 0

read_content=[]

with open("../build/results.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        read_content.append(line)

min_time_step = int(read_content[1][0])
min_x = int(read_content[1][1])
min_y = int(read_content[1][2])


max_time_step = int(read_content[-1][0])
max_x = int(read_content[-1][1])
max_y = int(read_content[-1][2])

print("Got min_x = {}, max_x = {}, min_y = {}, max_y = {}, min_time_step = {}, max_time_step = {}".format(min_x, max_x, min_y, max_y, min_time_step, max_time_step))

densities = [numpy.ndarray([max_y, max_x]) for time in range(0,max_time_step+1)]
velocities = [numpy.ndarray([max_y, max_x]) for time in range(0,max_time_step+1)]

for i in range(1, len(read_content)):
    line = read_content[i]
    densities[int(line[0])-min_time_step][int(line[2])-min_y][int(line[1])-min_x] = (1.0/3.0) * float(line[-1])
    velocities[int(line[0])-min_time_step][int(line[2])-min_y][int(line[1])-min_x] = numpy.sqrt(pow(float(line[3]),2) + pow(float(line[4]),2))

x_ticks = []
current_x_tick = 20
while current_x_tick < max_x:
    x_ticks.append(current_x_tick)
    current_x_tick += 20

y_ticks = []
current_y_tick = 10
while current_y_tick < max_y:
    y_ticks.append(current_y_tick)
    current_y_tick += 10

plot.figure(dpi=200)
plot.title('Velocity field')
p = plot.imshow(velocities[-1], extent=(min_x-0.5, max_x+0.5, min_y-0.5, max_y+0.5), cmap='inferno', vmin=0, vmax=numpy.max(velocities[-1]))
plot.colorbar(p, orientation='horizontal')
plot.xticks(x_ticks)
plot.yticks(y_ticks)
plot.xlabel('x')
plot.ylabel('y')
plot.savefig("../images/flow/velocity_field.pdf", format="pdf", bbox_inches="tight")
plot.close()

plot.figure(dpi=200)
p = plot.imshow(velocities[-1], extent=(min_x-0.5, max_x+0.5, min_y-0.5, max_y+0.5), cmap='inferno', vmin=0, vmax=numpy.max(velocities[-1]))
plot.axis('off')
plot.savefig("../images/flow/velocity_field_borderless.pdf", format="pdf", bbox_inches="tight", pad_inches = 0)
plot.savefig("../images/flow/velocity_field_borderless.svg", format="svg", bbox_inches="tight", pad_inches = 0)
plot.close()

plot.figure(dpi=200)
plot.title('Pressure')
p = plot.imshow(densities[-1], extent=(min_x-0.5, max_x+0.5, min_y-0.5, max_y+0.5), cmap='viridis', vmin=numpy.min(densities[-1]), vmax=numpy.max(densities[-1]))
plot.colorbar(p, orientation='horizontal')
plot.xticks(x_ticks)
plot.yticks(y_ticks)
plot.xlabel('x')
plot.ylabel('y')
plot.savefig("../images/flow/pressure.pdf", format="pdf", bbox_inches="tight")
plot.close()