#!/bin/python3

import subprocess
import math

DEBUG = False
version = "Debug" if DEBUG else "Release"
program = "godRay.exe"
out_path = "raw/"

exec_path = "C:/Users/Noah Pitts/Desktop/godRay_build4/bin/{}/{}".format(version, program)


f_fmt = ("raw/output_{:0>4d}.png").format
kwargs = {
    "-nf" : "1000",
    "-wh" : ["200", "150"]
}

start_frame = 0
num_frames = 200
step_size = 40

#camera_start =

cx_start = 100.0
cx_end = -60.0

cy_start = 115.0
cy_end = -40.0

cz_start = 0.0
cz_end = -250.0

for i in range(start_frame,num_frames+step_size,step_size):
    u = i / num_frames

    t1 = 80  * math.pi / 180
    t2 =  10  * math.pi / 180

    p1 = 0   * math.pi / 180
    p2 = 360 * math.pi / 180

    p = p1*u + p2*(1-u)

    cx = cx_start*u + cx_end*(1-u)
    cy = cy_start*u + cy_end*(1-u)
    cz = cz_start*u + cz_end*(1-u)

    if u < 0.5:
        t = t1*u*2 + t2*(1-u*2)
    else:
        t = t2*(u-0.5)*2 + t1*(1-(u-0.5)*2)

    kwargs["-st"] = "{:.4f}".format(t)
    kwargs["-sp"] = "{:.4f}".format(p)
    # kwargs["-cp"] = ["{:.4f}".format(cx), "{:.4f}".format(cy), "{:.4f}".format(cz)]
    kwargs["-f"] = f_fmt(i)

    args = [exec_path]
    for k in kwargs:
        args.append(k)
        if(isinstance(kwargs[k], list)):
            args += kwargs[k]
        else:
            args.append(kwargs[k])

    print("Calling:", args)
    subprocess.call(args)
