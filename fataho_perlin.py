import random
import numpy as np
from PIL import Image

# Simple implementation of Perlin as described in http://tinyurl.com/ycd7hurm
# This is not meant to be fast, it's meant to be clearly-understandable.
# Associated spreadsheet can be used to hand calculate pixel by the same methodology.

class Gradient:
    #orientation is upper-left corner = (0,0)   X-across Y-down

    def __init__(self, grids, width, height, vector_pool=None, precast=None):
        self.width = width
        self.height = height
        self.grids = grids
        self.gradients = dict()

        if precast is None:
            if vector_pool is None:
                vector_pool = [(1,1), (-1,1), (1,-1), (-1,-1)]
            for x in range(grids+1):
                for y in range(grids+1):
                    self.gradients[(x,y)] = random.choice(vector_pool  )
            print(f"using random gradients: {self.gradients}")
        else:
            assert len(precast) == 4 and self.grids==1, "precast gradients are meant just for a single grid"
            self.gradients[(0,0)] = precast[0]
            self.gradients[(1,0)] = precast[1]
            self.gradients[(0,1)] = precast[2]
            self.gradients[(1, 1)] = precast[3]
            print(f"using precast gradients: {self.gradients}")

    def __repr__(self):
        return f"{self.width}w x {self.height}h pixels.  Grids: {self.grids}.  {len(self.gradients)} gradient vectors"

    def vectors(self, x, y):
        #return the four gradient vectors surrounding pixel x,y

        left = x // (self.width / self.grids)
        right = left+1
        top = y // (self.height / self.grids)
        bottom = top +1
        #print(f"point {(x,y)} on {self.grids} grid of width {self.width}, height {self.height}")
        #print(f"left: {left}, right {right}, top {top}, bottom {bottom}")

        vectors = {
            (0,0): self.gradients[(left,top)],
            (1,0): self.gradients[(right,top)],
            (0,1): self.gradients[(left,bottom)],
            (1,1): self.gradients[(right,bottom)]
        }
        return vectors

    def local_coordinates(self, x,y):
        #return the local coordinates of x,y normalized to the relevant grid

        pixels_per_grid = self.width / self.grids
        local_x = x % pixels_per_grid
        local_y = y % pixels_per_grid
        #print(f"pixels_per_grid {pixels_per_grid}")
        #print(f"local_pixels: {(local_x, local_y)}")

        return (local_x/pixels_per_grid, local_y/pixels_per_grid)


def fade(t):
    #replaces linear interpolation curve with a more sigmoid-like curve
    return 6*t**5 -15*t**4 + 10*t**3

def linear(t):
    #linear interpolation curve
    return t

def interp(t, M,N, func=fade ):
    #print(f"t: {t}, M: {M}, N: {N}, func: {func}")
    t =  func(t)
    #print(f"adjusted t: {t}")
    #print(f"M+t*(N-M): {M + t*(N-M)}")
    return M + t*(N-M)

def perlin(x, y, gradients, func=fade):
    #print(f"\nPERLIN\nx: {x}, y: {y}")

    xx, yy = gradients.local_coordinates(x,y)
    #print(f"localized xx, yy = {(xx,yy)}")
    gvs = gradients.vectors(x,y)
    #print(f"relevant gradients for {xx,yy} are: {gvs}")
    dots = dict()

    #print(f"localized: xx: {xx},  yy: {yy}")

    for corner in [ (0,0), (1,0), (0,1), (1,1)  ]:
        #print(f"\ncorner: {corner}")
        dv = (xx-corner[0], yy-corner[1])  # direction vector
        gv = gvs[corner]
        dots[corner] = dv[0]*gv[0] + dv[1]*gv[1]
        #print(f"direction vector {dv}")
        #print(f"gradient vector {gv}")
        #print(f"dot product: {dots[corner]}")

    #print(f"\ninterpolate AB")
    ab = interp( xx, dots[0,0], dots[1,0], func)

    #print(f"\ninterpolate CD")
    cd = interp(xx, dots[0,1], dots[1,1], func )

    #print(f"\ninterpolate AB-CD")
    abcd = interp( yy, ab, cd, func)

    #print(f"ab: {ab},   cd: {cd},   abcd: {abcd}")
    return abcd

def main():

    grids = 10  #square grids overlayed on pixel screen
    width = 500
    height = 500

    #precast = [(-1,-1), (1,-1), (-1,1), (1,1)] #all pointing away
    #precast = [(1, 1), (-1, 1), (1, -1), (-1, -1)]  #all pointing in

    #precast = [(1, 1), (1, -1), (-1, 1), (1, 1)]  #UpperLeft pointing in
    #precast = [(-1,-1), (-1,1), (-1,1), (1,1)] #UpperRight pointing in
    #precast = [(-1,-1), (1,-1), (1,-1), (1,1)] # LowerLeft pointing in
    #precast =  [(-1,-1), (1,-1), (-1,1), (-1,-1)] # LowerRight pointing in

    # Whether to run with random or precast gradient vectors
    #g = Gradient(grids, width, height, precast=precast)
    g = Gradient(grids, width, height)

    data = np.zeros((width, height), )  #raw interpolated data
    imgdata_grayscale = np.zeros((width, height), dtype=np.uint8)    #data normalized for display

    for x in range(width):
        for y in range(height):
            value = perlin(x, y, g, func=fade)
            _r, _g, _b = 1, 1, 1
            data[x,y] = value
            adj_value = (value + 1) / 2
            imgdata_grayscale[x, y] = adj_value*255


    #pixel data to view for debugging
    checks = [
        (90,10), (30,70), (1,99),(10,99), (20,99), (30,99), (40,99), (50,99), (75,99)
    ]
    for c in checks:
        print(f"value and adjusted-value of data[{c}] = {data[c]} ({imgdata_grayscale[c]})")

    img = Image.fromarray(imgdata_grayscale, 'L')
    img.save('perlin.png')
    img.show()

    print("done")

if __name__ == "__main__":
    main()
