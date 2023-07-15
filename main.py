from typing import List, Union, Tuple
from datetime import datetime
from math import cos, sin, exp, sqrt, pi

# No numpy :'(

# Project TODOs/Progression:
# - [X] Get all the math stuff set up
# - [X] Get a minimum working 3d example
# - [X] Work on the rubik's cube getting cubes and faces and all that jazz
#       working
# - [X] Get rotations on the whole cube working as well as general
#       transformations
# - [X] Get coloring of cube working
# - [X] Get the cube rotating with mouse drag
# - [X] Get actual rubik's cube moves going
# - [X] W

class Vec3:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        
    def add(self, other: 'Vec3') -> 'Vec3':
        return Vec3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
        
    def sub(self, other: 'Vec3') -> 'Vec3':
        return Vec3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )
        
    def length(self) -> float:
        # If this gets too slow then I'm going to square it and compare squared
        # distances like back in the old days
        return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        
    def normalize(self) -> 'Vec3':
        # If the length is 0 all of the components are 0 so just default to a
        # length of one so no division by zero happens.
        length = self.length() or 1
        return Vec3(
            self.x / length,
            self.y / length,
            self.z / length
        )
        
    def to_list(self) -> List[int]:
        return [self.x, self.y]
        
    def mult_mat_in_place(self, other: 'Mat3') -> 'Vec3':
        x = self.x * other.i.x + self.y * other.j.x + self.z * other.k.x
        y = self.x * other.i.y + self.y * other.j.y + self.z * other.k.y
        z = self.x * other.i.z + self.y * other.j.z + self.z * other.k.z
        
        self.x = x
        self.y = y
        self.z = z
        
    # See Mat3.mult for a reasoning on overloading
    def mult(self, other: Union[float, int, 'Mat3']) -> 'Vec3':
        if isinstance(other, float) or isinstance(other, int):
            return Vec3(
                other * self.x,
                other * self.y,
                other * self.z
            )
        elif isinstance(other, Mat3):
            return Vec3(
                self.x * other.i.x + self.y * other.j.x + self.z * other.k.x,
                self.x * other.i.y + self.y * other.j.y + self.z * other.k.y,
                self.x * other.i.z + self.y * other.j.z + self.z * other.k.z
            )
        else:
            raise ValueError("Bad value in multiplying vector")
            
    def dot(self, other: 'Vec3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
        
    def as_tuple(self) -> Tuple[float]:
        return (self.x, self.y, self.z)
        
    def __repr__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"
        
# Not a rug, but a 3d matrix class. This matrix is defined by the three basis
# vectors, i, j, and k.

# Not sure if I want to do operator overloading yet because that might be a bit
# hard to read, but that's for later me to decide right.
class Mat3:
    def __init__(self, i: Vec3, j: Vec3, k: Vec3) -> None:
        self.i = i
        self.j = j
        self.k = k
        
    # Ok you're not exactly gonna like this (and by you I mean definitely future
    # me) but we're overloading multiplication as opposed to making mult_mat,
    # mult_vec, mult_scalar and all those specialized functions because it's far
    # less cluttered.
    #
    # If you're want to multiply a vector by a matrix, see Vec3.mult(). Because
    # this returns a vector, I figured it wouldn't really make sense to have
    # multiple return types.
    def mult(self, other: Union[float, int, 'Mat3']) -> 'Mat3':
        if isinstance(other, float) or isinstance(other, int):
            return Mat3(
                self.i.mult(other),
                self.j.mult(other),
                self.k.mult(other)
            )
        elif isinstance(other, Mat3):
            # This is a decently expensive operation, so don't do it in a loop I
            # guess.
            # Hah too late now
            i = Vec3(self.i.x, self.j.x, self.k.x)
            j = Vec3(self.i.y, self.j.y, self.k.y)
            k = Vec3(self.i.z, self.j.z, self.k.z)
            
            # Ok this having to be read sideways has already caused me a number
            # of problems and headaches. -_-
            return Mat3(
                Vec3(i.dot(other.i), j.dot(other.i), k.dot(other.i)),
                Vec3(i.dot(other.j), j.dot(other.j), k.dot(other.j)),
                Vec3(i.dot(other.k), j.dot(other.k), k.dot(other.k))
            )
        else:
            raise ValueError("Bad value in multiplying matrix")

    def det(self) -> float:
        # Rule of Sarrus for the win
        # (Please tell me this is right)
        return (
            + self.i.x * self.j.y * self.k.z
            + self.j.x * self.k.y * self.i.z
            + self.k.x * self.i.y * self.j.z
            - self.i.x * self.k.y * self.j.z 
            - self.j.x * self.i.y * self.k.z
            - self.k.x * self.j.y * self.i.z
        )
        
    def inv(self) -> 'Mat3':
        # Yes this is pain. Applied mathematics moment
        det = self.det()
        
        a = self.i.x
        b = self.j.x
        c = self.k.x
        d = self.i.y
        e = self.j.y
        f = self.k.y
        g = self.i.z
        h = self.j.z
        i = self.k.z
        
        A = e * i - f * h
        B = f * g - d * i
        C = d * h - e * g
        D = c * h - b * i
        E = a * i - c * g
        F = b * g - a * h
        G = b * f - c * e
        H = c * d - a * f
        I = a * e - b * d
        
        # No the sideways stuff is not a mistake this time
        return Mat3(
            Vec3(A, B, C),
            Vec3(D, E, F),
            Vec3(G, H, I)
        ).mult(1 / det)

    def __repr__(self) -> str:
        # If I'm going to be staring at matrices for a couple hours, I might as
        # well make them look pretty.
        style_float = lambda c: ("+" if c > 0 else "-" if c < 0 else " ") + f"{abs(float(c)):.6f}"
        
        return f"""
┌ {style_float(self.i.x)} {style_float(self.j.x)} {style_float(self.k.x)} ┐
│ {style_float(self.i.y)} {style_float(self.j.y)} {style_float(self.k.y)} │
└ {style_float(self.i.z)} {style_float(self.j.z)} {style_float(self.k.z)} ┘
        """.strip()
        
class Face:
    def __init__(self, verts: List[Vec3]) -> None:
        # Ensure that points with higher z-values will be drawn first (and be
        # drawn over by points with lower z-values)
        # Actually when we apply transformations the z-value will change and I'm
        # not quite sure how to handle that but that's a problem for future me
        # as we say in the business.
        # There's also the fun problem where some faces could potentially
        # intersect each other, meaning that some parts are over some parts, but
        # not others, resulting in us not simply being able to move things in
        # front or back. Luckily, there is an efficient O(1) algorithm called
        # ignoring it and asserting that it won't happen. We're working with
        # cubes anyways if that problem comes up something's already blown up
        # probably. :thumbsup: (only responsible coding practices here)
        """
        self.verts = sorted(verts, key=lambda v: v.z, reverse=True)
        
        self.drawn_verts = Group()
        """
        self.verts = verts
        self.color = "white"
        self.face = Polygon(fill=self.color, border="black")

        # The "principle"/greatest z value in all of the points. This is used
        # to order faces.
        self.principle_z = max(vert.z for vert in self.verts)
        
    def set_color(self, color: str) -> None:
        self.color = color
        self.face.fill = color
        
    def draw(self):
        for vert in self.verts:
            projected = proj(vert)
            self.face.addPoint(projected.x, projected.y)

# We define this cube from its starting point (least x, greatest y, least z)
class Cube:
    # Indices for the faces based on location before transformations (kinda
    # hacky but it's not terrible?)
    Top = 0
    Bottom = 1
    Left = 2
    Right = 3
    Front = 4
    Back = 5
    
    
    def __init__(self, start: Vec3, length: float):
        # My brain is smooth right now so this is hard coded.
        # Ok so after thinking about it it seems like a combinatorics style
        # problem where we walk along the eight vertices (starting at (0, 0, 0)
        # or (length, -length, length)) and need to return to the starting point
        # in 4 moves. This sounds complicated so I'll just not for right now.
        # The only problem with automatically generating them I guess is that
        # I don't know which index corresponds to which face lol.
        self.faces = [
            # Top face (perpendicular to positive Y)
            Face([start.add(Vec3(0, 0, 0)), start.add(Vec3(length, 0, 0)), start.add(Vec3(length, 0, length)), start.add(Vec3(0, 0, length))]),
            # Bottom face (perpendicular to negative Y)
            Face([start.add(Vec3(length, -length, length)), start.add(Vec3(length, -length, 0)), start.add(Vec3(0, -length, 0)), start.add(Vec3(0, -length, length))]),
            # Left face (perpendicular to negative X)
            Face([start.add(Vec3(0, 0, 0)), start.add(Vec3(0, 0, length)), start.add(Vec3(0, -length, length)), start.add(Vec3(0, -length, 0))]),
            # Right face (perpendicular to positive X)
            Face([start.add(Vec3(length, -length, length)), start.add(Vec3(length, -length, 0)), start.add(Vec3(length, 0, 0)), start.add(Vec3(length, 0, length))]),
            # Front face (perpendicular to negative Z)
            Face([start.add(Vec3(0, 0, 0)), start.add(Vec3(length, 0, 0)), start.add(Vec3(length, -length, 0)), start.add(Vec3(0, -length, 0))]),
            # Back face (perpendicular to positive Z)
            Face([start.add(Vec3(length, -length, length)), start.add(Vec3(0, -length, length)), start.add(Vec3(0, 0, length)), start.add(Vec3(length, 0, length))]),
        ]
        
    def color(self, face: int, color: str) -> None:
        self.faces[face].set_color(color)
        
    def color_all(self, color: str) -> None:
        for i in range(6):
            self.color(i, color)
        
    def draw(self) -> None:
        for face in self.faces:
            face.draw()

# The class that stores the animation information about a specific layer being
# rotated
class Animation:
    Length = 20
    
    def __init__(self, basis: Mat3, axis: int, layer: int) -> None:
        rot_matrix = rot_x if axis == Rubiks.X else rot_y if axis == Rubiks.Y else rot_z
        
        
        # Okay so a little bit of explaining to be done here. Specifically
        # when we're rotating individual layers or faces a small bit of math
        # has to be done on top of what we already have. Because the cube is
        # already at a certain rotation angle, simply just rotating a face
        # around an axis is not going to work. Instead what we have to do
        # is first invert the cube rotation, apply the layer rotation, and
        # then finally reapply the cube rotation.
        self.step_rotation = basis.mult(rot_matrix(-pi / 2 / Animation.Length)).mult(basis.inv())
        self.axis = axis
        self.layer = layer
        
        self.t = 0
        
    def done(self) -> bool:
        # Should be checked after step is called.
        return self.t >= 1
        
    def step(self, rubiks: 'Rubiks') -> None:
        self.t += 1 / Animation.Length
        
        selected = rubiks.get_layer(self.axis, self.layer)
        
        rubiks.rotate(self.step_rotation, selected)
        
    def swap_indices(self, rubiks: 'Rubiks'):
        # After we make the rotations, the cubes are no longer ordered, so we
        # have to swap the indices of the cubes around to make them work
        # correctly. While I could find some funky math thing to get all the
        # indices in the correct place, it's 8:32 PM right now so let's do a bit
        # of hardcoding, shall we? :)
        # Note: should be called only once after the rotation is finished
        
        # After a bit of thinking and looking at a cube, we really only need
        # these six swaps because the middle one doesn't change. It turns out
        # this pattern is (almost) the exact same for each axis. Yay!
        swaps = [
            ((0, 0), (0, 2)),
            ((0, 0), (2, 2)),
            ((0, 0), (2, 0)),
            ((0, 1), (1, 2)),
            ((0, 1), (2, 1)),
            ((0, 1), (1, 0))
        ]
        
        for swap in swaps:
            start, end = swap
            if self.axis == Rubiks.X:
                start, end = Rubiks.index(self.layer, start[0], start[1]), Rubiks.index(self.layer, end[0], end[1])
            elif self.axis == Rubiks.Y:
                start, end = Rubiks.index(start[0], self.layer, start[1]), Rubiks.index(end[0], self.layer, end[1])
            elif self.axis == Rubiks.Z:
                start, end = Rubiks.index(start[1], start[0], self.layer), Rubiks.index(end[1], end[0], self.layer)
            rubiks.cubes[start], rubiks.cubes[end] = rubiks.cubes[end], rubiks.cubes[start]

# The class for the Rubik's Cube itself.
class Rubiks:
    # Not really worth making a full fledged enum so here we are. See
    # self.get_layer() for more context.
    X = 0
    Y = 1
    Z = 2
    
    def __init__(self):
        self.length = 2/3
        
        self.start = Vec3(-self.length / 2, self.length / 2, -self.length / 2)
        
        cube_length = self.length / 3
        self.cubes = [
            Cube(self.start.add(Vec3(x * cube_length, -y * cube_length, z * cube_length)), cube_length)
            for x in range(3)
            for y in range(3)
            for z in range(3)
        ]
        
        self.scroll_reference = None
        self.scroll_to = None
        
        self.rotation_measure = [
            Vec3(1, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, 0, 1)
        ]
        
        # We store this value for use with rotating individual layers.
        self.total_rotation = app.identity
        self.total_inversion = app.identity
        
        self.selected_axis = Rubiks.X
        self.selected_layer = 0
        
        self.current_animation = None

    def index(x: int, y: int, z: int) -> int:
        # Returns an index into self.cubes given x, y, and z coordinates in the
        # discrete interval [0, 3).
        
        return z + 3 * y + 9 * x
    
    def draw(self):
        # The initial drawing of the cube.
        for cube in self.cubes:
            cube.draw()
            
    def get_layer(self, axis: int, index: int) -> List[Cube]:
        indices = self.get_layer_indices(axis, index)
        
        return [self.cubes[Rubiks.index(i[0], i[1], i[2])] for i in indices]
            
    def get_layer_indices(self, axis: int, index: int) -> List[Tuple[int]]:
        if axis == Rubiks.X:
            return [(index, y, z) for y in range(3) for z in range(3)]
        elif axis == Rubiks.Y:
            return [(x, index, z) for x in range(3) for z in range(3)]
        else:
            return [(x, y, index) for x in range(3) for y in range(3)]

    # Hot diggity dog it's time to optimize this because I cannot stand the
    # English slideshow presentation speed of this rotating
    def rotate(self, rotation: Mat3, cubes: List[Cube] = None) -> None:
        total = not bool(cubes)
        cubes = cubes or self.cubes
        
        if total:
            for i, measure in enumerate(self.rotation_measure):
                self.rotation_measure[i].mult_mat_in_place(rotation)
        
        for cube in cubes:
            for face in cube.faces:
                # Okay due to some finicky stuff related to Polygon.pointList,
                # we have to do some equally finicky stuff to change the points.
                # Because I really have to optimize this to get frame rates of
                # over 5 fps, I need to consider a couple different options for
                # constructing the new pointList (I'm mostly worried about
                # the list being copied and allocating memory for objects that
                # could possibly be shaved off). Here are the methods I'm
                # considering and I'll probably add some fps counts for them
                # that I get during testing.
                # 1. Create a new list and append projected points. Then set
                #    face.face.pointList to the list
                #    FPS: 4.22 <- Winner winner chicken dinner
                #
                # 2. Assign a variable to face.face.pointList and change that in
                #    place, setting it back to face.face.pointList in the end.
                #    FPS: 3.93
                #
                # 3. Use addPoint to add all the projected points and then set
                #    face.face.pointList to its slice that only contains the new
                #    projected points.
                #    FPS: 2.41
                #
                # So uhhhh yeah, things aren't looking too hopeful in terms of
                # performance it seems. The only other thing I can think of
                # is making some in place operations so let's try that I guess.
                #
                # The best optimization I've observed so far is getting a better
                # laptop lol.
                projected_list = []
                
                for i, vert in enumerate(face.verts):
                    face.verts[i].mult_mat_in_place(rotation)
                    
                    projected = proj(face.verts[i])
                    projected_list.append([projected.x, projected.y])
                    
                # Somehow calling max on the generator is faster than doing it
                # in our loop up above idk python things.
                face.principle_z = max(vert.z for vert in face.verts)
                
                face.face.pointList = projected_list
    
    def scroll_step(self) -> None:
        # Here's a brief rundown of how exactly treat scrolling (with the left
        # mouse button). When the user clicks, we set self.scroll_reference to
        # be the reference point or starting point of the scroll. When the user
        # then drags the mouse around, we set self.scroll_to this new
        # destination (note that while I do use 3d vectors, all of this is in 2d
        # and only x and y rotations are considered). As the program runs, the
        # reference/starting point then moves linearly towards the destination
        # (we normalize the vector to get a constant speed). If the reference
        # point is in a certain threshold distance to the destination point
        # (specifically if it is at a distance less than the step we use), we
        # set it to the end point and stop scrolling.
        # We used the aforementioned normalized difference vector for the
        # x and y rotation angles.
        if self.current_animation is None and self.scroll_reference is not None and self.scroll_to is not None:
            scroll_speed = 0.5
            diff = self.scroll_to.sub(self.scroll_reference).normalize().mult(scroll_speed)
            
            length = self.scroll_to.sub(self.scroll_reference).length()
            
            if length < scroll_speed:
                diff = diff.mult(length / scroll_speed)
                self.scroll_reference = self.scroll_to
            else:
                self.scroll_reference = self.scroll_reference.add(diff)
            
            x_rotation = rot_x(diff.y)
            y_rotation = rot_y(diff.x)
            
            # TODO: Make an in-place operation for this and see if it speeds it
            # up
            rotation = x_rotation.mult(y_rotation)
            self.rotate(rotation)
            self.update()
        
    def get_rotation(self) -> Mat3:
        # Originally I had tried to keep track of the rotation matrix by keeping
        # a matrix and just multiplying it by each rotation we undergo, but it
        # seems the problem is that we just lose precision (at least that was my
        # best guess). Thus, in order to get the closest rotation matrix,
        # possible, we can just figure it out by doing some math, knowing where
        # the original points were and where they are now.
        
        # It's really bad, but I'm not really sure what else to do at this point
        
        # Original points (just taken from the first couple vertices of the
        # first face. They could have been any points but these were the ones
        # chosen idk)
        x1, y1, z1 = (1, 0, 0)
        x2, y2, z2 = (0, 1, 0)
        x3, y3, z3 = (0, 0, 1)
        
        # Transformed points
        tx1, ty1, tz1 = self.rotation_measure[0].as_tuple()
        tx2, ty2, tz2 = self.rotation_measure[1].as_tuple()
        tx3, ty3, tz3 = self.rotation_measure[2].as_tuple()
        
        inv = Mat3(
            Vec3(x1, x2, x3),
            Vec3(y1, y2, y3),
            Vec3(z1, z2, z3)
        ).inv()
        
        a11, a21, a31 = Vec3(tx1, tx2, tx3).mult(inv).as_tuple()
        a12, a22, a32 = Vec3(ty1, ty2, ty3).mult(inv).as_tuple()
        a13, a23, a33 = Vec3(tz1, tz2, tz3).mult(inv).as_tuple()
        
        return Mat3(
            Vec3(a11, a12, a13),
            Vec3(a21, a22, a23),
            Vec3(a31, a32, a33)
        )
        
        # Let's gooooooooooooooooooooooooooooooooo it works finally
    
    def update(self) -> None:
        # After we apply a transformation to the points in the cube, update
        # the rendering on screen.
        all_faces = sorted(
            (face for cube in self.cubes for face in cube.faces),
            key=lambda face: face.principle_z,
            reverse=True
        )
        
        selected = self.get_layer(self.selected_axis, self.selected_layer)

        for face in all_faces:
            face.face.border = "black"
            face.face.toFront()
            
        for cube in selected:
            for face in cube.faces:
                face.face.border = rgb(0x4f, 0x4f, 0x4f)
            
    def select_left(self) -> None:
        if self.current_animation is not None:
            return
        
        self.selected_layer = max(0, self.selected_layer - 1)
    
    def select_right(self) -> None:
        if self.current_animation is not None:
            return
        
        self.selected_layer = min(2, self.selected_layer + 1)
    
    def select_axis(self, axis: int) -> None:
        if self.current_animation is not None:
            return
        
        self.selected_axis = axis
        
    def rotate_layer(self) -> None:
        if self.current_animation is not None:
            return
        
        rotation = self.get_rotation()
        self.current_animation = Animation(
            rotation,
            self.selected_axis,
            self.selected_layer
        )
            
    def color(self) -> None:
        colors = {"red" : 0, "orange" : 0, "yellow" : 0, "green" : 0, "blue" : 0 , "white" : 0}
        
        def generate_color() -> str:
            color = choice(list(colors.keys()))
            
            colors[color] += 1
            
            if colors[color] > 9:
                del colors[color]
            
            return color
        
        color_map = {
            Cube.Top: {Rubiks.index(x, 0, z) for x in range(3) for z in range(3)},
            Cube.Bottom: {Rubiks.index(x, 2, z) for x in range(3) for z in range(3)},
            Cube.Left: {Rubiks.index(0, y, z) for y in range(3) for z in range(3)},
            Cube.Right: {Rubiks.index(2, y, z) for y in range(3) for z in range(3)},
            Cube.Front: {Rubiks.index(x, y, 0) for x in range(3) for y in range(3)},
            Cube.Back: {Rubiks.index(x, y, 2) for x in range(3) for y in range(3)}
        }
        
        # Start by coloring all sides black and then adding in colors where
        # appropriate. Perhaps there's a better, more organized way to do this,
        # but I'd like to move on and spend my time on other things.
        for i, cube in enumerate(self.cubes):
            cube.color_all("black")
            
            for face in color_map:
                if i in color_map[face]:
                    cube.color(face, generate_color())
        
    
# A simple, globally defined orthogonal projection (for the time being at least
# idk I might change it later). To understand an orthogonal projection, one need
# only remember but one thing: Yeet the Z.
# NOTE: While we do "Yeet the Z," we do need it for rendering order. Points and
# objects with lower z-values will be rendered after or on top of those with
# higher z-values, and thus will be "closer" to the screen.
#
# Note that given the way Mat3 is constructed, these should actually be read
# sideways if we think of the usual matrix notation.
# ...That being said this particular matrix is the same sideways and upright LOL
app.projection = Mat3(
    Vec3(1, 0, 0),
    Vec3(0, 1, 0),
    Vec3(0, 0, 0)
)

# Mostly for testing purposes, but this is your standard identity matrix.
# "If you ever forget who you are don't be afraid to consult your **identity matrix**"
# - Wise words
app.identity = Mat3(
    Vec3(1, 0, 0),
    Vec3(0, 1, 0),
    Vec3(0, 0, 1)
)

# Our tried-and-true, locally grown, free-range, antibiotic-free rotation
# matrices.
def rot_x(angle: float) -> Mat3:
    return Mat3(
        Vec3(1, 0, 0),
        Vec3(0, cos(angle), -sin(angle)),
        Vec3(0, sin(angle), cos(angle)),
    )
    
def rot_y(angle: float) -> Mat3:
    return Mat3(
        Vec3(cos(angle), 0, sin(angle)),
        Vec3(0, 1, 0),
        Vec3(-sin(angle), 0, cos(angle)),
    )

def rot_z(angle: float) -> Mat3:
    return Mat3(
        Vec3(cos(angle), sin(angle), 0),
        Vec3(-sin(angle), cos(angle), 0),
        Vec3(0, 0, 1),
    )

# Maps from [-1, 1] to [0, 400]
def scale_component(component: float) -> float:
    return (component + 1) * 200
    
# Maps from [0, 400] to [-1, 1]
def descale_component(component: float) -> float:
    return component / 200 - 1

# The global projection function that takes a vector with components in the
# the interval [-1, 1] and maps it to a vector with components in the interval
# [0, 400] (effectively maps the rubik's coordinates to our canvas coordinates)
def proj(v: Vec3) -> Vec3:
    # We negate the y-component here to account for rendering being flipped.
    scaled = Vec3(
        scale_component(v.x),
        scale_component(-v.y),
        0 #scale_component(v.z),
    )
    
    return scaled    
    
    # Okay so matrix multiplication is pretty slow but luckily our projection is
    # orthogonal anyways so just like don't multiply it by the projection.
    # return scaled.mult(app.projection)
    
# Testing; delete later
rubiks = Rubiks()
rubiks.color()
rubiks.draw()
rubiks.update()

app.frames = 0
start = datetime.now()
# Not really the most accurate indicator of fps (more like the
# average frame per second over the total runtime of the program, but this will
# do.
fps = Label("fps:", 370, 20, align="right")

# Okay so it seems that composing these rotations as opposed to applying the
# rotations to the cube individually gives different (like opposite y?) results
# so that's definitely something to look into. For now I'll leave it but I'm
# sure it'll come up at a later point.
#
# I sure have a lot of work ahead of me.
# rotation = rot_x(0.05).mult(rot_y(0.05)).mult(rot_z(0.05))

# Small little constants for respective mouse buttons
LeftClick = 0
RightClick = 2

def onMousePress(x, y, button):
    if button == LeftClick:
        rubiks.scroll_reference = Vec3(descale_component(x), descale_component(y), 0)
    elif button == RightClick:
        shape = app.group.hitTest(x, y)
        if shape is not None and isinstance(shape, Polygon):
            pass

def onMouseRelease(x, y):
    rubiks.scroll_reference = None
    rubiks.scroll_to = None

def onMouseDrag(x, y, button):
    rubiks.scroll_to = Vec3(descale_component(x), descale_component(y), 0)
    
def onKeyPress(key):
    if key == "left":
        rubiks.select_left()
    elif key == "right":
        rubiks.select_right()
    elif key in "xyz":
        rubiks.select_axis(
            Rubiks.X if key == "x" else
            Rubiks.Y if key == "y" else
            Rubiks.Z
        )
    elif key == "space":
        rubiks.rotate_layer()
    else:
        return
    rubiks.update()

def onStep():
    rubiks.scroll_step()
    
    if rubiks.current_animation is not None:
        rubiks.current_animation.step(rubiks)
        rubiks.update()
        
        if rubiks.current_animation.done():
            rubiks.current_animation.swap_indices(rubiks)
            rubiks.current_animation = None

    app.frames += 1
    
    time_diff = datetime.now() - start
    fps.value = "fps: " + str(pythonRound(app.frames / (time_diff.seconds + 1e-6 * time_diff.microseconds), 2))
