import math
from dataclasses import dataclass
from typing import Union, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

Vector = Tuple[float, float]

def dot(a: Vector, b: Vector) -> float:
    return a[0]*b[0] + a[1]*b[1]

def sub(a: Vector, b: Vector) -> Vector:
    return (a[0]-b[0], a[1]-b[1])

def neg(a: Vector) -> Vector:
    return (-a[0], -a[1])

@dataclass
class FriendlyPose:
    x: float
    y: float
    theta: float

@dataclass
class Circle:
    radius: float

@dataclass
class Rectangle:
    w: float
    h: float

Primitive = Union[Circle, Rectangle]

@dataclass
class TheInstance:
    name: str
    pose: FriendlyPose
    primitive: Primitive

def get_rangle_corners(r: TheInstance) -> List[Vector]:
    x0, y0 = r.pose.x, r.pose.y
    w, h = r.primitive.w, r.primitive.h
    theta = math.radians(r.pose.theta)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    local = [(0,0), (w,0), (w,h), (0,h)]
    return [(x0 + dx * cos_t - dy * sin_t, y0 + dx * sin_t + dy * cos_t) for dx, dy in local]

def support_circle(c: TheInstance, d: Vector) -> Vector:
    length = (d[0]**2 + d[1]**2)**0.5
    if length == 0:
        return (c.pose.x + c.primitive.radius, c.pose.y)
    ux, uy = d[0]/length, d[1]/length
    return (c.pose.x + ux * c.primitive.radius, c.pose.y + uy * c.primitive.radius)

def support_rectangle(r: TheInstance, d: Vector) -> Vector:
    corners = get_rangle_corners(r)
    best = corners[0]
    best_val = dot(best, d)
    for p in corners[1:]:
        v = dot(p, d)
        if v > best_val:
            best_val, best = v, p
    return best

def support_minkowski(a: TheInstance, b: TheInstance, d: Vector) -> Vector:
    if isinstance(a.primitive, Circle):
        sa = support_circle(a, d)
    else:
        sa = support_rectangle(a, d)

    if isinstance(b.primitive, Circle):
        sb = support_circle(b, neg(d))
    else:
        sb = support_rectangle(b, neg(d))

    return sub(sa, sb)

def handle_simplex(simplex: List[Vector], d: Vector) -> bool:
    if len(simplex) == 2:
        B, A = simplex
        AB = sub(B, A)
        AO = neg(A)
        perp = (AB[1], -AB[0])
        if dot(perp, AO) < 0:
            perp = (-AB[1], AB[0])
        d = perp
        simplex.clear()
        simplex.extend([B, A])
        return False
    else:
        C, B, A = simplex
        AB = sub(B, A)
        AC = sub(C, A)
        AO = neg(A)

        ab_perp = (AB[1], -AB[0]) if dot((AB[1], -AB[0]), AC) > 0 else (-AB[1], AB[0])
        ac_perp = (AC[1], -AC[0]) if dot((AC[1], -AC[0]), AB) < 0 else (-AC[1], AC[0])

        if dot(ab_perp, AO) > 0:
            simplex.clear()
            simplex.extend([B, A])
            d[:] = ab_perp
            return False
        if dot(ac_perp, AO) > 0:
            simplex.clear()
            simplex.extend([C, A])
            d[:] = ac_perp
            return False
        return True

def gjk_collide(a: TheInstance, b: TheInstance) -> bool:
    d = sub((b.pose.x, b.pose.y), (a.pose.x, a.pose.y))
    if d == (0, 0):
        d = (1.0, 0.0)
    simplex = [support_minkowski(a, b, d)]
    d = neg(simplex[0])
    while True:
        A = support_minkowski(a, b, d)
        if dot(A, d) <= 0:
            return False
        simplex.append(A)
        d_list = list(d)
        if handle_simplex(simplex, d_list):
            return True
        d = tuple(d_list)

def read_objects(filename: str) -> List[TheInstance]:
    objects = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!'):
                continue
            parts = line.split()
            if parts[0] == 'circle':
                name = parts[1]
                x = float(parts[2])
                y = float(parts[3])
                radius = float(parts[4])
                pose = FriendlyPose(x, y, 0)
                objects.append(TheInstance(name, pose, Circle(radius)))
            elif parts[0] == 'rectangle':
                name = parts[1]
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                angle = float(parts[6])
                pose = FriendlyPose(x, y, angle)
                objects.append(TheInstance(name, pose, Rectangle(w, h)))
    return objects

def visualize_objects(objects: List[TheInstance], collisions: set):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    for obj in objects:
        color = 'red' if obj.name in collisions else 'blue' if isinstance(obj.primitive, Circle) else 'yellow'

        if isinstance(obj.primitive, Circle):
            circle = patches.Circle((obj.pose.x, obj.pose.y), obj.primitive.radius, color=color, fill=True, alpha=0.6)
            ax.add_patch(circle)
            ax.text(obj.pose.x, obj.pose.y + obj.primitive.radius, obj.name, ha='center', va='bottom', color='black', fontsize=8)
        else:
            corners = get_rangle_corners(obj)
            rect = patches.Polygon(corners, closed=True, color=color, fill=True, alpha=0.6)
            ax.add_patch(rect)
            # Calculate center for label placement
            center_x = sum(c[0] for c in corners) / 4
            center_y = sum(c[1] for c in corners) / 4
            ax.text(center_x, center_y, obj.name, ha='center', va='center', color='black', fontsize=8)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Collision Visualization')
    plt.autoscale()
    plt.show()

if __name__ == "__main__":
    objects = read_objects('objects.txt')
    collisions = set()
    n = len(objects)
    print("Colliding pairs:")
    for i in range(n):
        for j in range(i+1, n):
            a, b = objects[i], objects[j]
            if gjk_collide(a, b):
                print(f"  {a.name} - {b.name}")
                collisions.add(a.name)
                collisions.add(b.name)
    visualize_objects(objects, collisions)
