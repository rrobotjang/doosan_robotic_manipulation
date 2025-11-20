from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle
import trimesh

def load_mesh_to_moveit(path="/ros2_ws/collision_env.obj"):
    mesh = trimesh.load(path)

    co = CollisionObject()
    co.id = "environment"
    co.header.frame_id = "world"

    mesh_msg = Mesh()
    for f in mesh.faces:
        tri = MeshTriangle()
        tri.vertex_indices = [int(x) for x in f]
        mesh_msg.triangles.append(tri)
    for v in mesh.vertices:
        p = Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))
        mesh_msg.vertices.append(p)

    co.meshes = [mesh_msg]
    co.operation = CollisionObject.ADD

    pub.publish(co)
