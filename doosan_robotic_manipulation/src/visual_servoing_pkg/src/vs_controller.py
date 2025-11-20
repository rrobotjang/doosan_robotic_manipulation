import numpy as np

class VisualServoController:
    def __init__(self, fx, fy, cx, cy, gain=0.8):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.gain = gain

    def compute_control(self, u, v, z):
        du = u - self.cx
        dv = v - self.cy

        # Image Jacobian
        L = np.array([
            [-self.fx/z, 0, u/z, u*v/self.fx, -(self.fx**2 + u**2)/self.fx, v],
            [0, -self.fy/z, v/z, (self.fy**2 + v**2)/self.fy, -u*v/self.fy, -u]
        ])

        e = np.array([du, dv])
        v = -self.gain * np.linalg.pinv(L) @ e
        return v
