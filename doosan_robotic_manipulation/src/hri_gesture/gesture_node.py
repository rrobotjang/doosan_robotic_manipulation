def hri_cb(self, msg):
    if msg.data == "STOP":
        self.hri_cmd = "STOP"
    elif msg.data == "START":
        self.hri_cmd = "START"
    elif msg.data == "CANCEL":
        # MoveIt emergency stop
        self.arm.stop()
