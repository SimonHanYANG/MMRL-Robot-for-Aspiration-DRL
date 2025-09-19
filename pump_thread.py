# This code is for controling left injection pump motor
# Author: Aojun Jiang 
# Date: May 15, 2025
# Note: Other APIs can refer to https://www.phidgets.com/?view=api&product_id=1067_0&lang=Python
# Version: 0.1 (pass)
import ctypes
import os
from queue import Queue
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from Phidget22.Phidget import *
from Phidget22.Devices.Stepper import *
import time


class LeftPumpThread(QThread):
    position_updated = pyqtSignal(float, float, float)  

    def __init__(self):
        super().__init__()

        self.tasks = Queue()
        self.mutex = QMutex()  
        self._shutdown = False
        self.leftpump = Stepper()
        self.leftpump.openWaitForAttachment(1000)
        self.leftpump.setEngaged(True)
        self.leftpump.setAcceleration(160000)
        self.leftpump.setVelocityLimit(40000)
        self.leftpump.setCurrentLimit(2.0)

    def add_task(self, task):
        locker = QMutexLocker(self.mutex)  
        self.tasks.put(task)


    def shutdown(self):
        self._shutdown = True

    def run(self):
        while not self._shutdown:
            # 获取任务
            task = None
            with QMutexLocker(self.mutex):  
                if not self.tasks.empty():
                    task = self.tasks.get()

            if task:
                try:
                    if task[0] == "Step_CW":

                        self.leftpump.setControlMode(StepperControlMode.CONTROL_MODE_STEP)
                        position = self.leftpump.getPosition()
                        step = task[1]
                        target_position = position + step
                        self.leftpump.setTargetPosition(target_position)

                    elif task[0] == "Step_CCW":

                        self.leftpump.setControlMode(StepperControlMode.CONTROL_MODE_STEP)
                        position = self.leftpump.getPosition()
                        step = task[1]
                        target_position = position - step
                        self.leftpump.setTargetPosition(target_position)

                    elif task[0] == "SetSpeed":
                        self.leftpump.setControlMode(StepperControlMode.CONTROL_MODE_STEP)
                        speed = task[1]
                        self.leftpump.setVelocityLimit(speed)

                    elif task[0] == "Run":
                        print("commad run")
                        speed = task[1]
                        #time.sleep(0.2)
                        self.leftpump.setControlMode(StepperControlMode.CONTROL_MODE_RUN)
                        self.leftpump.setVelocityLimit(1000)

                except Exception as e:
                    print(f"Pump Motor Error: {str(e)}")

            # Update Positions
            #x = ctypes.c_double(0.0)
            #y = ctypes.c_double(0.0)
            #self.get_xy_pos(self.stage, ctypes.byref(x), ctypes.byref(y))
            #z = self.get_z_pos(self.stage)
            #self.position_updated.emit(x.value, y.value, z)

            self.msleep(33)  

    def shutdown(self):
        with QMutexLocker(self.mutex):
            self._shutdown = True

    def __del__(self):
        self.shutdown()