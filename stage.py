# stage.py - 修正版

import ctypes
import os

class Stage:
    def __init__(self):
        """Initialize the Stage controller for Nikon Ti2E"""
        # .dll path
        self.dll_dir = r"D:\\Aojun\\Ti2E_API\StageCPP\\x64\\Release"
        os.environ['PATH'] = self.dll_dir + os.pathsep + os.environ['PATH']

        # Load .dll 
        self.dll_path = os.path.join(self.dll_dir, "StageCPP.dll")
        self.stage_dll = ctypes.WinDLL(self.dll_path)

        # Setup function definitions
        self._setup_functions()
        
        # Create and connect stage
        self.stage = self._create_stage()
        self._connect_stage(self.stage)

    def _setup_functions(self):
        """Setup function definitions from the DLL"""
        # Create/Dispose stage
        self._create_stage = self.stage_dll.CreateStage
        self._create_stage.restype = ctypes.c_void_p

        self._dispose_stage = self.stage_dll.DisposeStage
        self._dispose_stage.argtypes = [ctypes.c_void_p]

        # Connection
        self._connect_stage = self.stage_dll.ConnectStage
        self._connect_stage.argtypes = [ctypes.c_void_p]  

        # Motion control
        self._move_xy_absolute = self.stage_dll.MoveXYtoAbsolute
        self._move_xy_absolute.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]

        self._move_xy_relative = self.stage_dll.MoveXYRelative
        self._move_xy_relative.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]

        self._move_z_absolute = self.stage_dll.MoveZtoAbsolute
        self._move_z_absolute.argtypes = [ctypes.c_void_p, ctypes.c_double]

        # Position Info
        self._get_z_pos = self.stage_dll.GetZPositionCurrent
        self._get_z_pos.restype = ctypes.c_double
        self._get_z_pos.argtypes = [ctypes.c_void_p]

        self._get_xy_pos = self.stage_dll.GetPositionCurrent
        self._get_xy_pos.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        ]

    def move_xy_to_absolute(self, x, y):
        """Move XY stage to absolute position
        
        Args:
            x (float): X position in μm
            y (float): Y position in μm
        """
        self._move_xy_absolute(self.stage, ctypes.c_double(x), ctypes.c_double(y))
        
    def move_xy_relative(self, x_delta, y_delta):
        """Move XY stage by relative amount
        
        Args:
            x_delta (float): X movement in μm
            y_delta (float): Y movement in μm
        """
        self._move_xy_relative(self.stage, ctypes.c_double(x_delta), ctypes.c_double(y_delta))
        
    def move_z_to_absolute(self, z):
        """Move Z stage to absolute position
        
        Args:
            z (float): Z position in μm
        """
        self._move_z_absolute(self.stage, ctypes.c_double(z))
        
    def move_z_relative(self, z_delta):
        """Move Z stage by relative amount
        
        Args:
            z_delta (float): Z movement in μm
        """
        current_z = self.get_z_position()
        target_z = current_z + z_delta
        self._move_z_absolute(self.stage, ctypes.c_double(target_z))
        
    def get_xy_position(self):
        """Get current XY position
        
        Returns:
            tuple: (x, y) position in μm
        """
        x = ctypes.c_double(0.0)
        y = ctypes.c_double(0.0)
        self._get_xy_pos(self.stage, ctypes.byref(x), ctypes.byref(y))
        return (x.value, y.value)
        
    def get_z_position(self):
        """Get current Z position
        
        Returns:
            float: Z position in μm
        """
        return self._get_z_pos(self.stage)
        
    def close(self):
        """Close the stage connection and dispose resources"""
        if hasattr(self, 'stage') and self.stage:
            self._dispose_stage(self.stage)
            self.stage = None
            
    def __del__(self):
        """Destructor to ensure resources are freed"""
        self.close()