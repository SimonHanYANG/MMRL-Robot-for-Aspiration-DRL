# test_stage.py

import time
from stage import Stage

def test_stage_movement():
    print("Initializing stage...")
    stage = Stage()
    
    try:
        # Get initial position
        x_init, y_init = stage.get_xy_position()
        z_init = stage.get_z_position()
        print(f"Initial position: X={x_init:.2f}, Y={y_init:.2f}, Z={z_init:.2f}")
        
        # Move right (positive X)
        print("Moving right by 10 μm...")
        stage.move_xy_relative(100, 0)
        time.sleep(1)  # Wait for movement to complete
        x, y = stage.get_xy_position()
        print(f"New position: X={x:.2f}, Y={y:.2f}")
        
        # Move left (negative X)
        print("Moving left by 10 μm...")
        stage.move_xy_relative(-200, 0)
        time.sleep(1)
        x, y = stage.get_xy_position()
        print(f"New position: X={x:.2f}, Y={y:.2f}")
        
        # Move up (positive Y)
        print("Moving up by 10 μm...")
        stage.move_xy_relative(0, 100)
        time.sleep(1)
        x, y = stage.get_xy_position()
        print(f"New position: X={x:.2f}, Y={y:.2f}")
        
        # Move down (negative Y)
        print("Moving down by 10 μm...")
        stage.move_xy_relative(0, -200)
        time.sleep(1)
        x, y = stage.get_xy_position()
        print(f"New position: X={x:.2f}, Y={y:.2f}")
        
        # Move Z up (positive Z)
        print("Moving Z up by 10 μm...")
        stage.move_z_relative(100)
        time.sleep(1)
        z = stage.get_z_position()
        print(f"New Z position: Z={z:.2f}")
        
        # Move Z down (negative Z)
        print("Moving Z down by 10 μm...")
        stage.move_z_relative(-100)
        time.sleep(1)
        z = stage.get_z_position()
        print(f"New Z position: Z={z:.2f}")
        
        # Move to absolute position
        print("Moving to absolute position X init, Y...")
        stage.move_xy_to_absolute(x_init, y_init)
        time.sleep(2)
        x, y = stage.get_xy_position()
        print(f"Final XY position: X={x:.2f}, Y={y:.2f}")
        
        # Move Z to absolute position
        print("Moving Z to absolute position Z init...")
        stage.move_z_to_absolute(z_init)
        time.sleep(1)
        z = stage.get_z_position()
        print(f"Final Z position: Z={z:.2f}")
        
    except Exception as e:
        print(f"Error during stage test: {e}")
    
    finally:
        print("Closing stage connection...")
        stage.close()
        print("Test completed")

if __name__ == "__main__":
    test_stage_movement()