import requests
import json
import time

headers={
    'User-Agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Mobile Safari/537.36'
}
s = requests.Session()

def responseHandle(response):
    # 确保请求成功
    if response.status_code == 200:
        try:
            # 解析JSON响应
            data = response.json()
            
            # 提取需要的字段
            successed = data.get("successed", False)  # 默认为False如果字段不存在
            result_data = data.get("resultData")      # 默认为None如果字段不存在
            
            #print(f"successed: {successed}")
            #print(f"resultData: {result_data}")
            
        except json.JSONDecodeError:
            print("响应不是有效的JSON格式")
    else:
        print(f"请求失败，状态码: {response.status_code}")

    return successed, result_data

def armConnectState(armID):
    url = f"http://localhost:5000/api/MMSMotor/ConnectedState?mmsType={armID}"
    print(f"Connect State URL {url}")

    # get method
    response = s.get(url=url, headers=headers)

    print(f"Arm Connected State Response: {response.text}")

    successed, result_data = responseHandle(response)

    if successed:
        x_state = result_data.get("xMotorIsConnected")
        y_state = result_data.get("yMotorIsConnected")
        z_state = result_data.get("yMotorIsConnected")
    else:
        x_state = False
        y_state = False
        z_state = False
    
    print(f"Arm Connect state: ({x_state}, {y_state}, {z_state})")

    return successed, x_state, y_state, z_state

def armWorkingState(armID):
    url = f"http://localhost:5000/api/MMSMotor/WorkingState?mmsType={armID}"
    print(f"Working State URL {url}")

    # get method
    response = s.get(url=url, headers=headers)

    print(f"Arm Working State Response: {response.text}")

    successed, result_data = responseHandle(response)

    if successed:
        x_state = result_data.get("xMotorIsWorking")
        y_state = result_data.get("yMotorIsWorking")
        z_state = result_data.get("zMotorIsWorking")
    else:
        x_state = False
        y_state = False
        z_state = False
    
    print(f"Arm Working state: ({x_state}, {y_state}, {z_state})")

    return successed, x_state, y_state, z_state

def armGetMotorPos(armID):
    url = f"http://localhost:5000/api/MMSMotor/GetMotorPosition?mmsType={armID}"
    # print(f"Get Motor Pos URL {url}")

    # get method
    response = s.get(url=url, headers=headers)

    # print(f"Arm Motor Pos Response: {response.text}")

    successed, result_data = responseHandle(response)

    if successed:
        x_steps = result_data.get("xSteps")
        y_steps = result_data.get("ySteps")
        z_steps = result_data.get("zSteps")
    else:
        x_steps = False
        y_steps = False
        z_steps = False
    
    # print(f"Arm Working steps: ({x_steps}, {y_steps}, {z_steps})")

    return successed, x_steps, y_steps, z_steps

def armMovebyPos(armID, motorType, speed, step):
    url = f"http://localhost:5000/api/MMSMotor/MotorMoveByDisplacementMode?mmsType={armID}&motorType={motorType}&speed={speed}&step={step}"
    # print(f"Arm Move URL {url}")
    #init_time = time.time()
    # post method
    response = s.post(url=url, headers=headers)
    
    # print(f"Arm Move Response: {response.text}")

    successed, result_data = responseHandle(response)
    #end_time = time.time()
    #print("controller time cost: ", end_time-init_time)
    
    return successed, result_data


# test
if __name__ == "__main__":
    # armConnectState(2)
    # armWorkingState(1)
    # armGetMotorPos(1)
    # armGetMotorPos(2)
    armMovebyPos(1, 1, 10, -122)