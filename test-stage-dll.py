from ctypes import CDLL
import os
 
dll_path = r"StageCPP.dll"  # 替换为实际的 DLL 文件路径
 
# 使用 winmode 参数将值指定为可以从本地路径加载
tem = CDLL(dll_path, winmode=0)
print(tem)