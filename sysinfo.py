
import psutil
#from gpiozero import CPUTemperature

def getCPU():
    return psutil.cpu_percent(interval=1)

def getRAM():
    return psutil.virtual_memory()


