import psutil
import typing 
from baseline_requirements import system

def check_resource_exceed(
    metric_type: typing.Literal['disk_info', 'memory_info', 'cpu_info'], 
    metrics: typing.Dicts[str, int]) -> bool:
    """
    Function checks for resource exceed for provided metric type
    Args:
        metric_type: str - type of the metric 'disk', 'memory' or 'cpu' 
        metrics: typing.Dict[str, int] - metric information
    """
    if metric_type == "disk_info":
        return metrics['Disk Usage (%)'] <= system.DISK_USAGE

    elif metric_type == "memory_info":
        return metrics['Memory Usage (GB)'] <= system.RAM_USAGE
    
    elif metric_type == "cpu_info":
        return metrics['CPU Usage (GB)'] <= system.CPU_USAGE
    
def get_system_resource_metrics():
    """
    Function returns information about internal hardware
    of the web application 
    """
    # Get CPU information
    cpu_info = {
        'CPU Cores': psutil.cpu_count(logical=False),
        'Total CPU Threads': psutil.cpu_count(logical=True),
        'CPU Frequency (MHz)': psutil.cpu_freq().current,
        'Usage': psutil.cpu_percent(interval=1),
    }

    # Get memory (RAM) information
    memory = psutil.virtual_memory()
    memory_info = {
        'Total Memory (MB)': memory.total >> 20,  # Convert bytes to megabytes
        'Available Memory (MB)': memory.available >> 20,
        'Used Memory (MB)': memory.used >> 20,
        'Usage': memory.percent,
    }

    # Get disk information
    disk = psutil.disk_usage('/')
    disk_info = {
        'Total Disk Space (GB)': disk.total >> 30,  # Convert bytes to gigabytes
        'Used Disk Space (GB)': disk.used >> 30,
        'Free Disk Space (GB)': disk.free >> 30,
        'Usage': disk.percent,
    }
    return {
        'disk_info': disk_info,
        'cpu_info': cpu_info,
        'memory_info': memory_info
    }
