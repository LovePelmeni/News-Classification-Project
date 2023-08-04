import psutil

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
        'CPU Usage (%)': psutil.cpu_percent(interval=1),
    }

    # Get memory (RAM) information
    memory = psutil.virtual_memory()
    memory_info = {
        'Total Memory (MB)': memory.total >> 20,  # Convert bytes to megabytes
        'Available Memory (MB)': memory.available >> 20,
        'Used Memory (MB)': memory.used >> 20,
        'Memory Usage (%)': memory.percent,
    }

    # Get disk information
    disk = psutil.disk_usage('/')
    disk_info = {
        'Total Disk Space (GB)': disk.total >> 30,  # Convert bytes to gigabytes
        'Used Disk Space (GB)': disk.used >> 30,
        'Free Disk Space (GB)': disk.free >> 30,
        'Disk Usage (%)': disk.percent,
    }
    return {
        'disk_info': disk_info,
        'cpu_info': cpu_info,
        'memory info': memory_info
    }