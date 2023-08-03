import psutil

def get_machine_resource_usage():
    """
    Function returns configuration resource 
    that are consumed by the web application
    """
    return {
        "Number of CPU's": psutil.cpu_count(),
        "CPU usage in %": psutil.cpu_percent(),
        "RAM used (GB)": psutil.virtual_memory()[3] / 1000.000000
    }