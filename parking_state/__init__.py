def singleton(cls):
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance

class DeviceSettings(object):
    def __init__(self, *args, **kwargs):
        self.state = {}

@singleton
class Settings(object):
    """docstring for Settings"""
    device = None
    def __init__(self, *args, **kwargs):
        if self.device is None:
            self.device = DeviceSettings()



