import signal
import ctypes

def set_pdeathsig(sig = signal.SIGTERM):
    """ Used for sending signals to subprocess when parent process dies. (used as parameter in Popen) """
    libc = ctypes.CDLL("libc.so.6")
    def callable():
        return libc.prctl(1, sig)
    return callable