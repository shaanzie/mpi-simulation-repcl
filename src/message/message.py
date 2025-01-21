from src.replay_clock.replay_clock import ReplayClock

class Message:

    def __init__(self, seq_no: int, pid: int, clock: ReplayClock) -> None:
        
        self.seq_no = seq_no
        self.pid = pid
        self.clock = clock
    
    