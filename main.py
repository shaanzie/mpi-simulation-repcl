from mpi4py import MPI
import random
import time

from src.message.message import Message
from src.replay_clock.replay_clock import ReplayClock

INTERVAL = 1
EPSILON = 10

def distributed_random_message(msg_interval: int = 1, iterations: int = 10):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    clock = ReplayClock(interval=INTERVAL, pid=rank, epsilon=EPSILON)

    for i in range(iterations):

        target = random.randint(0, size - 1)
        clock.advance(i)

        message = Message(i, rank, clock)

        if target != rank:

            comm.send(message, dest=target, tag=0)
            print(f"Process {rank} sent to process {target}: [seq_no={message.seq_no}]{message.clock}")

        while comm.Iprobe(source=MPI.ANY_SOURCE, tag=0):

            incoming_msg = comm.recv(source=MPI.ANY_SOURCE, tag=0)
            clock.merge(incoming_msg.clock, i)

            print(f"Process {rank} received message: [seq_no={message.seq_no}]{incoming_msg.clock}")

        # Delay between iterations
        time.sleep(msg_interval)

if __name__ == '__main__':
    distributed_random_message()