from vllm.v1.core.sched.scheduler import Scheduler as OriginalV1Scheduler

class MiniDragScheduler(OriginalV1Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

original_scheduler = None

def apply_patch():
    import vllm.v1.core.sched.scheduler
    global original_scheduler
    original_scheduler = vllm.v1.core.sched.scheduler.Scheduler
    vllm.v1.core.sched.scheduler.Scheduler = MiniDragScheduler

def revert_patch():
    import vllm.v1.core.sched.scheduler
    vllm.v1.core.sched.scheduler.Scheduler = original_scheduler
