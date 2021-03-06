import logging
import subprocess
import time
from collections import defaultdict, namedtuple
import sys
import ray
import traceback

from deephyper.evaluator.evaluate import Evaluator
import os

logger = logging.getLogger(__name__)


def compute_objective(func, x):
    return func(x)


class RayFuture:
    FAIL_RETURN_VALUE = Evaluator.FAIL_RETURN_VALUE

    def __init__(self, func, x, num_cpus=1, num_gpus=None, **kwargs):
        # print('RayFuture === ',  kwargs)
        cwd = kwargs['cwd']
        os.chdir(cwd)
        
        print('RayFuture xxx {x}')
        self.compute_objective = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(
            compute_objective
        )

        x['cwd'] = cwd        
        self.id_res = self.compute_objective.remote(func, x)
        self._state = "active"
        self._result = None

    def _poll(self):
        if not self._state == "active":
            return

        id_done, _ = ray.wait([self.id_res], num_returns=1, timeout=0.001)

        if len(id_done) == 1:
            try:
                self._result = ray.get(id_done[0])
                self._state = "done"
            except Exception as e:
                # print('hyliu eeexception failed ray in _poll ', e)
                traceback.print_exc()
                self._state = "failed"
        else:
            self._state = "active"

    def result(self):
        if not self.done:
            self._result = {
                'result' : self.FAIL_RETURN_VALUE,
                'node'   : '---'
            }

        return self._result

    def cancel(self):
        pass  # NOT AVAILABLE YET

    @property
    def active(self):
        self._poll()
        return self._state == "active"

    @property
    def done(self):
        self._poll()
        return self._state == "done"

    @property
    def failed(self):
        self._poll()
        return self._state == "failed"

    @property
    def cancelled(self):
        self._poll()
        return self._state == "cancelled"


class RayEvaluator(Evaluator):
    """The RayEvaluator relies on the Ray (https://ray.readthedocs.io) package. Ray is a fast and simple framework for building and running distributed applications.

    Args:
        redis_address (str, optional): The "IP:PORT" redis address for the RAY-driver to connect on the RAY-head.
    """

    WaitResult = namedtuple("WaitResult", ["active", "done", "failed", "cancelled"])

    def __init__(
        self,
        run_function,
        cache_key=None,
        ray_address=None,
        ray_password=None,
        num_cpus_per_task=1,
        num_gpus_per_task=None,
        **kwargs,
    ):
        super().__init__(run_function, cache_key, **kwargs)

        import socket
        logger.info(f"RAY Evaluator init: redis-address={ray_address}  {socket.gethostname()} ")

        if not ray_address is None:
            proc_info = ray.init(address=ray_address, _redis_password=ray_password)
        else:
            proc_info = ray.init()

        self.num_cpus_per_tasks = num_cpus_per_task
        self.num_gpus_per_tasks = num_gpus_per_task

        print('ray.nodes() = ', ray.nodes())
        print('print(ray.cluster_resources())  = ', ray.cluster_resources()) 

        self.num_cpus = int(
            sum([node["Resources"].get("CPU", 0) for node in ray.nodes()])
        )
        self.num_gpus = int(
            sum([node["Resources"].get("GPU", 0) for node in ray.nodes()])
        )
        self.num_workers = self.num_cpus // self.num_cpus_per_tasks

        logger.info(
            f"RAY Evaluator will execute: ' {self._run_function.__module__}   {self._run_function}', proc_info: {proc_info}"
        )

        self._cwd = kwargs.get('cwd', os.getcwd())
        

    def _eval_exec(self, x: dict):
        assert isinstance(x, dict)
        future = RayFuture(
            self._run_function, x, self.num_cpus_per_tasks, self.num_gpus_per_tasks, cwd=self._cwd
        )
        return future

    @staticmethod
    def _timer(timeout):
        if timeout is None:
            return lambda: True
        else:
            timeout = max(float(timeout), 0.01)
            start = time.time()
            return lambda: (time.time() - start) < timeout

    def wait(self, futures, timeout=None, return_when="ANY_COMPLETED"):
        assert return_when.strip() in ["ANY_COMPLETED", "ALL_COMPLETED"]
        waitall = bool(return_when.strip() == "ALL_COMPLETED")

        num_futures = len(futures)
        active_futures = [f for f in futures if f.active]
        time_isLeft = self._timer(timeout)

        if waitall:

            def can_exit():
                return len(active_futures) == 0

        else:

            def can_exit():
                return len(active_futures) < num_futures

        while time_isLeft():
            if can_exit():
                break
            else:
                active_futures = [f for f in futures if f.active]
                time.sleep(0.04)

        if not can_exit():
            raise TimeoutError(
                f"{timeout} sec timeout expired while "
                f"waiting on {len(futures)} tasks until {return_when}"
            )

        results = defaultdict(list)
        for f in futures:
            results[f._state].append(f)
        return self.WaitResult(
            active=results["active"],
            done=results["done"],
            failed=results["failed"],
            cancelled=results["cancelled"],
        )
