# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import functools
import queue
import logging
logging.basicConfig(format='[%(asctime)s] %(filename)s %(funcName)s():%(lineno)i [%(levelname)s] %(message)s', level=logging.DEBUG)
import traceback
from horovod.common.exceptions import HorovodInternalError, HostsUpdatedInterrupt
from horovod.runner.elastic.worker import WorkerNotificationManager


notification_manager = WorkerNotificationManager()


class State(object):
    """State representation used for tracking in memory state across workers.

    Args:
        bcast_object: Function used to broadcast a variable from rank 0 to the other workers.
        get_rank: Function that returns the current rank of this worker.
    """
    def __init__(self, bcast_object, get_rank):
        self._bcast_object = bcast_object
        self._rank = get_rank
        self._host_messages = queue.Queue()
        self._last_updated_timestamp = 0
        self._reset_callbacks = []

    def register_reset_callbacks(self, callbacks):
        """Register callbacks that will be invoked following a reset event (worker added or removed).

        For example, a common use of a reset callback would be to update the learning rate scale with the
        new number of workers.

        Args:
            callbacks: list of functions to execute.
        """
        self._reset_callbacks.extend(callbacks)

    def on_reset(self):
        self._host_messages = queue.Queue()
        self.reset()
        for callback in self._reset_callbacks:
            callback()

    def on_hosts_updated(self, timestamp):
        self._host_messages.put(timestamp)

    def commit(self):
        """Commits all modifications to state tracked by this object to host memory.

        This call will also check for any changes to known hosts, and raise a `HostsUpdatedInterrupt`
        if any were detected.

        Because commits are a heavy operation involving data copy (potentially from GPU to host), it is
        recommended to consider committing less frequently than once per batch. This allows users to tradeoff
        between per-batch execution time and lost training steps in the event of a worker failure.
        """
        self.save()
        self.check_host_updates()

    def check_host_updates(self):
        """Checks that a notification has been sent indicating that hosts can be added or will be removed.

        Raises a `HostsUpdatedInterrupt` if such a notification has been received.
        """
        # Iterate through the update messages sent from the server. If the update timestamp
        # is greater than the last update timestamp, then trigger a HostsUpdatedException.
        last_updated_timestamp = prev_timestamp = self._last_updated_timestamp
        while not self._host_messages.empty():
            timestamp = self._host_messages.get()
            if timestamp > last_updated_timestamp:
                last_updated_timestamp = timestamp

        # In order to ensure all workers raise the exception at the same time, we need to sync
        # the updated state across all the workers.
        # TODO(travis): this should be a max allreduce to account for changes in rank 0
        prev_timestamp, self._last_updated_timestamp = self._bcast_object((prev_timestamp, last_updated_timestamp))

        # At this point, updated state is globally consistent across all ranks.
        if self._last_updated_timestamp > prev_timestamp:
            raise HostsUpdatedInterrupt()

    def save(self):
        """Saves state to host memory."""
        raise NotImplementedError()

    def restore(self):
        """Restores the last committed state, undoing any uncommitted modifications."""
        raise NotImplementedError()

    def sync(self):
        """Synchronize state across workers."""
        raise NotImplementedError()

    def reset(self):
        """Reset objects and variables following a reset event (before synchronization)."""
        pass


class ObjectState(State):
    """State for simple Python objects.

    Every object is specified as a keyword argument, and will be assigned as an attribute.

    Args:
        bcast_object: Horovod broadcast object function used to sync state dictionary.
        get_rank: Horovod rank function used to identify is this process is the coordinator.
        kwargs: Properties to sync, will be exposed as attributes of the object.
    """
    def __init__(self, bcast_object, get_rank, **kwargs):
        self._bcast_object = bcast_object
        self._saved_state = kwargs
        self._set_attrs()
        super(ObjectState, self).__init__(bcast_object=bcast_object, get_rank=get_rank)

    def save(self):
        new_state = {}
        for attr in self._saved_state.keys():
            new_state[attr] = getattr(self, attr)
        self._saved_state = new_state

    def restore(self):
        self._set_attrs()

    def sync(self):
        if self._saved_state:
            self._saved_state = self._bcast_object(self._saved_state)
            self._set_attrs()

    def _set_attrs(self):
        for attr, value in self._saved_state.items():
            setattr(self, attr, value)


def run_fn(func, reset):
    # @functools.wraps(func) python中的装饰器，保证wrapper具有同func相同的属性，例如function name， docstring、argument
    @functools.wraps(func)
    def wrapper(state, *args, **kwargs):
        notification_manager.init() ##
        notification_manager.register_listener(state) ##

        try:
            while True:
                state.sync() # 同步state

                try:
                    return func(state, *args, **kwargs) # 执行训练换算
                except HorovodInternalError:
                    logging.info("HorovodInternalError")
                    logging.info(traceback.format_exc())
                    state.restore()  # 捕获HorovodInternalError异常，重新读取上一次commit的state
                except HostsUpdatedInterrupt:
                    logging.info("HostsUpdatedInterrupt")
                    pass # HostsUpdatedInterrupt 异常，对应着节点变动的异常，仅重启即可

                reset()
                state.on_reset()
        finally:
            notification_manager.remove_listener(state)
    return wrapper
