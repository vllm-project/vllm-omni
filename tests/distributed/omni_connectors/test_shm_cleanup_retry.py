# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real SHM ownership and the adapter's ordinary maintenance-loop contract."""

import _posixshmem
import errno
import fcntl
import os
import threading
import time
import unittest
import uuid
from multiprocessing import shared_memory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_omni.distributed.omni_connectors.connectors import shm_connector
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.transfer_adapter.base import OmniTransferAdapterBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestSharedMemoryCleanupRetry(unittest.TestCase):
    def setUp(self):
        self.connector = SharedMemoryConnector({"stage_id": 0})
        self.keys = []
        self.adapter = None

    def tearDown(self):
        if self.adapter is not None:
            self.adapter.shutdown()
            self.adapter.save_thread.join(timeout=2)
            self.adapter.recv_thread.join(timeout=2)
        self.connector.close()
        for key in self.keys:
            try:
                segment = shared_memory.SharedMemory(name=key)
                segment.unlink()
                segment.close()
            except FileNotFoundError:
                pass
            try:
                os.unlink(f"/dev/shm/shm_{key}_lockfile.lock")
            except FileNotFoundError:
                pass

    def put(self, value="payload", *, key=None):
        key = key or f"cleanup_retry_{uuid.uuid4().hex}"
        self.keys.append(key)
        self.assertTrue(self.connector.put("0", "1", key, value)[0])
        return key

    def start_sender(self):
        # The real base adapter starts its ordinary idle send/receive loops.
        # No queued model work or custom maintenance callback is supplied.
        self.adapter = OmniTransferAdapterBase.__new__(OmniTransferAdapterBase)
        self.adapter.connector = self.connector
        self.adapter.__init__(None)

    def test_failed_cleanup_stays_owned_and_idle_sender_retries(self):
        key = self.put()
        peer = self.put("peer", key=f"{key}_suffix")
        original_inode = os.stat(f"/dev/shm/{key}").st_ino
        real_unlink = _posixshmem.shm_unlink
        attempts = []

        def fail_once(name):
            if name.lstrip("/") == key:
                attempts.append(time.monotonic())
                if len(attempts) == 1:
                    raise PermissionError(errno.EACCES, "one-shot cleanup failure", name)
            return real_unlink(name)

        with patch.object(_posixshmem, "shm_unlink", fail_once):
            self.connector.cleanup(key)
            self.assertIn(key, self.connector._pending_keys)
            self.assertEqual(os.stat(f"/dev/shm/{key}").st_ino, original_inode)
            self.assertTrue(os.path.exists(f"/dev/shm/shm_{key}_lockfile.lock"))
            self.start_sender()
            deadline = time.monotonic() + 2
            while os.path.exists(f"/dev/shm/{key}") and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(os.path.exists(f"/dev/shm/{key}"), "idle adapter never retried failed cleanup")
            self.assertNotIn(key, self.connector._pending_keys)
            self.assertEqual(len(attempts), 2)
            self.assertEqual(self.connector.get("0", "1", peer)[0], "peer")

    def test_persistent_failure_is_bounded_and_close_flushes_once(self):
        key = self.put()
        real_unlink = _posixshmem.shm_unlink
        attempts = []
        clock = [100.0]

        def fail_target(name):
            if name.lstrip("/") == key:
                attempts.append(clock[0])
                raise PermissionError(errno.EACCES, "persistent cleanup failure", name)
            return real_unlink(name)

        with (
            patch.object(_posixshmem, "shm_unlink", fail_target),
            patch.object(shm_connector, "time", SimpleNamespace(monotonic=lambda: clock[0])),
        ):
            self.connector.cleanup(key)
            for _ in range(20):
                self.connector.cleanup(key)
                self.connector.reap_consumed()
            self.assertEqual(attempts, [100.0], "duplicate cleanup bypassed retry backoff")
            clock[0] = 100.11
            self.connector.reap_consumed()
            clock[0] = 100.32
            self.connector.reap_consumed()
            self.assertEqual(len(attempts), 3)
            clock[0] = 1000.0
            for _ in range(20):
                self.connector.cleanup(key)
                self.connector.reap_consumed()
            self.assertEqual(len(attempts), 3, "automatic retries exceeded their budget")
            self.assertIn(key, self.connector._pending_keys)
            self.assertTrue(os.path.exists(f"/dev/shm/shm_{key}_lockfile.lock"))
            self.connector.close()
            self.assertEqual(len(attempts), 4, "close must make one immediate flush attempt")
            self.assertIn(key, self.connector._pending_keys)
        receiver = SharedMemoryConnector({"stage_id": 1})
        try:
            self.assertEqual(receiver.get("0", "1", key)[0], "payload")
            self.connector.reap_consumed()
            self.assertNotIn(key, self.connector._pending_keys)
        finally:
            receiver.close()

    def test_retry_does_not_unlink_another_producers_reused_key(self):
        key = self.put("old generation")
        first_inode = os.stat(f"/dev/shm/{key}").st_ino
        real_unlink = _posixshmem.shm_unlink
        failed = False
        clock = [100.0]
        replacement = SharedMemoryConnector({"stage_id": 0})

        def fail_once(name):
            nonlocal failed
            if name.lstrip("/") == key and not failed:
                failed = True
                raise PermissionError(errno.EACCES, "one-shot cleanup failure", name)
            return real_unlink(name)

        try:
            with (
                patch.object(_posixshmem, "shm_unlink", fail_once),
                patch.object(shm_connector, "time", SimpleNamespace(monotonic=lambda: clock[0])),
            ):
                self.connector.cleanup(key)
                self.assertTrue(replacement.put("0", "1", key, "new generation")[0])
                second_inode = os.stat(f"/dev/shm/{key}").st_ino
                self.assertNotEqual(first_inode, second_inode)
                clock[0] = 101.0
                self.connector.reap_consumed()
                self.connector.close()
                self.assertTrue(os.path.exists(f"/dev/shm/{key}"), "old cleanup removed a replacement generation")
                self.assertEqual(os.stat(f"/dev/shm/{key}").st_ino, second_inode)
                self.assertNotIn(key, self.connector._pending_keys)
                self.assertEqual(replacement.get("0", "1", key)[0], "new generation")
        finally:
            replacement.close()

    def test_consumer_unlink_before_retry_and_duplicate_cleanup_are_safe(self):
        key = self.put()
        receiver = SharedMemoryConnector({"stage_id": 1})
        real_unlink = _posixshmem.shm_unlink
        attempts = []

        def fail_once(name):
            if name.lstrip("/") == key:
                attempts.append(name)
                if len(attempts) == 1:
                    raise PermissionError(errno.EACCES, "one-shot cleanup failure", name)
            return real_unlink(name)

        try:
            with patch.object(_posixshmem, "shm_unlink", fail_once):
                self.connector.cleanup(key)
                result = receiver.get("0", "1", key)
                self.assertIsNotNone(result, "failed cleanup made the existing payload unreadable")
                self.assertEqual(result[0], "payload")
                self.connector.reap_consumed()
                for _ in range(3):
                    self.connector.cleanup(key)
                self.assertNotIn(key, self.connector._pending_keys)
                self.assertEqual(len(attempts), 2)
                self.assertFalse(os.path.exists(f"/dev/shm/shm_{key}_lockfile.lock"))
        finally:
            receiver.close()

    def test_unexpected_cleanup_error_propagates_without_automatic_retry(self):
        key = self.put()
        with patch.object(_posixshmem, "shm_unlink", side_effect=RuntimeError("unexpected program error")):
            with self.assertRaisesRegex(RuntimeError, "unexpected program error"):
                self.connector.cleanup(key)
            self.connector.reap_consumed()
        self.assertIn(key, self.connector._pending_keys)

    def test_cleanup_and_replacement_writer_share_the_same_lock(self):
        self._check_waiting_writer("cleanup")

    def test_consumer_and_replacement_writer_share_the_same_lock(self):
        self._check_waiting_writer("consume")

    def _check_waiting_writer(self, operation):
        key = self.put("old generation")
        writer = SharedMemoryConnector({"stage_id": 0})
        receiver = SharedMemoryConnector({"stage_id": 1})
        unlink_entered = threading.Event()
        release_unlink = threading.Event()
        writer_opened_lock = threading.Event()
        writer_done = threading.Event()
        put_results = []
        read_results = []
        errors = []
        real_unlink = _posixshmem.shm_unlink
        real_flock = fcntl.flock

        def pause_unlink(name):
            if name.lstrip("/") == key and threading.current_thread().name == "old-generation":
                unlink_entered.set()
                if not release_unlink.wait(2):
                    raise TimeoutError("test did not release old unlink")
            return real_unlink(name)

        def observe_flock(fd, mode):
            if threading.current_thread().name == "replacement-writer" and mode == fcntl.LOCK_EX:
                writer_opened_lock.set()
            return real_flock(fd, mode)

        def old_generation():
            try:
                if operation == "cleanup":
                    self.connector.cleanup(key)
                else:
                    read_results.append(receiver.get("0", "1", key))
            except BaseException as exc:
                errors.append(exc)

        def new_generation():
            try:
                put_results.append(writer.put("0", "1", key, "replacement generation"))
            except BaseException as exc:
                errors.append(exc)
            finally:
                writer_done.set()

        old = threading.Thread(target=old_generation, name="old-generation")
        new = threading.Thread(target=new_generation, name="replacement-writer")
        try:
            with (
                patch.object(_posixshmem, "shm_unlink", pause_unlink),
                patch.object(fcntl, "flock", observe_flock),
            ):
                old.start()
                self.assertTrue(unlink_entered.wait(2))
                new.start()
                self.assertTrue(writer_opened_lock.wait(2))
                self.assertFalse(writer_done.wait(0.05), "writer bypassed the old generation's unlink lock")
                release_unlink.set()
                old.join(2)
                new.join(2)
            self.assertFalse(old.is_alive())
            self.assertFalse(new.is_alive())
            self.assertFalse(errors)
            self.assertTrue(put_results[0][0])
            if operation == "consume":
                self.assertEqual(read_results[0][0], "old generation")
            self.assertTrue(os.path.exists(f"/dev/shm/shm_{key}_lockfile.lock"))
            self.assertEqual(writer.get("0", "1", key)[0], "replacement generation")
            self.connector.reap_consumed()
            self.assertNotIn(key, self.connector._pending_keys)
        finally:
            release_unlink.set()
            old.join(2)
            if new.ident is not None:
                new.join(2)
            receiver.close()
            writer.close()

    def test_cleanup_never_waits_for_a_busy_transfer_lock(self):
        key = self.put()
        lock_path = f"/dev/shm/shm_{key}_lockfile.lock"
        with open(lock_path, "rb+") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            started = time.monotonic()
            self.connector.cleanup(key)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertIn(key, self.connector._pending_keys)
            self.assertTrue(os.path.exists(f"/dev/shm/{key}"))
            self.assertIsNone(self.connector.get_with_deadline("0", "1", key, deadline=time.monotonic() + 1))
        self.connector.close()
        self.assertNotIn(key, self.connector._pending_keys)
        self.assertFalse(os.path.exists(f"/dev/shm/{key}"))

    def test_same_producer_reuse_does_not_inherit_old_cleanup_request(self):
        key = self.put("old")
        real_unlink = _posixshmem.shm_unlink
        failed = False

        def fail_once(name):
            nonlocal failed
            if name.lstrip("/") == key and not failed:
                failed = True
                raise PermissionError(errno.EACCES, "one-shot cleanup failure", name)
            return real_unlink(name)

        with patch.object(_posixshmem, "shm_unlink", fail_once):
            self.connector.cleanup(key)
            self.put("new", key=key)
            self.start_sender()
            time.sleep(0.4)
            self.assertEqual(self.connector.get("0", "1", key)[0], "new")

    def test_consumer_returns_payload_when_lock_removal_fails(self):
        key = self.put("already consumed")
        receiver = SharedMemoryConnector({"stage_id": 1})
        lock_path = f"/dev/shm/shm_{key}_lockfile.lock"
        real_remove = os.remove
        attempts = []

        def fail_once(path):
            if path == lock_path:
                attempts.append(path)
                if len(attempts) == 1:
                    raise PermissionError(errno.EACCES, "one-shot lock removal failure", path)
            return real_remove(path)

        try:
            with patch.object(os, "remove", fail_once):
                result = receiver.get("0", "1", key)
                self.assertIsNotNone(result, "lock-file removal discarded an already consumed payload")
                self.assertEqual(result[0], "already consumed")
                self.assertFalse(os.path.exists(f"/dev/shm/{key}"))
                self.assertTrue(os.path.exists(lock_path))
                self.connector.reap_consumed()
                self.assertFalse(os.path.exists(lock_path))
                self.assertNotIn(key, self.connector._pending_keys)
                self.assertEqual(len(attempts), 2)
        finally:
            receiver.close()

    def test_consumer_returns_payload_when_post_read_identity_check_fails(self):
        key = self.put("already consumed")
        shm_path = f"/dev/shm/{key}"
        lock_path = f"/dev/shm/shm_{key}_lockfile.lock"
        real_identity = shm_connector._identity
        failed = False

        def fail_once(path):
            nonlocal failed
            if path == shm_path and not failed:
                # get's first identity lookup for the segment is after the
                # actual SHM helper has read and unlinked its payload.
                self.assertFalse(os.path.exists(shm_path))
                failed = True
                raise PermissionError(errno.EACCES, "one-shot post-read stat failure", path)
            return real_identity(path)

        with patch.object(shm_connector, "_identity", fail_once):
            result = self.connector.get("0", "1", key)
            self.assertTrue(failed, "fault did not reach the post-read observation")
            self.assertIsNotNone(result, "post-read stat failure discarded an already consumed payload")
            self.assertEqual(result[0], "already consumed")
            self.assertIn(key, self.connector._pending_keys)
            self.assertTrue(os.path.exists(lock_path))
        self.connector.reap_consumed()
        self.assertNotIn(key, self.connector._pending_keys)
        self.assertFalse(os.path.exists(lock_path))


if __name__ == "__main__":
    unittest.main()
