# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise repository review helpers without network access."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "scripts"


class HelperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        gh = self.bin / "gh"
        gh.write_text("""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ['FIXTURES'])
if os.environ.get('FAIL_GH'):
    print('GitHub unavailable', file=sys.stderr)
    raise SystemExit(1)
if args[:2] == ['pr', 'list']:
    print((root / 'prs.json').read_text())
elif args[:2] == ['pr', 'diff']:
    print((root / 'diff').read_text(), end='')
elif args[0] == 'api' and args[1].startswith('search/issues?'):
    print('1')
elif args[0] == 'api' and args[1].endswith('/comments'):
    print((root / 'comments.json').read_text())
elif args[0] == 'api' and args[1].endswith('/reviews'):
    print('0')
else:
    raise SystemExit('Unexpected gh arguments: ' + repr(args))
""")
        gh.chmod(0o755)
        self.env = dict(
            os.environ,
            PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            FIXTURES=str(self.root),
            PYTHONDONTWRITEBYTECODE="1",
        )

    def run_script(self, script, *args, data=None):
        return subprocess.run(
            ["bash", str(script), *args],
            input=data,
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_select_filters_drafts_wip_and_own_prs(self):
        prs = [
            {"number": i, "title": title, "author": {"login": author}, "isDraft": draft}
            for i, (title, author, draft) in enumerate(
                [
                    ("Fix bug", "author", False),
                    ("[WIP] Feature", "author", False),
                    ("[Draft] Feature", "author", False),
                    ("Don't merge yet", "author", False),
                    ("Feature", "reviewer", False),
                    ("Feature", "author", True),
                ],
                1,
            )
        ]
        (self.root / "prs.json").write_text(json.dumps(prs))
        result = self.run_script(REVIEW / "select_prs.sh", "--reviewer", "reviewer")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([pr["number"] for pr in json.loads(result.stdout)], [1])

    def verify(self, comments, diff=None):
        (self.root / "diff").write_text(
            diff
            or (
                "diff --git a/src/example.py b/src/example.py\n"
                "--- a/src/example.py\n+++ b/src/example.py\n"
                "@@ -10,2 +20,2 @@\n-old\n+new\n context\n"
            )
        )
        return self.run_script(
            REVIEW / "verify_line_numbers.sh",
            "1",
            data=json.dumps({"comments": comments}),
        )

    def test_verify_valid_nested_path_and_both_sides(self):
        result = self.verify(
            [
                {"path": "src/example.py", "line": line, "side": side, "body": "Check"}
                for side, line in [("RIGHT", 20), ("RIGHT", 21), ("LEFT", 10)]
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.count("OK:"), 3)

    def test_verify_rejects_outside_hunk_or_missing_file(self):
        for path, line in [("src/example.py", 999), ("missing.py", 20)]:
            with self.subTest(path=path):
                result = self.verify([{"path": path, "line": line, "body": "Check"}])
                self.assertNotEqual(result.returncode, 0)

    def test_verify_deleted_file_on_left(self):
        diff = "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n"
        result = self.verify([{"path": "gone.py", "line": 1, "side": "LEFT", "body": "Check"}], diff)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.verify([{"path": "gone.py", "line": 1, "side": "RIGHT", "body": "Check"}], diff)
        self.assertNotEqual(result.returncode, 0)

    def test_verify_new_file_with_explicit_zero_old_count(self):
        diff = "diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+one\n+two\n"
        result = self.verify([{"path": "new.py", "line": 2, "side": "RIGHT", "body": "Check"}], diff)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.verify([{"path": "new.py", "line": 1, "side": "LEFT", "body": "Check"}], diff)
        self.assertNotEqual(result.returncode, 0)

    def test_verify_checks_multiline_comment_start(self):
        result = self.verify(
            [
                {
                    "path": "src/example.py",
                    "start_line": 1,
                    "start_side": "RIGHT",
                    "line": 20,
                    "side": "RIGHT",
                    "body": "Check",
                }
            ]
        )
        self.assertNotEqual(result.returncode, 0)

    def test_verify_multiple_files_hunks_and_renamed_path(self):
        diff = (
            "diff --git a/old.py b/new.py\n--- a/old.py\n+++ b/new.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
            "@@ -10 +10 @@\n-old\n+new\n"
            "diff --git a/second.py b/second.py\n--- a/second.py\n+++ b/second.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        )
        result = self.verify(
            [
                {"path": path, "line": line, "body": "Check"}
                for path, line in [("new.py", 1), ("new.py", 10), ("second.py", 1)]
            ],
            diff,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.verify([{"path": "second.py", "line": 10, "body": "Check"}], diff)
        self.assertNotEqual(result.returncode, 0)

    def test_verify_quoted_path_and_header_like_content(self):
        diff = (
            'diff --git "a/a\\tb.py" "b/a\\tb.py"\n'
            '--- "a/a\\tb.py"\n+++ "b/a\\tb.py"\n'
            "@@ -1 +1 @@\n--- content\n+++ content\n"
        )
        result = self.verify([{"path": "a\tb.py", "line": 1, "body": "Check"}], diff)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_verify_rejects_malformed_input_and_fetch_failure(self):
        for data in ("not json", '{"comments": null}'):
            result = self.run_script(REVIEW / "verify_line_numbers.sh", "1", data=data)
            self.assertNotEqual(result.returncode, 0)
        self.env["FAIL_GH"] = "1"
        result = self.verify([{"path": "src/example.py", "line": 20, "body": "Check"}])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GitHub unavailable", result.stderr)

    def replies(self, answered):
        comments = [
            {
                "id": 100,
                "user": "reviewer",
                "body": "Please fix",
                "in_reply_to_id": None,
                "created_at": "2026-09-01T00:00:00Z",
                "url": "root",
            },
            {
                "id": 101,
                "user": "author",
                "body": "Fixed",
                "in_reply_to_id": 100,
                "created_at": "2026-09-02T00:00:00Z",
                "url": "reply",
            },
        ]
        if answered:
            comments.append(
                {
                    "id": 102,
                    "user": "reviewer",
                    "body": "Thanks",
                    "in_reply_to_id": 100,
                    "created_at": "2026-09-03T00:00:00Z",
                    "url": "answer",
                }
            )
        (self.root / "comments.json").write_text(json.dumps(comments))
        return self.run_script(REVIEW / "check_replies.sh", "--reviewer", "reviewer")

    def test_replies_ignores_answered_thread(self):
        result = self.replies(answered=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("No unanswered replies found", result.stderr)

    def test_replies_reports_unanswered_without_false_clean_summary(self):
        result = self.replies(answered=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("@author replied", result.stdout)
        self.assertNotIn("No unanswered replies found", result.stderr)


if __name__ == "__main__":
    unittest.main()
