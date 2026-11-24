# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fail a pull request or push whose new text matches a private list of terms.

The list comes from the PUBLIC_REPO_DENYLIST Actions secret (one regex per line, case-insensitive,
lines starting with # ignored), so it never appears in this public repository. Actions logs are
public too, so this script reports only where a term matched, never the term or the matching text.
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass

NULL_SHA = "0" * 40
RECORD_SEPARATOR = "\x1e"
FIELD_SEPARATOR = "\x1f"
DIFF_HEADER = re.compile(r"^diff (?:--git a/.*? b/(.*)|--(?:cc|combined) (.*))$")
HUNK_HEADER = re.compile(r"^(@+) .*?\+(\d+)(?:,\d+)? \1")
# A commit email on one of these is a machine's hostname, not a person's address.
HOSTNAME_EMAIL_SUFFIXES = (".local", ".localdomain", ".lan", ".internal")


@dataclass(frozen=True)
class Excerpt:
    """Text that becomes public, where it comes from, and the file line it starts at (for annotations)."""

    location: str
    text: str
    path: str | None = None
    line: int | None = None


def load_terms(denylist: str) -> re.Pattern[str]:
    """Compile the secret's terms into one pattern, failing on a bad line without echoing it."""
    patterns = []
    for line_number, raw_line in enumerate(denylist.splitlines(), start=1):
        pattern = raw_line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        try:
            re.compile(pattern)
        except re.error:
            sys.exit(f"::error::PUBLIC_REPO_DENYLIST line {line_number} is not a valid regex.")
        patterns.append(f"(?:{pattern})")
    if not patterns:
        sys.exit("::error::PUBLIC_REPO_DENYLIST has no terms.")
    return re.compile("|".join(patterns), re.IGNORECASE)


def git_log(revisions: list[str]) -> str:
    """Messages, identities and `--unified=0` patches of the given commits."""
    log_format = FIELD_SEPARATOR.join(["%H", "%ae", "%ce", "%B"])
    return subprocess.run(
        [
            "git",
            "log",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--patch",
            "--unified=0",
            "--diff-merges=dense-combined",
            f"--format={RECORD_SEPARATOR}{log_format}{FIELD_SEPARATOR}",
            *revisions,
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout


def patch_excerpts(patch: str, prefix: str) -> list[Excerpt]:
    """Return added lines and file names from a `--unified=0` patch, including combined merge diffs."""
    excerpts = []
    path = "?"
    in_hunk = False
    prefix_width = 1
    new_line = 0
    for line in patch.splitlines():
        header = DIFF_HEADER.match(line)
        if header:
            in_hunk = False
            path = header.group(1) or header.group(2)
            excerpts.append(Excerpt(f"{prefix} file name {path}", path))
            continue
        hunk = HUNK_HEADER.match(line)
        if hunk:
            in_hunk = True
            prefix_width = len(hunk.group(1)) - 1
            new_line = int(hunk.group(2))
            continue
        if not in_hunk:
            continue
        markers, content = line[:prefix_width], line[prefix_width:]
        if line.startswith("\\") or "-" in markers:
            continue
        if "+" in markers:
            excerpts.append(Excerpt(f"{prefix} {path}:{new_line}", content, path, new_line))
        new_line += 1
    return excerpts


def commit_excerpts(log_output: str) -> tuple[list[Excerpt], list[str]]:
    """Excerpts to scan, plus commits whose author or committer email is a machine hostname."""
    excerpts = []
    hostname_emails = []
    for record in log_output.split(RECORD_SEPARATOR)[1:]:
        sha, author_email, committer_email, message, patch = record.split(FIELD_SEPARATOR, 4)
        short_sha = sha[:12]
        excerpts.append(Excerpt(f"commit {short_sha} message", message.strip("\n")))
        excerpts.extend(patch_excerpts(patch, f"commit {short_sha}"))
        for role, email in (("author", author_email), ("committer", committer_email)):
            domain = email.rpartition("@")[2].lower()
            if "." not in domain or domain.endswith(HOSTNAME_EMAIL_SUFFIXES):
                hostname_emails.append(f"commit {short_sha} {role} email is a machine hostname")
    return excerpts, hostname_emails


def event_excerpts(event_name: str, event: dict) -> tuple[list[Excerpt], list[str]]:
    """Return what this event publishes, and the git revisions holding its new commits."""
    if event_name == "pull_request":
        pull_request = event["pull_request"]
        excerpts = [
            Excerpt("PR title", pull_request["title"] or ""),
            Excerpt("PR body", pull_request["body"] or ""),
            Excerpt("PR branch name", pull_request["head"]["ref"]),
        ]
        return excerpts, [pull_request["head"]["sha"], "--not", pull_request["base"]["sha"]]
    if event_name == "push":
        if event["before"] == NULL_SHA:
            return [], ["-1", event["after"]]
        return [], [event["after"], "--not", event["before"]]
    sys.exit(f"::error::Unsupported event {event_name}.")


def matching_locations(excerpts: list[Excerpt], terms: re.Pattern[str]) -> list[Excerpt]:
    """One excerpt per matching line, with the line number folded into its location."""
    matches = []
    for excerpt in excerpts:
        lines = excerpt.text.splitlines() or [""]
        for line_number, line in enumerate(lines, start=1):
            if terms.search(line):
                location = f"{excerpt.location} line {line_number}" if len(lines) > 1 else excerpt.location
                matches.append(Excerpt(location, "", excerpt.path, excerpt.line))
    return matches


def main() -> int:
    """Scan the triggering event and print only the locations of matches."""
    denylist = os.environ.get("DENYLIST", "")
    if not denylist.strip():
        print(
            "::error::The PUBLIC_REPO_DENYLIST Actions secret is not set or not available to this run "
            "(pull requests from forks don't receive secrets). A maintainer must set it for this check to pass."
        )
        return 1
    terms = load_terms(denylist)
    with open(os.environ["GITHUB_EVENT_PATH"]) as event_file:
        event = json.load(event_file)
    excerpts, revisions = event_excerpts(os.environ["GITHUB_EVENT_NAME"], event)
    commit_texts, problems = commit_excerpts(git_log(revisions))
    excerpts.extend(commit_texts)

    for match in matching_locations(excerpts, terms):
        if match.path is not None:
            print(f"::error file={match.path},line={match.line}::Added line matches a private term ({match.location}).")
        problems.append(f"{match.location} matches a private term")
    for problem in problems:
        print(f"- {problem}")
    if problems:
        print(f"::error::{len(problems)} problem(s) found. Reword the text above; terms are not shown in public logs.")
        return 1
    print(f"Scanned {len(excerpts)} excerpts; no private terms found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
