# Instructions for llama250.cpp

This repository is a fork of `ggml-org/llama.cpp` (Naster17/llama250.cpp).
AI-generated code is allowed. You are responsible for understanding every line you ship.

## Working Principles

- Understand the code fully - be able to explain any change without AI assistance.
- Prefer reusing existing infrastructure over introducing new subsystems.
- Avoid invasive changes that risk breaking existing behavior.
- Before writing code, read relevant files and follow existing patterns.
- If a change is large or introduces a new pattern, pause and confirm with the user before proceeding.

## Code and Commit Standards

- Avoid emdash `—`, unicode arrow `→`, or other unicode characters: `×`, `…`; use ASCII: `-`, `->`, `x`, `...`
- Code comments:
    - Keep comments concise (usually 1-2 lines).
    - Avoid redundant or excessive inline commentary.
    - Do not hard-wrap comments to a fixed column width.
    - Use simple, clear technical English.
- Do NOT split a line into multiple lines mid-sentence, and do NOT force lines to a fixed character count.
- Write code first, then add comments only where they are needed.
- Llama.cpp does not use Minja; it has a dedicated Jinja engine in `common/jinja`.

Before implementing a new feature, read [skills/code-review/SKILL.md](skills/code-review/SKILL.md).

### Commit Messages

When the user asks you to commit on their behalf, use `Assisted-by: <assistant name>` in the commit message. Do NOT use `Co-authored-by:`.

```
// GOOD: concise commit

llama : fix KV being cleared during context shift

Assisted-by: opencode
```

## Examples

Code comments:

```cpp
// GOOD (code is self-explanatory, no comment needed)

n_ctx = read_metadata("context_length", 1024);


// BAD (too verbose, restates what the code already says)

// Populate the n_ctx from metadata key name "context_length", default to 1024 if the key doesn't exist
n_ctx = read_metadata("context_length", 1024);
```

```cpp
// GOOD (explains a non-obvious invariant)

accept();
bool has_client = listen(idle_interval);
if (has_client) {
  task_queue->on_idle(); // also signal child disconnection
}


// BAD (too verbose, restates what the code already says)

// Instead of blocking indefinitely on accept(), the server polls the listening socket with idle_interval as a timeout. If no new client connects within that interval, it fires task_queue->on_idle() and loops back
```

```cpp
// GOOD (generic, useful to any future reader)

// reset here, as we will release the slot below
n_tokens = 0;
// ... (a lot of code)
release();


// BAD (addresses the user's task, meaningless out of context)

// Reset n_tokens to 0 before releasing the slot. This fixes the problem you mentioned where "phantom" content gets preserved across multiple requests.
n_tokens = 0;
```

```cpp
// GOOD (code is copied from another place; context is already clear, no comment added)

ggml_tensor * inp_pos = build_inp_pos();

// BAD (code copied from elsewhere - do not add comments that weren't there originally)

// inp_pos - contains the positions
ggml_tensor * inp_pos = build_inp_pos();
```

Commands:

```sh
# GOOD: commands that gather context
gh search issues
gh search prs
grep ...
```

## Useful Resources

To conserve context space, load these resources as needed:

Skills: reusable task workflows live in the [skills/](skills/) directory - check there for a skill matching your task before starting.

General documentation:
- [Contributing guidelines](CONTRIBUTING.md)
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [PR template](.github/pull_request_template.md)

Server:
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md)

Chat template and parser:
- [PEG parser](docs/development/parsing.md)
- [Auto parser](docs/autoparser.md)
- [Jinja engine](common/jinja/README.md)

Testing:
- [Slice model](skills/slice-model/SKILL.md)
