# Project workspace instructions

Read START_HERE.md and repository/docs/PROJECT.md before editing. Work only in the explicitly selected study workspace. V3 is active; V4 is a scope scaffold. Study instructions live in that workspace's AGENTS.md and STUDY.md. Never edit a different study's outputs through compatibility junctions. Register corpus, split, cache, and release identities before connecting them to a new study. Preserve existing evaluation locks and historical hashes. Machine paths, private recordings, datasets, checkpoints, environments, and browser state stay outside Git.

The owner's website build standard and automatic GitHub publishing preference continue to apply. UI source must follow the project's Impeccable and motion instructions. Publish only task-owned changes after the applicable checks pass.

## Owner website build standard

For a website, landing page, portfolio, storefront, client-facing page, or creation/editing of user-facing .html, .css, .scss, .jsx, .tsx, .vue or .svelte files:

1. If this project's .codex/hooks.json is absent, run `npx impeccable install --providers=codex --scope=project`.
2. Run /impeccable init. Mark the surface as brand. Product lane: premium web presence for content creators and local businesses — Designs of Desire studio; never generic SaaS or corporate templates. Anti-references: cream with heavy serif and terracotta/rust; near-black with a single neon accent; centered gradient-blob hero and pill buttons; Inter/Poppins/Roboto for both display/body; repeated three-card grids; 01/02/03 on non-sequential content; generic stock photography; star-rating testimonial carousels; four-column footers on one-page sites. Decide palette and typography from the specific brief.
3. Build to the plan and resolve design-hook findings before continuing.
4. Before completion run /impeccable audit and capture desktop/mobile screenshots with chrome-devtools-mcp.
5. Apply emil-design-eng guidance for easing and timing of hover states, transitions and motion.

## Owner GitHub publishing preference

When completing or continuing project work, publish the completed task automatically after required verification passes. Use the current task branch and existing GitHub remote. If none exists, use authenticated GitHub CLI to create a private repository named after the project; never create a public repository without explicit instruction.

Commit and push only task-owned changes. Preserve unrelated, uncommitted and other-agent work. Never force-push, replace remote history, or push when required checks fail. Exclude credentials, secrets, local build outputs and private user data. If CLI/authentication is missing, remote ownership is ambiguous, or remote history would need replacement, report the exact setup or decision required. Report commit and push results clearly.
