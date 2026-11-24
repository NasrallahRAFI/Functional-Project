# Design System · Model Comparison Report

This visual world belongs to the offline drone audio research report. It treats the model trail as an editorial evidence record: the selection protocol, model search, test results, uncertainty, and caveats remain visible as one connected argument.

## Surface and audience

- Brand surface, in the studio lane required for this project: premium and specific, never a generic SaaS or corporate template.
- Readers are the project owner and technical reviewers comparing frozen model results.
- The report has five shared sections and one dedicated detail page for each of the 14 frozen model entries. Local navigation and same-folder assets keep it offline; no remote fonts, scripts, or data requests are used.

## Visual language

- Light mode uses a cool mineral background, near-black green ink, deep teal data marks, and restrained lime highlights. Dark mode inverts the surfaces and adjusts chart colors for contrast.
- Headings use a large, compact sans voice. Monospaced type is reserved for thresholds, hashes, counts, and configuration labels.
- Horizontal rules and open editorial spacing group the report. Data charts carry the information; repeated ornamental cards and stock imagery are avoided.
- Use actual project values only. Keep test uncertainty, model seed, frozen threshold, and caveats near the claims they qualify.

## Interaction and motion

- The model comparison ranks validation-selected finalists by validation or locked-test F1, then links to one detailed page per frozen model.
- The Model Lab adds a second horizontal ribbon directly below the report navigation. It links to the comparison and all 14 model detail pages, with the active page marked.
- Detail pages show candidate ledgers, ROC and precision-recall curves, a live threshold view of the frozen score vector, score distributions, paired intervals, and fit footprint. CNN stage tabs expose the selection path.
- Motion is limited to a single reveal treatment and chart growth. Use the custom exponential ease-out, and remove movement under `prefers-reduced-motion` while keeping content and state available.

## Accessibility and responsive behavior

- Keep semantic headings, named regions, keyboard operation, visible focus, and labeled controls.
- At narrow widths, stack the report, preserve chart scrolling inside its own frame, and prevent page-level horizontal overflow.
- The light/dark switch must preserve readable text and data marks in both themes.
