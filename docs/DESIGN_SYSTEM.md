# Design system: overlay

This document describes the look and behaviour of the live overlay in `overlay/ui/`. All values below are quoted from `overlay/ui/style.css`, `overlay/ui/themes.js` and `overlay/ui/app.js`. Clubs and players shown in the overlay are fictional (Kestrel Vale FC, Port Saltmere United) and the data is synthetic.

## Monochrome glow broadcast look

The base look is black and white. Panels are near-black glass with a hairline white edge and a faint white glow, and text is white or grey.

Core tokens from the `:root` block of `style.css`:

| Token | Value |
| --- | --- |
| `--bg` | `rgba(5, 5, 6, 0.88)` |
| `--bg-solid` | `#050506` |
| `--edge` | `rgba(255, 255, 255, 0.22)` |
| `--edge-strong` | `rgba(255, 255, 255, 0.55)` |
| `--text` | `#f5f5f5` |
| `--muted` | `#9a9a9a` |
| `--dim` | `#5c5c5c` |
| `--glow` | `0 0 14px rgba(255, 255, 255, 0.35)` |
| `--ease` | `cubic-bezier(0.2, 0.7, 0.2, 1)` |
| `--font` | `"Segoe UI Variable Display", "Segoe UI", system-ui, sans-serif` |
| `--hex` | a faint hexagon SVG pattern used by `.hex` at `stroke-opacity='0.07'` |

These are the only custom properties declared in `:root`. `--accent` is not declared there. It is set per cue by `app.js` (`box.style.setProperty("--accent", ...)`), and CSS reads it as `var(--accent, #fff)`, so `#fff` is the fallback.

The timeline declares `--turning: #fbbf24` (turning point diamond, 12px, glow) and `--yours: #22d3ee` (your player diamond, 10px, white ring); supporting diamonds are 7px hollow `var(--muted)`. Size and outline differ as well as colour, and a legend labels all three.

The frame style (`.frame`) combines a 1px `var(--edge)` border, a dark outer ring, a 22px faint white outer glow and an inset glow. The `.frame::before` and `.frame::after` pseudo-elements draw two short accent-coloured bars on opposite corners (top left, 34% wide, and bottom right, 28% wide), each with `drop-shadow(0 0 6px var(--accent, #fff))`. The accent is the only colour in the frame.

Tags (`.tag`) are white with black uppercase text. `.tag.ghost` and `.tag.warn` are outline-only.

## One accent per theme

`overlay/ui/themes.js` is the registry for accents, icons, motion and priority. Each theme has exactly one accent colour. The table is copied from that file.

| Theme key | Label | Accent | Motion | Priority |
| --- | --- | --- | --- | --- |
| `breakthrough` | Breakthrough | `#22d3ee` | `sweep` | medium |
| `pressure_shift` | Pressure shift | `#fb923c` | `jolt` | medium |
| `momentum_swing` | Momentum swing | `#a78bfa` | `swing` | medium |
| `player_spotlight` | Player spotlight | `#facc15` | `spot` | medium |
| `goal_threat` | Goal threat | `#f87171` | `pulse` | high |
| `defensive_masterclass` | Defensive masterclass | `#4ade80` | `lock` | medium |
| `chaos_index` | Chaos index | `#f472b6` | `flicker` | low |
| `control_index` | Control index | `#60a5fa` | `settle` | low |
| `milestone` | Milestone | `#e5e7eb` | `rise` | medium |
| `recap_cue` | Recap | `#cbd5e1` | `settle` | high |

Notes on the table:

- `settle` is named in `themes.js` but has no `@keyframes` or `[data-motion="settle"]` rule in `style.css`. Cues with that motion get no motion effect beyond the base transitions.
- The `priority` field in `themes.js` is not read anywhere in `app.js`. The overlay gates on `cue.priority` from the cue itself (see the priority section below). The registry value is therefore informational in the current UI.
- `milestone` and `recap_cue` use neutral greys (`#e5e7eb`, `#cbd5e1`), so they stay close to the monochrome look.

Icons come from the `icon` field of each theme. The names are `breakthrough`, `pressure`, `wave`, `spotlight`, `target`, `shield`, `chaos`, `control`, `flag` and `recap`. The icon paths are defined in `app.js` (`svg()`), not in `themes.js`.

Stings: `app.js` defines `STINGS` for five themes only: `goal_threat` ("Goal threat"), `momentum_swing` ("Momentum shift"), `player_spotlight` ("Player spotlight"), `control_index` ("Tactical reset") and `chaos_index` ("Tactical reset"). A sting is a short splash in the theme accent (see the splash section).

## Type scale and tokens

Base: `html, body` use `font: 14px/1.4 var(--font)` with `font-variant-numeric: tabular-nums`.

Font sizes actually used in `style.css`, grouped by role:

| Size | Used for |
| --- | --- |
| 9px | `.synthetic` label |
| 10px | `.tag`, field labels (`label`), `.trail`, `.stat .label`, `.q-row .q-meta` |
| 11px | `.feed-head`, `.kicker`, `.legend`, `.foot`, `.s-controls`, panel button labels |
| 12px | `.stat .values i`, `label.check`, `#commentary .note` |
| 13px | Inputs and selects, `.cue-chip`, `#answer`, `#commentary` |
| 14px | Base body text, feed headings (`.entry h3`) |
| 15px | `.tile-label`, `#commentary h3` |
| 17px | Team names (`.team`), `.clock` |
| 18px | `#onboard h2` |
| 19px | `.chips b` |
| 20px | `.stat .values` |
| 22px | `.score em` |
| 24px | `#alert h2` |
| 30px | `.score`, `#alert.goal h2` |
| `clamp(36px, 6vw, 88px)` | `.splash-text` |
| `clamp(20px, 2.6vw, 38px)` | `#splash.sting .splash-text` |

Letter spacing is used as a style device: `0.1em` to `0.16em` on uppercase labels, `0.42em` on the full splash, reduced to `0.14em` once shown.

Font weights used: 300 (scores, stats, splash), 400 (headings in commentary), 500 (titles), 600 (team names, alert titles), 700 (tags).

## Click-through rule

The overlay is a transparent full-screen window. Only elements marked `data-hit` take clicks. Everything else passes through to the desktop.

How it works, from `overlay/ui/app.js` and `overlay/src-tauri/src/main.rs`:

- `index.html` marks the interactive surfaces with `data-hit`: `#dockBtn`, `#panel`, `#askBtn`, `#askPanel`, `#valueBtn`, `#valuePanel`, `#studio` (producer controls), `#history` (Recent cues), `#timeline` (timeline bar and its diamonds) and `#onboard`.
- `reportHitRegions()` in `app.js` collects the bounding rectangles of every `[data-hit]` element that has a non-zero size. It sends them to Rust with `set_hit_regions` only when the list has changed.
- In `main.rs`, the window starts with `set_ignore_cursor_events(true)`. A background thread checks the cursor every 30 ms. It stops ignoring events only when the cursor is inside one of the reported rectangles.
- `Ctrl+Shift+O` toggles a lock (`LOCKED`). When locked, the whole overlay is click-through and a `click-through` event is emitted. `app.js` then adds `body.locked`, which hides `#dock` (`body.locked #dock { opacity: 0; pointer-events: none; visibility: hidden; }`), which also removes the controls from the tab order. The on-screen hint reads "Ctrl+Shift+O hides controls".
- `#lane` and `#splash` set `pointer-events: none`.
- Outside the Tauri build, `body:not(.tauri) .tauri-only { display: none; }` hides elements with the `tauri-only` class, including the Ctrl+Shift+O hint. The plain web build has no click-through.

## Reduced motion

The `@media (prefers-reduced-motion: reduce)` block in `style.css` is:

```
*, *::before, *::after { animation: none !important; transition-duration: 0.01ms !important; }
#alert, .cue-chip, .feed-item, #splash { transform: none !important; filter: none !important; transition: opacity 0.25s linear !important; }
```

So all keyframe animations stop. The moment alert, cue chips, feed items and splash keep only a 0.25s opacity fade and lose their transforms and blurs.

## Mobile behaviour (900px breakpoint)

`@media (max-width: 900px)` switches the overlay to a single column:

- `html, body` allow scrolling (`overflow: auto; height: auto;`). Body becomes a flex column with `gap: 22px` and `padding: 16px`.
- `#scorebar, #dock, #alert, #cam, #feed, #tiles, #timeline, #lane` become `position: static`, lose their transforms, and take `width: 100%`.
- A fixed order is set: scorebar 1, lane 2, alert 2 (with `margin-top: 18px`), cam 3, tiles 4, timeline 4 (after tiles), feed 5, dock 6, studio 7.
- `#alert:not(.show)` is hidden.
- `#studio` becomes static full width with no max height.
- `.stat-wrap { flex: 1 1 140px; }` lets stat tiles wrap.

The stat tiles (`#tiles`) use `flex-wrap: wrap` and `max-width: calc(100vw - 36px)` at all widths, so they wrap rather than clip.

## Priority tiers and viewer depth

Each cue carries a `priority` of `high`, `medium` or `low`. The priority is set by the Python side (`src/football_analytics/overlay_cues.py`, field `priority`), not by `themes.js`.

`app.js` defines the depth filter:

```
const DEPTH_PRIORITIES = { headlines: ["high"], balanced: ["high", "medium"], deep: ["high", "medium", "low"] };
```

`visiblePriorities()` returns the list for `state.prefs.depth`, falling back to `balanced`. `routeCue()` drops any cue whose priority is not in that list (`if (!visiblePriorities().includes(cue.priority)) return;`).

Depth defaults come from the interest dials (`PERSONA_DEPTH` in `app.js`): a mean of 70 or more (analyst) uses `deep`, 25 or less (newcomer) uses `headlines`, and anything else uses `balanced`. The feedback control "Too basic" raises depth one step and "Too technical" lowers it one step (`apply_feedback` in `src/football_analytics/commentary.py`).

Producer view (`?view=studio`) does not use the depth filter. In `routeCue()`, the studio branch (`if (STUDIO) return studioCue(data);`) returns before the `visiblePriorities()` check. Cues go to the producer queue instead, and airing is controlled by the per-priority Auto toggles (defaults: `high` and `medium` on, `low` off) and the minimum confidence slider.

## Files

- `overlay/ui/style.css`
- `overlay/ui/themes.js`
- `overlay/ui/app.js`
- `overlay/ui/index.html`
- `overlay/src-tauri/src/main.rs`
