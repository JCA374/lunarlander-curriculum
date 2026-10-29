# How the work was run: check-ins, changes, a live page and a podcast

This is the working method from the Bipedal CMA-ES project (Oct 2026). It's written so the next project can
use it from day one. The short version: **long training runs in the background; check in every 2 hours;
change one thing at a time, for a written reason; verify every claim on independent tests; publish every
step as a log entry on a web page and the big steps as a podcast episode.**

## 1. Roles

- **The user** sets the goal and the deadline ("a fast walker, one foot in front of the other, by
  Friday 17:00"), gives direction ("try a curriculum", "feel free to change the model") and listens to the
  podcast and reads the page. The user doesn't need to be at the laptop.
- **Claude** runs the experiments, decides what to try next within that direction, and reports honestly,
  including failures and corrections.

## 2. The loop (every 2 hours)

A session cron job (`CronCreate`, every 2 h, also one for the final report at the deadline) sends a
"check-in" prompt. A cron job lasts only for the session, so it has to be set up again after a restart. Each check-in does this:

1. **Look:** is everything still running (`pgrep "[c]urriculum"`, `pgrep "[t]rain.py"`), the stage
   logs, `stats.csv` (best/mean/held-out champion, σ), memory (`free -h`), load.
2. **Judge:** is the run improving, plateaued (the automatic `plateau_watch.sh` rule: no gain ≥5 over 200
   generations after generation 400), or stuck in a trap? A trap is a behaviour that games the reward
   without doing the task, e.g. diving forward, falling early on purpose, standing still, tapping a foot.
3. **Decide:** keep it running, or stop it and start the next attempt. Change one thing at a time, so a
   result can be explained. Write the reason down *before* the result is known.
4. **Record:** a journal entry (`experiments/journal.py "title" "text"`, shown on the site), a line in
   `TRAINING_LOG.md`, and for a new attempt an entry in `docs/data/curriculum_attempts.json` (title,
   note, the exact settings and reward of every stage).
5. **Publish:** the page updates by itself (`curriculum_publish.sh` every 15 min). If something major
   happened, make a podcast episode (§5).
6. **Report here:** a few lines in the chat: what's running, the numbers, what changed and why.

If nothing changed, the check-in says so in one line and changes nothing.

## 3. Experiments: rules that held up

- **Curriculum as a script.** Each attempt is a driver script (`curriculum_X.sh`) with stages written as
  `name|pass check|generation cap|support|training args`. Each stage starts from the previous stage's
  champion, and moves on when its check passes or the cap runs out. The driver is restartable: every stage
  resumes from its checkpoint.
- **Every stage has a written reward and an expected outcome**, shown on the site before the result is in.
- **Verify on independent tests, more than you think.** Champions chosen on 20 tracks looked 15 points better
  than they were (the winner's curse), and it happened three times. The final rule: a goal check on 50 new
  tracks, and any "better than" claim as a paired comparison on 100 tracks with a bootstrap confidence
  interval. Choose champions from the search's mean, not from the luckiest candidate.
- **Archive every champion** (`runs/<run>/champions/genNNNNN_label`). A verified best walker was once
  overwritten and couldn't be recovered.
- **Correct mistakes in public.** When a claimed result turned out wrong, the journal and the next podcast
  said so plainly.
- **Determinism first.** Any hidden state (world reuse in Box2D, the global RNG) breaks common random numbers
  and resume. Find it with a "same seed twice gives the same reward" test.
- **Training wheels.** Helping in training (a support torque that fades out stage by stage) worked better than
  heavy reward shaping. Shaping terms are where the traps come from.
- **Background jobs:** `setsid nohup … &`, SIGTERM saves a checkpoint, and never `pkill -f <pattern>` when the
  pattern also appears in your own command line (use `pgrep "[x]yz"`, then kill the PIDs).
- **The CPU budget:** training gets most of the cores. Narration and video run with `nice`.

## 4. The web page (GitHub Pages from `main:/docs`)

- **Lessons** (one page each: the idea, a diagram, an interactive JS demo, a quiz, a short narrated clip) explain
  the background while training runs.
- **The log page** (newest first): every journal note, stage start/end, pass check, goal check and new
  champion, each with an embedded replay of the agent *as it was at that moment* (a snapshot is taken when the
  event is first seen). A goal table at the top, plus each attempt's stages with the reward and the expected outcome.
- **A local dashboard** (`dashboard.py`, auto-refresh, y-axis autoscale over the selected window, the range kept
  across refreshes) for detail the site doesn't need.
- The deployment counts as done only when `gh api repos/<repo>/pages/builds/latest` reports `built <sha>` and the pages
  return HTTP 200.

## 5. The podcast

- **When:** as soon as something major happens: a breakthrough, a stage passed, a new kind of experiment, a clear
  failure, a correction. No schedule and no daily cap; routine "still training" check-ins get no episode.
- **What:** 2-6 minutes, first person, every number from the logs, written for the ear (short sentences,
  numbers as spoken, no lists/symbols/run ids). Structure: where we were → what happened → what worked → what
  failed and why → what's next. A notes file holds 3-5 show notes.
- **How:** the user-level skill `project-podcast` (`~/.claude/skills/project-podcast/`): a `podcast.conf`
  in the project, then `prepare.sh NN` (voice-text work dir + speech-flow scan) and `publish.sh NN`
  (Kokoro narration, Whisper listen-check, export, feed, commit, push, wait for Pages, check HTTP 200 and that
  the feed lists the episode). It ends with `DEPLOYED <url>` or `FAILED: <reason>`.

## 6. Pausing and continuing

Stop the drivers first, then the trainers (SIGTERM → checkpoint), the publisher, the watchers and the servers. Delete the
cron jobs. Save the final model in a tracked folder (`models/…`) with its verification files. Write
`CONTINUE.md` (the result, exact resume commands, open ideas), add a final journal entry, then commit and push.
