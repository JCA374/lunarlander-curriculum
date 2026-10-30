# Episode 06: A lost night, and a flaw in my own obstacle

Good morning. This episode has two confessions in it, and one fix you approved this morning.

The first confession. Last night at a quarter past nine, the training crashed. And nothing noticed until this morning. That's about eleven hours of computer time lost. The cause was a bug in my code for resuming a run. Every new neuron in NEAT gets a number. After a resume, the counter for those numbers started again from the highest number in one network, but another network already had a higher one. Two neurons with the same number, and neat python stopped with an error. It is fixed, with a test. And because the regular check ins also didn't run overnight, training now runs under a supervisor. If it ever crashes again, the supervisor restarts it from its last save within half a minute, and the crash shows up on the project page.

Before the crash, the run had settled at about seventy one percent landed on the real task, a little below the hand written pilot. So I tried giving the pilot itself a tune up, on the real task only. That gained very little, about three points, and it got worse in plain wind. The pilot's settings had reached their limit.

That made me suspicious of the task itself. So I mapped where the failures happen, by the boulder's height and position. And that's the second confession. The boulder is my design. And in about one flight in seven, I had placed its top within two and a half metres of where the lander starts. The lander begins with a random push downwards, and it fell onto the rock within a second or two. I tried the hardest escape I could think of, full main engine straight away while pushing sideways. Even that still hit the rock in about two flights out of a hundred. With ninety eight percent as the goal, that leaves no room for any other mistake. The obstacle was close to unfair.

I asked you, and you chose to fix it. The boulder's top now always stays at least three and a half metres below the start. The same crude escape now fails in well under one flight in a hundred, and a smarter pilot should avoid nearly all of those.

There's an honest side effect. With the rock lower down, it crowds the approach to the pad. On the new layout the pilot lands about seventy two percent of the time. It now lands off the pad more often, about one flight in seven. The evolved network sits at the same level. So the task is fairer, but no easier.

The run carries on from where it was, same population, now on the corrected task. And from here on, every real task number on the page is measured on the new layout.

What's next is the hard part. Evolution has kept the pilot's skill, but it hasn't yet beaten it on the real task. Five days remain until the deadline. I'll keep working on what actually fails. Landing off the pad after going around the rock, and being blown sideways off the screen. And I'll tell you plainly how far it gets.
