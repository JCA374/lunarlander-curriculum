# Episode 04: A correction, and a fresh start

This episode starts with a correction. In the first episode I told you that NEAT got a head start, a copy of the hand written pilot that landed about eighty two percent of the time. That copy was real. But the network NEAT actually received was not the same as that copy. It was a broken version of it. Let me explain how I found out, because it is a good story about checking your own work.

Here is where we were. The boulder stage had run for three hundred generations. With the boulder growing from small to full size, the best champion climbed from forty to sixty five percent landed. Real progress, but far too slow for ninety eight.

So I asked a simple question. The copy of the pilot, flown directly, how well does it handle the boulder? The answer was seventy one percent. But the same network, converted into NEAT's format at the very start, landed only thirty. In plain wind, ninety two against forty three. The same network should fly the same way. It didn't.

The cause was a single detail. NEAT's output neurons squash every value through a curve called tanh, which flattens out at one. The copied network gives each action a score, and some scores were as large as twenty. Squash twenty and you get one. Squash fifteen, and you also get one, exactly. So two actions were tied, and the computer picked the first one, not the best one. About one step in nine, the lander did the wrong thing. My test had missed it, because it used a small random network whose scores never got that large.

The fix was one line. The output neurons now pass their scores through unchanged. And I changed the test so that it fails without the fix and passes with it. The exact copy now lands about ninety eight percent of the time in full wind, and about seventy seven percent with the boulder. That's on small test sets, so take the numbers as rough. But it's a far better start than anything evolution reached after seven hundred generations from the broken one.

So I'm starting again. And since a restart was forced anyway, three more changes come with it. The network gets two more sensors, a kind of route guide. One says how far the lander is from where it should be heading, beside the boulder or over the pad. The other says how far it is above the boulder's top. Then, the mutations are gentler. I measured that small random changes to the weights cost about ten points with the boulder. That skill is fragile. And last, three in ten training flights in the wind stages now have a boulder in them. The network forgot skills twice when a stage didn't use them. This time selection keeps checking that skill all the way through.

To be clear about what still stands. The earlier stage results were real measurements. The runs really did pass calm air, the steady push and the gusts. They just did it from a handicapped start.

The new run began a few minutes ago, and the first generations land every calm flight. I'll tell you how far the exact copy plus evolution can get. And on the project page there is now a growing explanation of the network itself, what each part does, and how it changed as it evolved.
