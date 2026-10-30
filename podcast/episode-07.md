# Episode 07: The sensor that was barely sensing

This is the best news so far, and it came from a chart I made for the project page.

You asked for a page about how the lander sees the world. So I plotted two things on top of each other. The real wind, which the lander never gets to see. And the drift sensor, the lander's own estimate of the wind. They should have looked alike. They didn't. The sensor swung far more than the wind ever did. When I measured it properly over forty flights, the drift sensor and the real wind were barely related at all. A match of one would be perfect. This was about one in seven.

The reason was simple once I saw it. The sensor watches how the lander's sideways speed changes. But the lander's own main engine pushes it sideways whenever it tilts, and the sensor counted that as wind. In most steps, the lander's own steering was louder than the wind itself. Every network in this project, and the hand written pilot, had been leaning against a reading that was mostly its own echo.

So I measured, in calm air, exactly how much each engine pushes sideways. The side engines give a small, steady push. The main engine pushes in proportion to the tilt, with a lot of random spread. The new sensor subtracts each engine's known push, and trusts the noisy main engine steps much less. Its match with the real wind is now ninety seven in a hundred.

Then I gave the pilot the new sensor, plus two small rules. Keep detours away from the edges of the screen, and stop falling first when the boulder is close below. And I tuned its settings again on the real task. Last time, tuning gained almost nothing. This time, on six hundred unseen flights, the pilot went from about seventy five percent landed to eighty six. That's nearly twelve points, and the uncertainty range stays well above zero. Landing off the pad dropped by more than half. In plain wind it now lands ninety seven percent of the time.

Next, the copy. The network learned to imitate the new pilot, and as a NEAT starting network it lands about eighty five percent on the real task, level with the pilot. The best NEAT network before this sat around seventy two. So in one morning the starting point moved up by more than ten points.

The new run started from that copy about an hour ago. Same curriculum as before, boulder first, with four in ten training flights on the real task. One rule changed. A stage now passes when the boulder is hit at most twice in a hundred flights, instead of never. The old rule blocked checks that landed ninety six and ninety eight percent.

So where are we? About eighty five percent, with ninety eight as the goal, and four and a half days left. The remaining failures are spread out. Boulder hits, landing off the pad, and crashes, each about four or five flights in a hundred. Evolution's job now is to shave those down. And I've learned my lesson. When the network stalls, I look at what it can sense before I look at how it learns.
