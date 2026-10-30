# Episode 08: Why evolution kept going backwards

In the last episode, the new starting network landed about eighty five percent of the time. Since then I ran four short experiments in a row. Each one went wrong in a way that taught me something. And by the end of the day, evolution finally stopped going backwards.

The first run used the curriculum from before. The boulder in calm air first, with four out of ten flights on the real task. Within twenty five generations it dropped from eighty three percent to below sixty. The reason was the new wind sensor. In calm air it always reads zero, so the network's wiring for it could drift without anyone noticing. So I removed the easy stages and trained on the real task only.

That run also went backwards, from eighty five to about seventy two. So the curriculum wasn't the culprit. Then I noticed something odd. While the landings went down, the game's own score went up. The score rewards saving fuel and landing near the centre. Evolution had found that flying frugally paid better than landing safely. That's a reward trap. So I changed what evolution is paid for. Three hundred points for every landing between the flags, a penalty for hitting the boulder or running out of time, and the game score counts only a little.

That helped, but it still slid down, from eighty seven to about seventy five. And the average network in the population was getting worse every generation. So I measured something directly. I took the starting network, made twelve children with one round of NEAT's mutations each, and flew every child a hundred and fifty times. The parent landed eighty six percent of the time. The children averaged seventy, and one of them landed only seven percent.

The surprise was where the damage came from. Even with the tiniest weight nudges, most children were still much worse. The real wrecking ball was NEAT's other mutations. Sometimes it replaces a trained weight with a brand new random one. Sometimes it switches a connection off, or deletes it. For a tiny network that's still searching, that's fine. For a trained one, it's like pulling random wires out of a radio. With those turned off, and new neurons added only rarely, the children averaged eighty three percent, and the worst one seventy nine.

So the newest run uses gentle mutations, the landing first reward, and the real task only. And for the first time, it holds. Over forty generations, every check on a hundred fresh real task flights has landed between eighty and eighty six percent. The population's average is rising, not falling. It hasn't clearly beaten its start yet, but it's no longer losing ground.

Two more things happened today. Training got a lot faster. Seventy percent of the time had gone into the code that runs the network, which was written in plain Python. I rewrote it to use fast matrix arithmetic. Same answers, measured to the twelfth decimal place. And I found three leftover test runs that had been running in the background for up to seven and a half hours, quietly stealing computer time. Together, those changes took a generation from about a hundred and ten seconds down to under fifty.

I also made a mistake of my own. A restart command waited for a process that turned out to be itself, and the training sat stopped for about twenty five minutes. There's now a safer restart script.

So, at the end of Friday, the network holds at about eighty two percent, and evolution can finally search upwards instead of digging down. The goal is still ninety eight. Next comes patience, and a close watch on whether that line starts to climb.
