I'm interested in building a game (played via web browser, with daily challenges - similar to Wordle/NYT-games, in that respect) based on finding similarities between Magic The Gathering cards, with similarity evaluated similarly to semantic-similarity for words; that is, the cards would be represented as vectors in a space of features, and "similarity" would be distance in that space. I can imagine two playstyles:
* There is a single secret "target card". The player sequentially submits guesses, and are told how similar each guess is to the target
* There are two given "target cards", and the player must construct a chain from one to the other where links can only exist between cards that have a maximum given similarity (like https://linxicon.com/)

I want you to assist me in building this game.

First, consider those play modes, and suggest some others. Figure out which mode would present the best balance of being "fun to play" vs. "easy to implement".

Next, plan out the implementation. In particular, for building the vector embedding of Magic cards, I have access to an RTX 5090 and 48Gb of RAM to work with, though I'd need your help on building the script/harness to vectorize, as well as sourcing the dataset _of_ the cards themselves.
