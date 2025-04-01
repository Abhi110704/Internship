from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

asa = ChatBot("ASA")

trainer = ChatterBotCorpusTrainer(asa)

trainer.train("chatterbot.corpus.english")

print("Training complete!")
