from tokenizer import Tokenizer
import regex as re

with open("taylorswift.txt", 'r') as f:
    text = f.read()
toke = Tokenizer()
toke.train(text, 276)
print(toke.decode([120]))
print(toke.decode(toke.encode("hello world!!!? (안녕하세요!) lol123 😉")))

