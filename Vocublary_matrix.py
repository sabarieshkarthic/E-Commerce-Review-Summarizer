import numpy as np

class VocabMatrix:
    def __init__(self, em_dim=16):
        self.em_dim = em_dim

        self.word_to_token = {
            "<pad>": 0,
            "<unk>": 1,
            "<start>": 2,
            "<end>": 3,

            # sentiment / common words
            "good": 4, "bad": 5, "excellent": 6, "poor": 7, "amazing": 8,
            "terrible": 9, "great": 10, "average": 11, "satisfied": 12,
            "unsatisfied": 13, "fast": 14, "slow": 15, "comfortable": 16,
            "durable": 17, "cheap": 18, "expensive": 19, "perfect": 20,
            "broken": 21, "useful": 22, "useless": 23, "love": 24,
            "hate": 25, "quality": 26, "price": 27, "worth": 28,

            # product / service words
            "delivery": 29, "service": 30, "refund": 31, "return": 32,
            "packaging": 33, "product": 34, "recommend": 35,
            "seller": 36, "issue": 37, "support": 38,

            # attributes
            "design": 39, "color": 40, "size": 41, "fit": 42,
            "battery": 43, "sound": 44, "material": 45,
            "working": 46, "damaged": 47,

            # verbs / modifiers
            "easy": 48, "difficult": 49, "value": 50,
            "expectation": 51, "function": 52,
            "and": 53, "is": 54, "from": 55, "to": 56,

            # dataset-specific words
            "connect": 57,
            "case": 58,
            "solid": 59,
            "charging": 60,
            "after": 61,
            "two": 62,
            "days": 63,
            "unworthy": 64,
            "bass": 65,
            "noise": 66,
            "cancellation": 67,
            "works": 68,
            "perfectly": 69,
            "life": 70,
            "long": 71,
            "left": 72,
            "earbud": 73,
            "stopped": 74,
            "month": 75,
            "charger": 76,
            "problem": 77,
            "defective": 78,
        }

        self.token_to_word = {v: k for k, v in self.word_to_token.items()}
        self.vocab_size = len(self.word_to_token)

        limit = np.sqrt(6.0 / (self.vocab_size + self.em_dim))
        self.embedding_matrix = np.random.uniform(
            -limit, limit, size=(self.vocab_size, self.em_dim)
        )

    def update(self, learning_rate, d_embedding_matrix):
        self.embedding_matrix -= learning_rate * d_embedding_matrix
