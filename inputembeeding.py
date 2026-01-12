import numpy as np
from Positional_encoding import PositionalEncoding

class InputEmbedding:
    def initializeEmbedding(self,sentence,word_to_token,embedding_matrix):
        self.word_to_token=word_to_token
        self.embedding_matrix=embedding_matrix
        tokens=sentence.lower().split()
        self.token_ids=[]
        unk_id=word_to_token["<unk>"]
        for w in tokens:
            self.token_ids.append(word_to_token.get(w,unk_id))
        self.embedded=self.embedding_matrix[self.token_ids]
        self.pos_enc=PositionalEncoding()
        self.pos_mat=self.pos_enc.Calculate(self.embedded)
        self.out=self.embedded+self.pos_mat
        return self.out

    def backpropagate(self,d_input):
        d_embedded=d_input
        vocab_size,em_dim=self.embedding_matrix.shape
        d_embedding_matrix=np.zeros_like(self.embedding_matrix)
        for idx,token_id in enumerate(self.token_ids):
            d_embedding_matrix[token_id]+=d_embedded[idx]
        return d_embedding_matrix