import numpy as np
def prepare_data_sets():
    with open("names.txt") as file:
        data = file.read().splitlines()
    n = int(0.9*len(data))
    train = data[:n]
    test = data[n:]
    return train, test 

train ,test = prepare_data_sets()
vocabulary =[]
vocabulary.append('.')
vocabulary.extend(sorted(list(set("".join(train)))))
encoding_table = {char:i for i,char in enumerate(vocabulary)}
decoding_table = {i:char for i,char in enumerate(vocabulary)}
def get_bigram_lookup_table(data,encoding_table,size):
    table = np.zeros((size,size),dtype=np.int32)
    data = ["."+word+"." for word in data]
    for word in data:
        for char1,char2 in zip(word,word[1:]):
            i1 = encoding_table.get(char1)
            i2 = encoding_table.get(char2)
            table[i1,i2] += 1
    return table 

def get_propabiblity_vector(x):
    return np.float32(x) / np.sum(x,axis=1,keepdims=True)

bigram_lookup_table = get_bigram_lookup_table(train,encoding_table,len(vocabulary))
bigram_probs = get_propabiblity_vector(bigram_lookup_table)

def generateName(bigram_probs,decoding_table,min):
    for _ in range(10):
        size = 0
        i=0
        while True:
            p = bigram_probs[i]
            next_token = np.random.choice(len(p), p=p)
            i= next_token
            if size+1 < min and i == 0:
                continue
            size += 1
            if(i == 0):
                break
            print(decoding_table[i],end="")
        print(f"{i=}")

generateName(bigram_probs,decoding_table,4)