from ..import_package import *


class BPE(object):
    def __init__(self, vocab_size_limit=2048, load=True):
        super(BPE, self).__init__()
        self.special_tokens = {"</PAD>": 0, "</START>": 1, "</END>": 2, "</UNK>": 3, "</w>": 4}
        self.encoding_map = self.special_tokens.copy()
        self.decoding_map = {}
        self.vocab_size_limit = vocab_size_limit
        if load:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(curr_path, "bpe.json")
            self.load(path)

    def text2sentences(self, raw_text: str):
        raw_text = raw_text.lower()
        cleaned_text = re.sub(r'[\n\r\t]+', ' ', raw_text)
        sentences = re.split(r'[,.;:!?。？！]', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def __preprocess_sentences(self, word_freq):
        """
        :param word_freq: {"B":1, "A":2}
        :return:
        """
        split_words = []
        freq_arr = []
        for word, freq in word_freq.items():
            chars = list(word)
            chars[-1] = chars[-1] + "</w>"
            split_words.append(chars)
            freq_arr.append(freq)
        return split_words, freq_arr

    def __get_pair_count(self, split_words, freq_arr):
        """
        统计所有**相邻字符对**的出现频率（核心步骤）
        """
        pair_counts = defaultdict(int)
        for chars, freq in zip(split_words, freq_arr):
            if len(chars) < 2:
                continue
            # 遍历所有相邻字符对
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                pair_counts[pair] += freq
        return pair_counts

    def __merge_pair(self, split_words, pair, new_token):
        """
        合并最高频字符对：将所有出现的 (a,b) 替换为新token ab
        """
        merged_words = []
        for chars in split_words:
            new_chars = []
            i = 0
            while i < len(chars):
                # 匹配到目标字符对，执行合并
                if i < len(chars) - 1 and chars[i] == pair[0] and chars[i + 1] == pair[1]:
                    new_chars.append(new_token)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            merged_words.append(new_chars)
        return merged_words

    def __reorder_encoding_map(self):
        special = list(self.special_tokens.keys())
        train_tokens = [t for t in self.encoding_map if t not in special] + list(string.ascii_lowercase) + list(
            r",./<>?;':~!@#$%^&*()_+`1234567890-=|\\")
        train_tokens.sort(key=lambda x: (-len(x), x))
        self.encoding_map = self.special_tokens.copy()
        self.encoding_map = {token: idx for idx, token in enumerate(special + train_tokens)}
        self.decoding_map = {v: k for k, v in self.encoding_map.items()}

    def train(self, text, file_name="./bpe.json", save_step=500):
        sentences = self.text2sentences(text)
        words_arr = []
        for s in sentences:
            words_arr.extend(s.split(" "))
        word_counter = Counter(words_arr)
        print("参与训练的词数(去重):", len(word_counter))
        split_words, words_freq = self.__preprocess_sentences(word_freq=word_counter)
        while len(self.encoding_map) < self.vocab_size_limit:
            # 统计字符对频率
            pair_count = self.__get_pair_count(split_words, words_freq)
            if not pair_count:
                break  # 无字符对可合并，提前终止
            # 选取频率最高的字符对
            best_pair = max(pair_count, key=pair_count.get)
            new_token = "".join(best_pair)
            # 将新token加入词汇表
            self.encoding_map[new_token] = len(self.encoding_map)
            # 执行合并
            split_words = self.__merge_pair(split_words, best_pair, new_token)
            # 更新反向解码映射
        self.__reorder_encoding_map()  # 重排序
        self.save(file_name=file_name)
        print(f"BPE训练完成！最终词汇表大小：{len(self.encoding_map)}/{self.vocab_size_limit}")

    def __encode_word(self, word):
        """最长后缀匹配"""
        word += "</w>"
        ptr = len(word)
        tokens = []
        unk_id = self.encoding_map["</UNK>"]
        while ptr > 0:
            for i in range(ptr):
                cur = word[i:ptr]
                if cur in self.encoding_map:
                    tokens.append(self.encoding_map[cur])
                    ptr = i
                    break
            else:
                tokens.append(unk_id)
                ptr -= 1
        return tokens[::-1]

    def __encode_sentence(self, sentence: str, dim=128):
        start = self.encoding_map["</START>"]
        end = self.encoding_map["</END>"]
        words = sentence.lower().split(" ")
        tokens = []
        for word in words:
            tokens.extend(self.__encode_word(word=word))
        if len(tokens) > dim - 2:
            return [start] + tokens[:dim - 2] + [end]
        tokens.extend([self.encoding_map["</PAD>"]] * (dim - 2 - len(tokens)))
        return [start] + tokens + [end]

    def encode_sentences(self, sentences: list = ["author is linlan"], dim: int = 128, numpy=True):
        tokens = []
        for sentence in sentences:
            tokens.append(self.__encode_sentence(sentence, dim=dim))
        if numpy:
            return np.array(tokens)
        return tokens

    def decode_tokens(self, tokens):
        sentence = []
        for token in tokens:
            words = []
            for key in token:
                if key == 0:
                    continue
                words.append(self.decoding_map[key])
            sentence.append("".join(words).replace("</w>", " "))
        return sentence

    def save(self, file_name="./bpe.json"):
        with open(file_name, 'w') as f:
            json.dump(self.encoding_map, f)

    def load(self, file_name):
        with open(file_name, "r") as f:
            self.encoding_map = json.load(f)
        self.decoding_map = {v: k for k, v in self.encoding_map.items()}
